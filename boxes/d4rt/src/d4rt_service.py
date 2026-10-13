"""d4rt box - OpenD4RT 4D reconstruction and tracking behind the shared envelope.

Wraps https://github.com/Lijiaxin0111/Open-d4rt (pinned in docker/Dockerfile):
a feedforward video transformer that answers one kind of question - "where is
the point at pixel (u, v) of frame t_src, at frame t_tgt, in the camera frame
of t_cam?" - and builds tracking, point clouds and camera poses out of it.

Three commands, all driven by that same query interface:

* **``track``** (default) - follow query points through the clip. Returns each
  point's 3D position per frame, in two frames of reference, plus visibility.
* **``reconstruct``** - sweep a uv grid to get a per-frame point cloud in the
  coordinates of frame 0.
* **``cameras``** - per-frame intrinsics and extrinsics, recovered from the
  same queries by least squares and Umeyama alignment.
* **``reset``** - standard no-op; this box keeps no per-caller state.

Contract (see the registry's contract/ and CONTRIBUTING.md):

* response ``config_json`` is namespaced ``{"d4rt": {"status": …}}`` with
  ``status`` in ``done | empty_request | error``.
* every payload is an ``np.save`` blob declared ``numpy``.
* the model loads lazily on first use and a watchdog parks it back on CPU
  after ``_IDLE_TIMEOUT`` seconds, so an idle box holds no VRAM. At ~1.16B
  parameters in fp32 that is about 4.6 GB, which is worth giving back.

Upstream quirks that shape this file:

* the repo has no package metadata and its imports are repo-root relative, so
  ``D4RT_REPO`` goes on ``sys.path``.
* the checkpoint is loaded with ``strict=False`` upstream, which turns a
  mismatched config into a silently garbage model. We load the same way and
  then refuse to serve if any key was missing or unexpected.
* ``clip_frames`` (32 or 48 depending on the checkpoint) is the most frames
  the model can see at once, and query timesteps are silently clamped to it
  upstream. Longer videos are handled by the upstream clip-anchoring logic in
  ``_infer_tracks``; for ``reconstruct`` and ``cameras`` the same applies via
  their own helpers.
"""

import concurrent.futures as futures
import json
import io
import logging
import os
import sys
import tempfile
import threading
import time

sys.path.append("./protos")
import pipeline_pb2 as folder_wd_pb2  # noqa: E402
import pipeline_pb2_grpc as folder_wd_pb2_grpc  # noqa: E402
from aux import wrap_value, unwrap_value
import serve  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_PORT_ENV_VAR = "PORT"
_IDLE_TIMEOUT = float(os.getenv("D4RT_IDLE_TIMEOUT", "120"))

BOX_KEY = "d4rt"

#: Where the Dockerfile put the upstream checkout and the weights.
_REPO_DIR = os.getenv("D4RT_REPO", "/opt/opend4rt")
_CKPT_DIR = os.getenv("D4RT_CKPT_DIR", "/weights")
_VARIANT = os.getenv("D4RT_VARIANT", "OpenD4RT_48CLIP_9Mix_NoCropAUG")

_DEFAULTS = {
    "grid_size": 32,          # track: uv grid when no explicit points are given
    "point_grid_size": 64,    # reconstruct: uv grid side
    "max_points": 4096,       # reconstruct: cap after gridding
    "camera_grid_size": 16,   # cameras: coarse grid for the least-squares fit
    "chunk_size": 4096,       # queries per decoder call
    "t_src": 0,               # track: frame the query pixels are read from
    "frame_step": 1,          # video decoding
    "max_frames": 48,         # video decoding; also the model's clip limit
}


# ---------------------------------------------------------------- envelope
def _done(extra, data=None, encoding=None):
    section = {"status": "done", **extra}
    if encoding:
        section["encoding"] = encoding
    return folder_wd_pb2.Envelope(
        config_json=json.dumps({BOX_KEY: section}), data=data or {})


def _empty_request():
    return folder_wd_pb2.Envelope(
        config_json=json.dumps({BOX_KEY: {"status": "empty_request"}}))


def _error(message):
    return folder_wd_pb2.Envelope(
        config_json=json.dumps({BOX_KEY: {"status": "error", "error": message}}))


def np_to_bytes(arr) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr))
    return buf.getvalue()


def _num(parameters, key, cast, default=None):
    v = parameters.get(key, _DEFAULTS.get(key) if default is None else default)
    try:
        return cast(v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"parameters.{key} must be a number, got {v!r}") from e


def _flag(parameters, key, default):
    v = parameters.get(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off", ""):
            return False
    raise ValueError(f"parameters.{key} must be a boolean, got {v!r}")


def torch_device_default():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ------------------------------------------------------------------ input
def _decode_images(blobs):
    """JPEG/PNG bytes -> a [T, H, W, 3] uint8 RGB array."""
    frames = []
    for i, raw in enumerate(blobs):
        arr = np.frombuffer(bytes(raw), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"could not decode image #{i + 1} of {len(blobs)}")
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    shapes = {f.shape for f in frames}
    if len(shapes) > 1:
        raise ValueError(f"all frames must have the same size, got {sorted(shapes)}")
    return np.stack(frames, axis=0)


def _decode_video(blob, frame_step, max_frames):
    """A video file -> a [T, H, W, 3] uint8 RGB array.

    Upstream ships a loader for this but it references ``Path`` without
    importing it, so calling it raises NameError; this is the same job done
    here.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as fh:
        fh.write(bytes(blob))
        path = fh.name
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError("could not open the video (unsupported container?)")
        frames, index = [], 0
        while len(frames) < max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            if index % max(1, frame_step) == 0:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            index += 1
        cap.release()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if not frames:
        raise ValueError("the video decoded to zero frames")
    return np.stack(frames, axis=0)


def _uv_grid(side, margin=0.05):
    """A normalised uv grid in [0, 1], inset by ``margin`` from the border."""
    side = max(2, int(side))
    axis = np.linspace(margin, 1.0 - margin, side, dtype=np.float32)
    uu, vv = np.meshgrid(axis, axis, indexing="xy")
    return np.stack([uu.ravel(), vv.ravel()], axis=1).astype(np.float32)


def _points_from_request(values, num_frames):
    """``data.points`` is a flat [u0, v0, u1, v1, …] float list in [0, 1]."""
    flat = np.asarray(list(values), dtype=np.float32)
    if flat.size % 2 != 0:
        raise ValueError(f"data.points has {flat.size} values; it must be "
                         f"pairs of normalised (u, v)")
    uv = flat.reshape(-1, 2)
    if uv.size and (uv.min() < -0.001 or uv.max() > 1.001):
        raise ValueError("data.points must be normalised to [0, 1] "
                         f"(got {uv.min():.3f}..{uv.max():.3f})")
    return np.clip(uv, 0.0, 1.0)


# ------------------------------------------------------------------ model
class _Model:
    """Lazy load, device placement, and the idle watchdog.

    Held separately from the servicer so the watchdog has one thing to lock.
    """

    def __init__(self):
        self.model = None
        self.clip_frames = None
        self.image_hw = (256, 256)
        self.placed = None
        self.lock = threading.Lock()
        self.last_used = time.time()
        threading.Thread(target=self._watchdog, daemon=True).start()

    def _watchdog(self):
        while True:
            time.sleep(10)
            with self.lock:
                idle = time.time() - self.last_used
                if (idle > _IDLE_TIMEOUT and self.model is not None
                        and self.placed and self.placed.startswith("cuda")):
                    logging.info("idle %.0fs: parking the model on CPU", idle)
                    self.model.to("cpu")
                    self.placed = "cpu"
            try:
                import torch
                torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass

    def get(self, device):
        """The model, loaded and on ``device``."""
        with self.lock:
            self.last_used = time.time()
            if self.model is None:
                self._load()
            if self.placed != device:
                logging.info("moving the model to %s", device)
                self.model.to(device)
                self.placed = device
            return self.model, self.clip_frames, self.image_hw

    def _load(self):
        # Upstream imports are repo-root relative and there is no package
        # metadata, so the checkout has to be on sys.path.
        if _REPO_DIR not in sys.path:
            sys.path.insert(0, _REPO_DIR)

        from src.core import load_yaml_config, load_checkpoint
        from src.model import build_model
        from infer_track_3d import _unwrap_state_dict

        config_path = os.path.join(_CKPT_DIR, _VARIANT, "model.yaml")
        ckpt_path = os.path.join(_CKPT_DIR, _VARIANT, "opend4rt.ckpt")
        for path in (config_path, ckpt_path):
            if not os.path.isfile(path):
                raise RuntimeError(
                    f"{path} is missing. The image should carry the weights; "
                    f"set D4RT_CKPT_DIR/D4RT_VARIANT, or mount them there.")

        logging.info("loading %s", _VARIANT)
        t0 = time.time()
        cfg = load_yaml_config(config_path)
        model = build_model(cfg["model"]).eval()
        payload = load_checkpoint(ckpt_path, map_location="cpu")
        result = model.load_state_dict(_unwrap_state_dict(payload), strict=False)

        # Upstream loads non-strict, so a config/checkpoint mismatch produces a
        # model that runs and returns nonsense. Refuse instead.
        missing = list(getattr(result, "missing_keys", []))
        unexpected = list(getattr(result, "unexpected_keys", []))
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint does not match the config: {len(missing)} missing "
                f"and {len(unexpected)} unexpected key(s). "
                f"First missing: {missing[:3]}; first unexpected: {unexpected[:3]}")

        self.model = model
        self.placed = "cpu"
        self.clip_frames = int(getattr(model.query_embedder, "max_frames", 48))
        size = cfg.get_path("model.input.image_size", [256, 256]) \
            if hasattr(cfg, "get_path") else [256, 256]
        self.image_hw = (int(size[0]), int(size[1]))
        logging.info("loaded in %.1fs: clip_frames=%d image_hw=%s",
                     time.time() - t0, self.clip_frames, self.image_hw)


_MODEL = _Model()


# -------------------------------------------------------------- inference
def _prepare(frames, image_hw):
    """Resize to what the checkpoint was trained at, keeping the true aspect."""
    from infer_track_3d import _resize_video

    native_aspect = float(frames.shape[2]) / float(max(1, frames.shape[1]))
    return _resize_video(frames, image_hw=image_hw), native_aspect


def _run_track(model, frames, aspect, uv, chunk, t_src):
    from infer_track_3d import _infer_tracks

    src = None
    if t_src:
        src = np.full((uv.shape[0],), int(t_src), dtype=np.int64)
    return _infer_tracks(model=model, video_model_rgb=frames,
                         native_aspect_ratio=aspect, query_uv_norm=uv,
                         query_chunk_size=chunk,
                         query_src_indices_global=src)


def _run_reconstruct(model, frames, aspect, uv, chunk):
    from vis.build_like_demo import _infer_point_cloud_ref0

    xyz, vis, conf, _ = _infer_point_cloud_ref0(
        model=model, video_model_rgb=frames, native_aspect_ratio=aspect,
        point_query_uv_norm=uv, query_chunk_size=chunk,
        umeyama_slide_window=False)
    return xyz, vis, conf


def _run_cameras(model, frames, image_hw, grid, chunk, intrinsics, extrinsics):
    from vis.build_like_demo import _predict_camera_branches

    return _predict_camera_branches(
        model=model, video_model_rgb=frames, image_hw=image_hw,
        camera_grid_size=grid, camera_query_chunk_size=chunk,
        predict_intrinsics=intrinsics, predict_extrinsics=extrinsics)


# ---------------------------------------------------------------- service
class PipelineService(folder_wd_pb2_grpc.PipelineServiceServicer):

    def Process(self, request, context):
        start = time.time()
        try:
            if not request.config_json:
                return _error("No config JSON")

            config = json.loads(request.config_json)
            if not isinstance(config, dict) or not isinstance(config.get(BOX_KEY), dict):
                return _error(f"config section {BOX_KEY!r} missing or not an object")
            section = config[BOX_KEY]
            command = section.get("command") or "track"
            parameters = section.get("parameters", {})
            if not isinstance(parameters, dict):
                return _error("parameters must be an object")

            if command == "reset":
                return _done({"action": "reset"})
            if command not in ("track", "reconstruct", "cameras"):
                return _error(f"unknown command {command!r} (expected "
                              f"'track', 'reconstruct', 'cameras' or 'reset')")

            # ---- input ------------------------------------------------
            has_images = "images" in request.data
            has_video = "video" in request.data
            if has_images and has_video:
                return _error("images and video are mutually exclusive")

            frame_step = _num(parameters, "frame_step", int)
            max_frames = _num(parameters, "max_frames", int)
            if max_frames < 2:
                return _error("parameters.max_frames must be at least 2")

            if has_video:
                frames = _decode_video(unwrap_value(request.data["video"]),
                                       frame_step, max_frames)
            elif has_images:
                blobs = unwrap_value(request.data["images"])
                if isinstance(blobs, (bytes, bytearray)):
                    blobs = [blobs]
                if not blobs:
                    return _empty_request()
                frames = _decode_images(blobs)[::max(1, frame_step)][:max_frames]
            else:
                return _empty_request()

            if frames.shape[0] < 2:
                return _error(f"need at least 2 frames, got {frames.shape[0]}")

            device = str(parameters.get("device") or torch_device_default())
            model, clip_frames, image_hw = _MODEL.get(device)
            chunk = _num(parameters, "chunk_size", int)
            resized, aspect = _prepare(frames, image_hw)

            points = unwrap_value(request.data["points"]) \
                if "points" in request.data else None

            import torch
            with torch.no_grad():
                if command == "track":
                    out = self._track(parameters, model, resized, aspect, chunk,
                                      points, clip_frames)
                elif command == "reconstruct":
                    out = self._reconstruct(parameters, model, resized, aspect, chunk)
                else:
                    out = self._cameras(parameters, model, resized, image_hw, chunk)

            arrays, extra = out
            data = {k: wrap_value(np_to_bytes(v)) for k, v in arrays.items()}
            encoding = {k: "numpy" for k in data}
            return _done({**extra,
                          "command": command,
                          "device": device,
                          "variant": _VARIANT,
                          "num_frames": int(frames.shape[0]),
                          "clip_frames": int(clip_frames),
                          "input_hw": [int(frames.shape[1]), int(frames.shape[2])],
                          "model_hw": [int(image_hw[0]), int(image_hw[1])],
                          "runtime": time.time() - start},
                         data=data, encoding=encoding)
        except Exception as e:                              # noqa: BLE001
            logging.exception("Error in Process")
            return _error(str(e))

    # ------------------------------------------------------------ commands
    def _track(self, parameters, model, frames, aspect, chunk, points, clip_frames):
        # Query points come as data.points (an ff FloatList - the idiomatic
        # place for a payload) or, for convenience when hand-writing a config,
        # as parameters.points. A grid is the fallback.
        if points is not None:
            uv = _points_from_request(points, frames.shape[0])
            source = "data.points"
        elif "points" in parameters:
            uv = _points_from_request(parameters["points"], frames.shape[0])
            source = "parameters.points"
        else:
            side = _num(parameters, "grid_size", int)
            uv = _uv_grid(side)
            source = f"grid {side}x{side}"
        if uv.shape[0] == 0:
            raise ValueError("no query points")

        t_src = _num(parameters, "t_src", int)
        if not 0 <= t_src < frames.shape[0]:
            raise ValueError(f"parameters.t_src must be in [0, {frames.shape[0] - 1}], "
                             f"got {t_src}")

        res = _run_track(model, frames, aspect, uv, chunk, t_src)
        arrays = {
            "query_uv": uv.astype(np.float32),
            "tracks_xyz_local": np.asarray(res["tracks_xyz_local"], np.float32),
            "tracks_xyz_ref0": np.asarray(res["tracks_xyz_ref0"], np.float32),
            "tracks_uv": np.asarray(res["tracks_uv_norm"], np.float32),
            "tracks_visibility": np.asarray(res["tracks_visibility"], bool),
            "tracks_confidence": np.asarray(res["tracks_confidence"], np.float32),
        }
        return arrays, {"num_queries": int(uv.shape[0]), "query_source": source,
                        "t_src": t_src}

    def _reconstruct(self, parameters, model, frames, aspect, chunk):
        side = _num(parameters, "point_grid_size", int)
        cap = _num(parameters, "max_points", int)
        uv = _uv_grid(side)
        if uv.shape[0] > cap:
            keep = np.linspace(0, uv.shape[0] - 1, cap).astype(int)
            uv = uv[keep]
        xyz, vis, conf = _run_reconstruct(model, frames, aspect, uv, chunk)
        arrays = {
            "query_uv": uv.astype(np.float32),
            "points_xyz": np.asarray(xyz, np.float32),
            "points_visibility": np.asarray(vis, bool),
            "points_confidence": np.asarray(conf, np.float32),
        }
        return arrays, {"num_points": int(uv.shape[0]), "grid_size": side}

    def _cameras(self, parameters, model, frames, image_hw, chunk):
        grid = _num(parameters, "camera_grid_size", int)
        want_k = _flag(parameters, "intrinsics", True)
        want_t = _flag(parameters, "extrinsics", True)
        if not want_k and not want_t:
            raise ValueError("cameras needs at least one of intrinsics / extrinsics")

        res = _run_cameras(model, frames, image_hw, grid, chunk, want_k, want_t)
        if res is None:
            raise RuntimeError("the camera branch returned nothing")
        arrays = {
            "intrinsics": np.asarray(res["K"], np.float32),
            "extrinsics": np.asarray(res["T_ref0_cam"], np.float32),
            "valid_intrinsics": np.asarray(res["valid_intrinsics"], bool),
            "valid_extrinsics": np.asarray(res["valid_extrinsics"], bool),
        }
        return arrays, {"camera_grid_size": grid,
                        "intrinsics_requested": want_k,
                        "extrinsics_requested": want_t}


# ----------------------------------------------------------- server setup
def get_port():
    try:
        port = int(os.getenv(_PORT_ENV_VAR, _PORT_DEFAULT))
        return port if port > 0 else None
    except ValueError:
        logging.exception("Invalid port value")
        return None


def run_server(server):
    port = get_port()
    if not port:
        return
    target = f"[::]:{port}"
    server.add_insecure_port(target)
    server.start()
    logging.info(f"Server started at {target}")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    import grpc
    import grpc_reflection.v1alpha.reflection as grpc_reflection

    logging.basicConfig(
        format="[ %(levelname)s ] %(asctime)s (%(module)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=serve.max_workers(4)),
        options=serve.server_options(),
        maximum_concurrent_rpcs=serve.max_concurrent_rpcs())
    folder_wd_pb2_grpc.add_PipelineServiceServicer_to_server(
        PipelineService(), server)

    grpc_reflection.enable_server_reflection(
        (folder_wd_pb2.DESCRIPTOR.services_by_name["PipelineService"].full_name,
         grpc_reflection.SERVICE_NAME), server)

    run_server(server)
