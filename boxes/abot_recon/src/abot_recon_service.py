"""abot_recon box - ABot-Recon streaming 3D reconstruction behind the shared envelope.

Wraps https://github.com/amap-cvlab/ABot-Recon (pinned in docker/Dockerfile):
"a deliberately local" streaming reconstruction model. At each step it caches
KV features from the preceding 11 frames, predicts the current frame's point
map in that frame's camera frame, and estimates the adjacent relative pose;
the global trajectory and world point cloud follow from sequential pose
composition. Model state never grows with sequence length - the paper reports
24 FPS at 504x280 on one H100 with 6.7 GiB - so this box is meant for long
videos, not just 16-frame clips.

Two commands, both one causal pass:

* **``reconstruct``** (default) - the full output: camera poses, adjacent
  relative poses, per-frame local point maps, the world-frame point maps,
  and confidence, with optional loop-closure trajectory refinement.
* **``trajectory``** - poses only. The cheap call: no point maps are
  requested from the model, so it is faster and the reply stays small.
* **``reset``** - standard no-op; this box keeps no per-caller state.

Contract (see the registry's contract/ and CONTRIBUTING.md):

* response ``config_json`` is namespaced ``{"abot_recon": {"status": …}}``
  with ``status`` in ``done | empty_request | error``.
* every payload is an ``np.save`` blob declared ``numpy``.
* the model loads lazily on first use and a watchdog parks it back on CPU
  after ``_IDLE_TIMEOUT`` seconds, so an idle box holds no VRAM. At ~1.0 B
  parameters that is worth giving back.

Upstream facts this file depends on:

* the public API is ``ABotRecon.from_pretrained(...)`` / ``.infer(...)`` -
  a stable, documented pair, so (unlike the d4rt box) nothing here pokes at
  private helpers.
* ``infer`` takes *file paths*, which it sorts lexicographically. Frames are
  therefore written to a scratch dir as ``frame_00000N.jpg``; the order the
  caller sent them in is the order they get named, and zero-padding makes
  the lexicographic order identical up to 10^6 frames.
* point maps come out at the model's fixed 280x504 grid, whatever the input
  resolution: the model sees a resized view.
* ``camera_poses`` are camera-to-world (upstream derives relative poses
  from them with ``relative_from_c2w``), ``relative_poses`` is one fewer
  row than ``camera_poses`` (between consecutive frames).
* the released checkpoint loads ``strict=True`` upstream - a mismatch is a
  loud failure, not a silent garbage model.
* loop closure is an add-on: DINOv2-SALAD retrieval plus pose-graph
  refinement, with its own two checkpoints. It is off by default here;
  turn it on and you need the assets (baked in with
  ``--build-arg LOOP_ASSETS=true`` or mounted at ``ABOT_RECON_LOOP_DIR``).
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
_IDLE_TIMEOUT = float(os.getenv("ABOT_RECON_IDLE_TIMEOUT", "120"))

BOX_KEY = "abot_recon"

#: Where the Dockerfile put the weights and (optionally) the loop assets.
_CKPT = os.path.abspath(os.getenv("ABOT_RECON_CKPT", "/weights/abot_recon.safetensors"))
_LOOP_DIR = os.path.abspath(os.getenv("ABOT_RECON_LOOP_DIR", "/weights/loop"))
_ATTENTION = os.getenv("ABOT_RECON_ATTENTION_BACKEND", "auto").strip()
_AMP_DTYPE = os.getenv("ABOT_RECON_AMP_DTYPE", "bf16").strip()
#: Upstream's own default stream cap (InferenceConfig.max_frames).
_MAX_STREAM = int(os.getenv("ABOT_RECON_MAX_STREAM", "22000"))
#: The model needs real context before geometry is meaningful; this is also
#: the shortest input its own integration test uses.
_MIN_FRAMES = 3

_DEFAULTS = {
    "max_frames": 256,        # cap on accepted frames (upstream allows up to 22000)
    "frame_step": 1,          # video decoding: keep every Nth frame
    "dense_stride": 1,        # save dense maps every Nth frame; poses stay per-frame
    "loop_closure": False,    # trajectory refinement (needs the loop assets)
    "output_local_points": True,
    "output_world_points": True,
    "output_confidence": True,
    "confidence_threshold": 0.0,
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


def _num(parameters, key, cast):
    v = parameters.get(key, _DEFAULTS.get(key))
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
    """A video file -> a [T, H, W, 3] uint8 RGB array."""
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


def _write_frames(directory, frames):
    """Frames -> files, in the order given.

    Upstream sorts its input paths lexicographically, so the *names* encode
    the order: zero-padded and sequential, which keeps lexicographic and
    temporal order identical up to a million frames.
    """
    names = []
    for i, rgb in enumerate(frames):
        name = f"frame_{i:06d}.jpg"
        path = os.path.join(directory, name)
        if not cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise ValueError(f"could not write frame {i + 1} to scratch storage")
        # Absolute: abot.infer() opens these relative to its own working dir,
        # not to the scratch dir we wrote them into. The zero-padded name also
        # keeps upstream's lexicographic sort identical to the temporal order.
        names.append(os.path.abspath(path))
    return names


# ------------------------------------------------------------------ model
class _Model:
    """Lazy load and the idle watchdog.

    Held separately from the servicer so the watchdog has one thing to lock.
    The model is built once - device, precision and attention backend are
    deployment-level choices (env vars in the image), not per-request ones;
    per-request knobs go through ``infer()`` instead.
    """

    def __init__(self):
        self.abot = None
        self.placed = None
        self.lock = threading.Lock()
        self.last_used = time.time()
        threading.Thread(target=self._watchdog, daemon=True).start()

    def _watchdog(self):
        while True:
            time.sleep(10)
            with self.lock:
                idle = time.time() - self.last_used
                if (idle > _IDLE_TIMEOUT and self.abot is not None
                        and self.placed and self.placed.startswith("cuda")):
                    logging.info("idle %.0fs: parking the model on CPU", idle)
                    self.abot.model.to("cpu")
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
            if self.abot is None:
                self._load(device)
            if self.placed != device:
                logging.info("moving the model to %s", device)
                self.abot.model.to(device)
                self.placed = device
            return self.abot

    def _check_checkpoint(self):
        if not os.path.isfile(_CKPT) and not os.path.isdir(os.path.dirname(_CKPT)):
            return
        path = _CKPT if os.path.isfile(_CKPT) else \
            os.path.join(os.path.dirname(_CKPT), "abot_recon.safetensors")
        if not os.path.isfile(path):
            raise RuntimeError(
                f"{path} is missing. The image should carry the checkpoint; "
                f"set ABOT_RECON_CKPT, or mount it there (build with "
                f"--build-arg DOWNLOAD_WEIGHTS=false to skip the download).")

    def _loop_assets(self):
        return _loop_assets()

    def _load(self, device):
        from abot_recon import ABotRecon

        self._check_checkpoint()
        if _AMP_DTYPE not in ("fp32", "fp16", "bf16"):
            raise RuntimeError(f"ABOT_RECON_AMP_DTYPE={_AMP_DTYPE!r} is not "
                               f"one of fp32, fp16, bf16")
        if _ATTENTION not in ("auto", "paged", "sdpa"):
            raise RuntimeError(f"ABOT_RECON_ATTENTION_BACKEND={_ATTENTION!r} is not "
                               f"one of auto, paged, sdpa")

        salad, dino = self._loop_assets()
        if salad is None:
            logging.info("loop-closure assets not present in %s; "
                         "loop_closure=true will fail until they are", _LOOP_DIR)

        overrides = {
            "device": device,
            "amp_dtype": _AMP_DTYPE,
            "attention_backend": _ATTENTION,
            "max_frames": _MAX_STREAM,
            "loop_closure": False,
            "output_local_points": True,
            "output_world_points": True,
            "output_confidence": True,
        }
        if salad is not None and dino is not None:
            overrides["loop_salad_checkpoint"] = salad
            overrides["loop_dino_checkpoint"] = dino

        logging.info("loading ABot-Recon (checkpoint %s, %s, %s)",
                     _CKPT, _AMP_DTYPE, _ATTENTION)
        t0 = time.time()
        self.abot = ABotRecon.from_pretrained(_CKPT,
                                              local_files_only=True, **overrides)
        self.abot.model.eval()
        self.placed = device
        params = sum(p.numel() for p in self.abot.model.parameters()) / 1e6
        logging.info("loaded in %.1fs: %.2f M parameters, on %s",
                     time.time() - t0, params, device)


_MODEL = _Model()


# -------------------------------------------------------------- inference
def _loop_assets():
    """The loop-closure retrieval assets, or None for the missing ones."""
    salad = os.path.join(_LOOP_DIR, "dino_salad.ckpt")
    dino = os.path.join(_LOOP_DIR, "dinov2_vitb14_pretrain.pth")
    return (salad if os.path.isfile(salad) else None,
            dino if os.path.isfile(dino) else None)


def _to_np(value):
    return value.detach().cpu().numpy()


def _loop_assets_error():
    salad, dino = _loop_assets()
    missing = [n for n, p in (("dino_salad.ckpt", salad),
                              ("dinov2_vitb14_pretrain.pth", dino)) if p is None]
    return ("loop closure was requested but the retrieval assets are not in "
            f"{_LOOP_DIR} (missing: {', '.join(missing)}). Build the image "
            "with --build-arg LOOP_ASSETS=true, or mount both files there.")


def _run_recon(abot, paths, command, parameters):
    """One causal pass. Returns (arrays, extra) for the reply."""
    t = len(paths)
    loop = _flag(parameters, "loop_closure", False)
    if loop:
        salad, dino = _loop_assets()
        if salad is None or dino is None:
            raise ValueError(_loop_assets_error())

    stride = _num(parameters, "dense_stride", int)
    if stride < 1:
        raise ValueError(f"parameters.dense_stride must be at least 1, got {stride}")
    dense = list(range(0, t, stride)) if stride > 1 else None

    threshold = _num(parameters, "confidence_threshold", float)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"parameters.confidence_threshold must be in [0, 1], "
                         f"got {threshold}")

    with_points = (command == "reconstruct")
    local_on = _flag(parameters, "output_local_points", True) if with_points else False
    world_on = _flag(parameters, "output_world_points", True) if with_points else False
    conf_on = _flag(parameters, "output_confidence", True) if with_points else False

    res = abot.infer(
        paths,
        output_local_points=local_on,
        output_world_points=world_on,
        output_confidence=conf_on,
        confidence_threshold=threshold,
        loop_closure=loop,
        dense_output_indices=dense,
    )

    arrays = {
        "camera_poses": _to_np(res.camera_poses),
        "relative_poses": _to_np(res.relative_poses),
        "camera_poses_noloop": _to_np(res.camera_poses_noloop),
        "relative_poses_noloop": _to_np(res.relative_poses_noloop),
    }
    if res.camera_poses_loop is not None:
        arrays["camera_poses_loop"] = _to_np(res.camera_poses_loop)
        arrays["relative_poses_loop"] = _to_np(res.relative_poses_loop)
    if res.local_points is not None:
        arrays["local_points"] = _to_np(res.local_points)
    if res.world_points is not None:
        arrays["world_points"] = _to_np(res.world_points)
    if res.confidence is not None:
        arrays["confidence"] = _to_np(res.confidence)
        if res.confidence_mask is not None:
            arrays["confidence_mask"] = _to_np(res.confidence_mask).astype(bool)

    metadata = res.metadata
    if not isinstance(metadata, dict):
        metadata = {}
    safe_metadata = {}
    for k, v in metadata.items():
        if isinstance(v, (bool, int, float, str)):
            safe_metadata[k] = v
        elif isinstance(v, (list, tuple)):
            safe_metadata[k] = [x if isinstance(x, (bool, int, float, str)) else str(x)
                                for x in v]
        else:
            safe_metadata[k] = str(v)

    extra = {
        "num_frames": t,
        "dense_stride": stride,
        "num_dense_frames": t if dense is None else len(dense),
        "loop_closure_used": loop,
        "local_points_requested": local_on,
        "world_points_requested": world_on,
        "confidence_requested": conf_on,
        "confidence_threshold": threshold,
        "metadata": safe_metadata,
    }
    return arrays, extra


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
            command = section.get("command") or "reconstruct"
            parameters = section.get("parameters", {})
            if not isinstance(parameters, dict):
                return _error("parameters must be an object")

            if command == "reset":
                return _done({"action": "reset"})
            if command not in ("reconstruct", "trajectory"):
                return _error(f"unknown command {command!r} (expected "
                              f"'reconstruct', 'trajectory' or 'reset')")

            # ---- input --------------------------------------------------
            has_images = "images" in request.data
            has_video = "video" in request.data
            if has_images and has_video:
                return _error("images and video are mutually exclusive")

            frame_step = _num(parameters, "frame_step", int)
            if frame_step < 1:
                return _error(f"parameters.frame_step must be at least 1, "
                              f"got {frame_step}")
            max_frames = _num(parameters, "max_frames", int)
            if max_frames < _MIN_FRAMES:
                return _error(f"parameters.max_frames must be at least "
                              f"{_MIN_FRAMES}, got {max_frames}")
            if max_frames > _MAX_STREAM:
                return _error(f"parameters.max_frames is {max_frames}; the model "
                              f"supports at most {_MAX_STREAM} frames")

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

            if frames.shape[0] < _MIN_FRAMES:
                return _error(f"need at least {_MIN_FRAMES} frames, "
                              f"got {frames.shape[0]}")

            # ---- inference ----------------------------------------------
            device = str(parameters.get("device") or torch_device_default())
            if not (device.startswith("cuda") or device.startswith("cpu")):
                return _error(f"parameters.device must be 'cuda', 'cuda:N' or "
                              f"'cpu', got {device!r}")
            abot = _MODEL.get(device)

            with tempfile.TemporaryDirectory(prefix="abot_recon_") as scratch:
                names = _write_frames(scratch, frames)
                arrays, extra = _run_recon(abot, names, command, parameters)

            data = {k: wrap_value(np_to_bytes(v)) for k, v in arrays.items()}
            encoding = {k: "numpy" for k in data}
            return _done({**extra,
                          "command": command,
                          "device": device,
                          "model_hw": [280, 504],
                          "input_hw": [int(frames.shape[1]), int(frames.shape[2])],
                          "runtime": time.time() - start},
                         data=data, encoding=encoding)
        except Exception as e:                              # noqa: BLE001
            logging.exception("Error in Process")
            return _error(str(e))


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
        futures.ThreadPoolExecutor(max_workers=serve.max_workers(2)),
        options=serve.server_options(),
        maximum_concurrent_rpcs=serve.max_concurrent_rpcs())
    folder_wd_pb2_grpc.add_PipelineServiceServicer_to_server(
        PipelineService(), server)

    grpc_reflection.enable_server_reflection(
        (folder_wd_pb2.DESCRIPTOR.services_by_name["PipelineService"].full_name,
         grpc_reflection.SERVICE_NAME), server)

    logging.info(serve.describe(2))
    run_server(server)
