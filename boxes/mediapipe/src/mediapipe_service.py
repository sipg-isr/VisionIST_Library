"""mediapipe box — Google MediaPipe landmark tasks behind the shared envelope.

A standard shared-envelope box (one RPC: ``Process``) with four commands in
the ``config_json`` section ``{"mediapipe": {"command": ..., "parameters": {...}}}``:

* **``hands``** (the default) — 21 hand-knuckle landmarks per hand, plus the
  metric ``world_landmarks``, the left/right label and its score.
* **``pose``** — 33 body landmarks per person, plus ``world_landmarks`` and a
  per-landmark ``visibility``.  ``parameters.model`` picks the lite / full /
  heavy bundle.
* **``face``** — the 478-point face mesh, optionally the 52 blendshape scores
  and the 4x4 canonical-face transformation matrix.
* **``reset``** — closes the cached landmarkers.  Every box accepts it; here it
  really does free something, since a landmarker is built lazily on first use
  and kept for the next request.

Contract (see docs/gRPC_Services_Reference.md):

* the response ``config_json`` is **namespaced**: ``{"mediapipe": {"status": …}}``
  with ``status`` in ``done | empty_request | error``;
* every payload declares its codec in ``"encoding"`` — the arrays are
  ``np.save`` blobs declared ``numpy``, the blendshape names are ``json``;
* ``landmarks`` is **padded** to the largest detection count in the batch, so
  ``counts`` (int32, one per image) is the only honest way to know how many
  rows of it are real;
* coordinates are MediaPipe's own: x, y normalized to [0, 1] and z on roughly
  the same scale as x, as (x, y, z) — *not* the (y, x) tapnext reports.
  ``parameters.coordinates = "pixels"`` converts them.

Models: one ``.task`` bundle per command, baked into the image at build time
under ``MEDIAPIPE_MODEL_DIR`` (default ``/workspace/models``, falling back to
``<box>/models`` when the service is run from a checkout).  The box never
downloads anything at runtime and needs no network.

CPU only: MediaPipe's GPU delegate targets Android, iOS and the web, not a
Linux server, so there is no device parameter and no GPU watchdog here.
"""

import concurrent.futures as futures
import io
import json
import logging
import os
import pathlib
import sys
import threading
import time

sys.path.append("./protos")
import pipeline_pb2 as folder_wd_pb2  # noqa: E402
import pipeline_pb2_grpc as folder_wd_pb2_grpc  # noqa: E402
from aux import wrap_value, unwrap_value  # noqa: E402
import serve  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_PORT_ENV_VAR = "PORT"

#: The config section this box answers to. Must equal `key` in box.yaml.
BOX_KEY = "mediapipe"

#: Where the .task bundles live. The Dockerfile puts them in /workspace/models;
#: a checkout gets them from `python3 tools/fetch_assets.py mediapipe`, which
#: writes boxes/mediapipe/models/.
_MODEL_DIR_ENV = "MEDIAPIPE_MODEL_DIR"
_MODEL_DIR_DEFAULT = "/workspace/models"

#: Landmark counts are fixed by the bundles, and a client needs them to size a
#: buffer before it has seen a reply, so they are part of the contract.
LANDMARKS = {"hands": 21, "pose": 33, "face": 478}
N_BLENDSHAPES = 52

_DEFAULTS = {
    "max_results": None,          # per-command: 2 hands, 1 pose, 1 face
    "model": "lite",              # pose only
    "coordinates": "normalized",  # normalized | pixels
    "min_detection_confidence": 0.5,
    "min_presence_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "blendshapes": False,         # face only
    "transformation_matrix": False,  # face only
}
_MAX_RESULTS_DEFAULT = {"hands": 2, "pose": 1, "face": 1}
_POSE_MODELS = ("lite", "full", "heavy")


def model_dir() -> pathlib.Path:
    """The directory the .task bundles were baked into."""
    env = os.getenv(_MODEL_DIR_ENV)
    if env:
        return pathlib.Path(env)
    packaged = pathlib.Path(_MODEL_DIR_DEFAULT)
    if packaged.is_dir():
        return packaged
    # Running from a checkout: boxes/mediapipe/src/ -> boxes/mediapipe/models/
    return pathlib.Path(__file__).resolve().parent.parent / "models"


def model_path(command: str, pose_model: str = "lite") -> pathlib.Path:
    if command == "hands":
        return model_dir() / "hand_landmarker.task"
    if command == "pose":
        return model_dir() / f"pose_landmarker_{pose_model}.task"
    if command == "face":
        return model_dir() / "face_landmarker.task"
    raise ValueError(f"no model bundle for command {command!r}")


# ---------------------------------------------
# Envelope helpers
# ---------------------------------------------
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


def load_error(path, exc):
    """Why a landmarker would not build — as a ValueError a caller can act on.

    Two very different failures arrive here with equally unhelpful upstream
    text, and telling them apart is the whole point:

    * a missing system library — ``libmediapipe.so`` has DT_NEEDED entries for
      ``libEGL.so.1`` and ``libGLESv2.so.2`` even on a CPU-only box, so an
      image built without libegl1/libgles2 fails before it ever opens the
      bundle. That is a Dockerfile problem, not a model problem, and the 0.1.0
      image shipped with it;
    * a truncated or wrong-format bundle — MediaPipe says "Unable to open zip
      archive.", naming nothing.
    """
    message = str(exc)
    if "cannot open shared object file" in message or "undefined symbol" in message:
        return ValueError(
            f"MediaPipe's native library could not be loaded ({message}). This is a "
            f"missing system library in the image, not a problem with {path.name}: "
            f"libmediapipe.so needs libEGL.so.1 and libGLESv2.so.2 even with no GPU. "
            f"Rebuild the box (its Dockerfile installs libegl1 and libgles2), or "
            f"`apt-get install -y libegl1 libgles2` inside the running container to "
            f"confirm it before rebuilding")
    return ValueError(
        f"MediaPipe could not load {path.name} from {path.parent} ({message}) — the "
        f"bundle looks truncated or is not a .task archive; re-fetch it with "
        f"docker/fetch_models.py or tools/fetch_assets.py mediapipe")


def np_to_bytes(arr: np.ndarray) -> bytes:
    """np.save keeps dtype and shape in the blob, so the client's `numpy`
    codec restores the array exactly."""
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr))
    return buf.getvalue()


# ---------------------------------------------
# Parameters
# ---------------------------------------------
def _num(parameters, key, default, cast, lo=None, hi=None):
    v = parameters.get(key, default)
    if v is None:
        v = default
    try:
        v = cast(v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"parameters.{key} must be a number, got {v!r}") from e
    if lo is not None and v < lo or hi is not None and v > hi:
        raise ValueError(f"parameters.{key} must be in [{lo}, {hi}], got {v!r}")
    return v


def _flag(parameters, key, default):
    v = parameters.get(key, default)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def parse_parameters(command, parameters):
    """-> the option dict the landmarker cache is keyed on. Raises ValueError
    with a message a caller can act on."""
    spec = {
        "max_results": _num(parameters, "max_results",
                            _MAX_RESULTS_DEFAULT[command], int, lo=1),
        "min_detection_confidence": _num(parameters, "min_detection_confidence",
                                         0.5, float, lo=0.0, hi=1.0),
        "min_presence_confidence": _num(parameters, "min_presence_confidence",
                                        0.5, float, lo=0.0, hi=1.0),
        "min_tracking_confidence": _num(parameters, "min_tracking_confidence",
                                        0.5, float, lo=0.0, hi=1.0),
        "model": "lite",
        "blendshapes": False,
        "transformation_matrix": False,
    }
    if command == "pose":
        model = str(parameters.get("model") or "lite").strip().lower()
        if model not in _POSE_MODELS:
            raise ValueError(
                f"parameters.model must be one of {', '.join(_POSE_MODELS)}, "
                f"got {model!r}")
        spec["model"] = model
    if command == "face":
        spec["blendshapes"] = _flag(parameters, "blendshapes", False)
        spec["transformation_matrix"] = _flag(parameters, "transformation_matrix", False)

    coordinates = str(parameters.get("coordinates") or "normalized").strip().lower()
    if coordinates not in ("normalized", "pixels"):
        raise ValueError("parameters.coordinates must be 'normalized' or 'pixels', "
                         f"got {coordinates!r}")
    return spec, coordinates


# ---------------------------------------------
# Packing: MediaPipe result objects -> padded arrays
#
# Kept free of any mediapipe import so it can be tested without the bundles:
# everything below needs only objects carrying .x/.y/.z (and .visibility,
# .category_name, .score).
# ---------------------------------------------
def _xyz(landmarks, n_points, scale):
    """One detection -> (n_points, 3) float32, in the requested units."""
    sx, sy, sz = scale
    out = np.zeros((n_points, 3), dtype=np.float32)
    for i, lm in enumerate(landmarks[:n_points]):
        out[i] = (lm.x * sx, lm.y * sy, lm.z * sz)
    return out


def pack_landmarks(per_image, n_points, scales):
    """[[detection, …], …] -> ((n_images, M, n_points, 3), counts).

    M is the largest detection count in the batch, so an image with fewer
    detections is zero-padded and an empty batch gives M = 0. `counts` is the
    only field that says how many rows are real.
    """
    counts = np.array([len(d) for d in per_image], dtype=np.int32)
    m = int(counts.max()) if len(counts) else 0
    out = np.zeros((len(per_image), m, n_points, 3), dtype=np.float32)
    for i, detections in enumerate(per_image):
        for j, landmarks in enumerate(detections[:m]):
            out[i, j] = _xyz(landmarks, n_points, scales[i])
    return out, counts


def pack_scalars(per_image, m, fill=0.0, dtype=np.float32):
    """[[value, …], …] -> (n_images, M), padded with `fill`."""
    out = np.full((len(per_image), m), fill, dtype=dtype)
    for i, values in enumerate(per_image):
        for j, v in enumerate(values[:m]):
            out[i, j] = v
    return out


def pack_vectors(per_image, m, width, dtype=np.float32):
    """[[sequence of `width` numbers, …], …] -> (n_images, M, width)."""
    out = np.zeros((len(per_image), m, width), dtype=dtype)
    for i, vectors in enumerate(per_image):
        for j, vec in enumerate(vectors[:m]):
            out[i, j, :len(vec)] = np.asarray(vec, dtype=dtype)[:width]
    return out


def pack_matrices(per_image, m, rows, cols, dtype=np.float32):
    """[[(rows, cols) matrix, …], …] -> (n_images, M, rows, cols)."""
    out = np.zeros((len(per_image), m, rows, cols), dtype=dtype)
    for i, matrices in enumerate(per_image):
        for j, mat in enumerate(matrices[:m]):
            arr = np.asarray(mat, dtype=dtype).reshape(rows, cols)
            out[i, j] = arr
    return out


def _visibility(per_image, m, n_points):
    """Pose only: (n_images, M, 33) from the landmarks' .visibility."""
    out = np.zeros((len(per_image), m, n_points), dtype=np.float32)
    for i, detections in enumerate(per_image):
        for j, landmarks in enumerate(detections[:m]):
            for k, lm in enumerate(landmarks[:n_points]):
                out[i, j, k] = float(getattr(lm, "visibility", 0.0) or 0.0)
    return out


def _handedness_codes(per_image, m):
    """(n_images, M) int8 0 = left, 1 = right, -1 = padding, + the scores."""
    codes = np.full((len(per_image), m), -1, dtype=np.int8)
    scores = np.zeros((len(per_image), m), dtype=np.float32)
    for i, hands in enumerate(per_image):
        for j, categories in enumerate(hands[:m]):
            if not categories:
                continue
            top = categories[0]
            name = str(getattr(top, "category_name", "") or "").lower()
            codes[i, j] = 0 if name.startswith("l") else 1
            scores[i, j] = float(getattr(top, "score", 0.0) or 0.0)
    return codes, scores


# ---------------------------------------------
# Service
# ---------------------------------------------
class PipelineService(folder_wd_pb2_grpc.PipelineServiceServicer):

    def __init__(self):
        # One landmarker per command, rebuilt when the options change: the
        # detection limit and the thresholds are baked into the object, so a
        # request that asks for something else cannot reuse it.
        self._landmarkers = {}   # command -> (options_key, landmarker)
        self._lock = threading.Lock()

    # ------------------------------------------------------------- models
    def _landmarker(self, command, spec):
        """Get or build the landmarker for this command and these options.

        Held under the service lock: a MediaPipe landmarker is a stateful
        graph, and this box is served by a thread pool.
        """
        key = (spec["model"], spec["max_results"],
               spec["min_detection_confidence"], spec["min_presence_confidence"],
               spec["min_tracking_confidence"],
               spec["blendshapes"], spec["transformation_matrix"])
        cached_key, cached = self._landmarkers.get(command, (None, None))
        if cached is not None and cached_key == key:
            return cached
        if cached is not None:
            cached.close()
            self._landmarkers.pop(command, None)

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        path = model_path(command, spec["model"])
        if not path.is_file():
            raise FileNotFoundError(
                f"model bundle {path.name} is missing from {path.parent} — the "
                f"image bakes it in at build time; from a checkout run "
                f"`python3 tools/fetch_assets.py mediapipe`")
        base = mp_python.BaseOptions(model_asset_path=str(path))
        common = dict(base_options=base, running_mode=mp_vision.RunningMode.IMAGE)

        # Upstream names neither the file nor the missing library when this
        # fails; load_error() decides which of the two it was.
        def _build(factory, options):
            try:
                return factory(options)
            except Exception as e:
                raise load_error(path, e) from None

        if command == "hands":
            options = mp_vision.HandLandmarkerOptions(
                **common,
                num_hands=spec["max_results"],
                min_hand_detection_confidence=spec["min_detection_confidence"],
                min_hand_presence_confidence=spec["min_presence_confidence"],
                min_tracking_confidence=spec["min_tracking_confidence"])
            landmarker = _build(mp_vision.HandLandmarker.create_from_options, options)
        elif command == "pose":
            options = mp_vision.PoseLandmarkerOptions(
                **common,
                num_poses=spec["max_results"],
                min_pose_detection_confidence=spec["min_detection_confidence"],
                min_pose_presence_confidence=spec["min_presence_confidence"],
                min_tracking_confidence=spec["min_tracking_confidence"])
            landmarker = _build(mp_vision.PoseLandmarker.create_from_options, options)
        else:
            options = mp_vision.FaceLandmarkerOptions(
                **common,
                num_faces=spec["max_results"],
                min_face_detection_confidence=spec["min_detection_confidence"],
                min_face_presence_confidence=spec["min_presence_confidence"],
                min_tracking_confidence=spec["min_tracking_confidence"],
                output_face_blendshapes=spec["blendshapes"],
                output_facial_transformation_matrixes=spec["transformation_matrix"])
            landmarker = _build(mp_vision.FaceLandmarker.create_from_options, options)

        self._landmarkers[command] = (key, landmarker)
        logging.info(f"{command} landmarker built from {path.name} ({key})")
        return landmarker

    def _close_all(self):
        for command, (_, landmarker) in list(self._landmarkers.items()):
            try:
                landmarker.close()
            except Exception as e:  # a closed graph is not worth failing a reset
                logging.warning(f"closing the {command} landmarker: {e}")
        self._landmarkers.clear()

    # ------------------------------------------------------------- inputs
    @staticmethod
    def _decode_images(blobs):
        """JPEG/PNG bytes -> BGR ndarray list; raises on any bad frame."""
        frames = []
        for i, raw in enumerate(blobs):
            arr = np.frombuffer(bytes(raw), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"could not decode image #{i + 1} of "
                                 f"{len(blobs)} (not a supported raster?)")
            frames.append(img)
        return frames

    @staticmethod
    def _scales(frames, coordinates):
        """Per-image (sx, sy, sz) for the requested units.

        MediaPipe normalizes x and y to [0, 1] and puts z on roughly the same
        scale as x — so pixels means width, height, width.
        """
        if coordinates == "pixels":
            return [(float(f.shape[1]), float(f.shape[0]), float(f.shape[1]))
                    for f in frames]
        return [(1.0, 1.0, 1.0)] * len(frames)

    # ------------------------------------------------------------ detect
    def _detect(self, command, spec, frames):
        """-> the per-image result lists, in MediaPipe's own objects."""
        import mediapipe as mp

        results = []
        with self._lock:
            landmarker = self._landmarker(command, spec)
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=np.ascontiguousarray(rgb))
                results.append(landmarker.detect(image))
        return results

    @staticmethod
    def _pack(command, spec, results, scales):
        """MediaPipe results -> the declared fields, as plain arrays."""
        n_points = LANDMARKS[command]
        if command == "hands":
            per_image = [r.hand_landmarks for r in results]
        elif command == "pose":
            per_image = [r.pose_landmarks for r in results]
        else:
            per_image = [r.face_landmarks for r in results]

        landmarks, counts = pack_landmarks(per_image, n_points, scales)
        m = landmarks.shape[1]
        out = {"landmarks": landmarks, "counts": counts}
        encoding = {"landmarks": "numpy", "counts": "numpy"}

        if command == "hands":
            world = [r.hand_world_landmarks for r in results]
            out["world_landmarks"], _ = pack_landmarks(
                world, n_points, [(1.0, 1.0, 1.0)] * len(results))
            codes, scores = _handedness_codes([r.handedness for r in results], m)
            out["handedness"], out["scores"] = codes, scores
            encoding.update(world_landmarks="numpy", handedness="numpy", scores="numpy")
        elif command == "pose":
            world = [r.pose_world_landmarks for r in results]
            out["world_landmarks"], _ = pack_landmarks(
                world, n_points, [(1.0, 1.0, 1.0)] * len(results))
            out["visibility"] = _visibility(per_image, m, n_points)
            encoding.update(world_landmarks="numpy", visibility="numpy")
        else:
            if spec["blendshapes"]:
                per_face = [[[c.score for c in face] for face in (r.face_blendshapes or [])]
                            for r in results]
                out["blendshapes"] = pack_vectors(per_face, m, N_BLENDSHAPES)
                encoding["blendshapes"] = "numpy"
                names = []
                for r in results:
                    for face in (r.face_blendshapes or []):
                        names = [c.category_name for c in face]
                        break
                    if names:
                        break
                out["blendshape_names"] = names
                encoding["blendshape_names"] = "json"
            if spec["transformation_matrix"]:
                per_face = [list(r.facial_transformation_matrixes or []) for r in results]
                out["transformation_matrixes"] = pack_matrices(per_face, m, 4, 4)
                encoding["transformation_matrixes"] = "numpy"
        return out, encoding

    # ---------------------------------------------------------------- Process
    def Process(self, request, context):
        start_time = time.time()
        try:
            if not request.config_json:
                return _error("No config JSON")

            config = json.loads(request.config_json)
            if not isinstance(config, dict) or not isinstance(config.get(BOX_KEY), dict):
                return _error(f"config section {BOX_KEY!r} missing or not an object")
            section = config[BOX_KEY]
            command = section.get("command") or "hands"
            parameters = section.get("parameters", {})
            if not isinstance(parameters, dict):
                return _error("parameters must be an object")

            if command == "reset":
                with self._lock:
                    self._close_all()
                return _done({"action": "reset"})

            if command not in LANDMARKS:
                return _error(f"unknown command {command!r} (expected "
                              f"hands, pose, face or reset)")

            spec, coordinates = parse_parameters(command, parameters)

            images = unwrap_value(request.data["images"]) if "images" in request.data else None
            if isinstance(images, (bytes, bytearray)):
                images = [images]
            if not images:
                return _empty_request()

            frames = self._decode_images(images)
            scales = self._scales(frames, coordinates)
            results = self._detect(command, spec, frames)
            fields, encoding = self._pack(command, spec, results, scales)

            data = {}
            for name, value in fields.items():
                if encoding[name] == "json":
                    data[name] = wrap_value(json.dumps(value).encode())
                else:
                    data[name] = wrap_value(np_to_bytes(value))

            counts = fields["counts"]
            return _done(
                {
                    "command": command,
                    "model": spec["model"] if command == "pose" else command,
                    "landmarks_per_detection": LANDMARKS[command],
                    "coordinates": coordinates,
                    "num_images": len(frames),
                    "num_detections": int(counts.sum()),
                    "runtime": time.time() - start_time,
                },
                data=data, encoding=encoding)
        except FileNotFoundError as e:
            logging.error(f"missing model bundle: {e}")
            return _error(str(e))
        except ValueError as e:
            return _error(str(e))
        except Exception as e:
            logging.exception(f"Error in Process: {e}")
            return _error(str(e))


# ---------------------------------------------
# Server setup
# ---------------------------------------------
def get_port():
    try:
        port = int(os.getenv(_PORT_ENV_VAR, _PORT_DEFAULT))
        if port <= 0:
            logging.error("Port must be positive")
            return None
        return port
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
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=serve.max_workers()),
        options=serve.server_options(),
        maximum_concurrent_rpcs=serve.max_concurrent_rpcs())
    folder_wd_pb2_grpc.add_PipelineServiceServicer_to_server(PipelineService(), server)

    service_names = (
        folder_wd_pb2.DESCRIPTOR.services_by_name["PipelineService"].full_name,
        grpc_reflection.SERVICE_NAME,
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)
