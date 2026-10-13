import concurrent.futures as futures
import io
import json
import logging
import os
import sys
import threading
import time

import numpy as np
import torch
from PIL import Image

# add the vendored vggt codebase + the vendored protos to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vggt"))
sys.path.append("./protos")

import grpc
import grpc_reflection.v1alpha.reflection as grpc_reflection

import pipeline_pb2
import pipeline_pb2_grpc
from aux import wrap_value, unwrap_value
import serve

from vggt.models.vggt import VGGT
from vggt.visual_util import predictions_to_glb
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from utils.preprocess import preprocess_images_batch


_PORT_ENV_VAR = 'PORT'
_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_IDLE_TIMEOUT = 60  # seconds

# Accept the box key first; "aispgradio" is the pre-convention name, kept so
# old callers keep working (same alias pattern as lang_segm).
_BOX_KEYS = ("vggt", "aispgradio")

#: Declared payload encoding (visionist_client contract): the tensors travel as
#: torch.save()-format bytes, the GLB is a raw binary blob (identity — the
#: explicit declaration is what keeps the client's legacy guess-chain from
#: mis-decoding the glTF binary as a float buffer).
ENCODING = {
    "world_points": "torch",
    "world_points_conf": "torch",
    "depth": "torch",
    "depth_conf": "torch",
    "extrinsic": "torch",
    "intrinsic": "torch",
    "images": "torch",
    "glb_file": "identity",
}

# Model weights (baked into the image at build time; downloaded on startup if absent)
_WEIGHTS_FILENAME = "vggt-1b.pt"          # runtime name (WORKDIR)
_WEIGHTS_REPO = "facebook/VGGT-1B"
_WEIGHTS_HF_FILE = "model.pt"             # filename inside the HF repo


def _tensor_bytes(t: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(t.detach().cpu().contiguous(), buf, pickle_protocol=4)
    return buf.getvalue()


def _cfg_envelope(status: str, **extra) -> "pipeline_pb2.Envelope":
    return pipeline_pb2.Envelope(
        config_json=json.dumps({"vggt": {"status": status, **extra}}))


def _weights_path() -> str:
    """Return a path to the VGGT 1B checkpoint, downloading it on first use."""
    if os.path.isfile(_WEIGHTS_FILENAME):
        return _WEIGHTS_FILENAME
    from huggingface_hub import hf_hub_download
    logging.info("VGGT weights not found locally — downloading %s/%s ...",
                 _WEIGHTS_REPO, _WEIGHTS_HF_FILE)
    return hf_hub_download(repo_id=_WEIGHTS_REPO,
                           filename=_WEIGHTS_HF_FILE,
                           local_dir=".")


def _clear_position_caches(model):
    """VGGT's ``PositionGetter`` caches position grids keyed by ``(h, w)`` on
    the *first caller's device* (upstream VGGT bug: the key omits the
    device). The getter is a plain attribute (``aggregator.position_getter``
    — not an ``nn.Module``), so ``model.to(device)`` never moves those
    tensors. Clear the cache after every device move, or the next forward
    cats cuda/cpu tensors."""
    for module in model.modules():
        candidates = [module] + list(vars(module).values())
        for obj in candidates:
            cache = getattr(obj, "position_cache", None)
            if isinstance(cache, dict):
                cache.clear()


class PipelineService(pipeline_pb2_grpc.PipelineServiceServicer):

    def __init__(self):
        # Always load to CPU first; moved to GPU lazily on request (see below).
        self._model = None
        self._device = "cpu"
        self._last_request_time = time.time()
        self._lock = threading.Lock()

        self._load_event = threading.Event()
        self._load_error = None

        # Background model loader
        self._loader_thread = threading.Thread(target=self._load_model_async, daemon=True)
        self._loader_thread.start()

        # Watchdog thread
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model_async(self):
        """Background thread that loads the VGGT model on CPU."""
        try:
            logging.info("Loading VGGT model asynchronously...")
            mdl = VGGT()
            mdl.load_state_dict(
                torch.load(_weights_path(), map_location="cpu"))
            with self._lock:
                self._model = mdl
                self._device = "cpu"
            logging.info("VGGT model fully loaded on CPU.")
        except Exception as e:
            self._load_error = f"{type(e).__name__}: {e}"
            logging.exception("Failed to load VGGT model")
        finally:
            self._load_event.set()

    def _watchdog_loop(self):
        while True:
            time.sleep(10)
            with self._lock:
                idle_time = time.time() - self._last_request_time
                # Move back to CPU only if currently on GPU
                if idle_time > _IDLE_TIMEOUT and self._device.startswith("cuda"):
                    logging.info("Idle timeout reached: moving model back to CPU")
                    self._model.to("cpu")
                    # The PositionGetter caches are plain dicts on the
                    # module (not nn.Modules): they keep their CUDA tensors
                    # after the model move, so empty_cache() alone can't
                    # release them — clear them first.
                    _clear_position_caches(self._model)
                    torch.cuda.empty_cache()
                    self._device = "cpu"

    def _resolve_device(self, requested):
        """Decide where the model runs for this request.

        * explicit ``parameters.device`` (``"cpu"`` / ``"cuda[*]"``) wins;
        * otherwise: CUDA when visible, CPU otherwise (fleet convention).
        The move is in-place (``.to(device)`` on the torch modules).
        """
        with self._lock:
            self._last_request_time = time.time()
            if requested:
                target = str(requested).lower()
                if target.startswith("cuda") and not torch.cuda.is_available():
                    logging.warning("CUDA requested but not available. Staying on CPU.")
                    return self._device
                if target == "cpu" or target.startswith("cuda"):
                    if target != self._device:
                        logging.info("Request received: moving model to %s", target)
                        self._model.to(target)
                        _clear_position_caches(self._model)
                        self._device = target
            elif self._device == "cpu" and torch.cuda.is_available():
                logging.info("Request received: moving model to GPU")
                self._model.to("cuda")
                _clear_position_caches(self._model)
                self._device = "cuda"
            return self._device

    # ------------------------------------------------------------------
    # gRPC entry point
    # ------------------------------------------------------------------

    def Process(self, request, context):
        start = time.time()

        # Wait for model load to complete (blocking)
        while not self._load_event.is_set():
            logging.info("VGGT model still loading... waiting before processing.")
            time.sleep(1)

        if self._load_error:
            logging.error(f"VGGT model failed to load: {self._load_error}")
            return _cfg_envelope("error", error=self._load_error,
                                 runtime=time.time() - start)

        try:
            if not request.config_json:
                return _cfg_envelope("error",
                                     error="No config JSON")

            try:
                config = json.loads(request.config_json)
            except json.JSONDecodeError:
                return _cfg_envelope("error", error="config_json is not valid JSON")

            # Box section: namespaced under the box key (legacy flat / aispgradio
            # accepted for old callers, same as lang_segm).
            section = {}
            for key in _BOX_KEYS:
                if isinstance(config, dict) and isinstance(config.get(key), dict):
                    section = config[key]
                    break
            command = section.get("command")

            # Stateless box: "reset" is accepted as a no-op (client convenience).
            if command == "reset":
                return _cfg_envelope("done", action="reset",
                                     runtime=time.time() - start)

            params = section.get("parameters") or {}
            if not isinstance(params, dict) and isinstance(config, dict):
                params = config.get("parameters") or {}   # legacy flat form

            image_list = (unwrap_value(request.data["images"])
                          if "images" in request.data else None)
            # A lone image arrives as a bare bytes value (Value.b), not a
            # one-element BytesList; iterating it would yield ints.
            if isinstance(image_list, (bytes, bytearray)):
                image_list = [image_list]
            if not image_list:
                return _cfg_envelope("empty_request")
            if not all(isinstance(b, (bytes, bytearray)) for b in image_list):
                return _cfg_envelope(
                    "error",
                    error="data.images must be image file bytes (send the file "
                          "contents, not a path string)")

            device = self._resolve_device(params.get("device"))
            conf_thres = params.get("conf_threshold", 30)
            logging.info("device=%s conf_threshold=%s images=%d",
                         device, conf_thres, len(image_list))

            predictions, glb_bytes = self._run(image_list, device, conf_thres)

            response = pipeline_pb2.Envelope(
                config_json=json.dumps({
                    "vggt": {
                        "status": "done",
                        "runtime": time.time() - start,
                        "num_images": len(image_list),
                        "device": self._device,
                        "encoding": ENCODING,
                    }
                }),
                data={
                    "world_points": wrap_value(_tensor_bytes(predictions["world_points"])),
                    "world_points_conf": wrap_value(_tensor_bytes(predictions["world_points_conf"])),
                    "depth": wrap_value(_tensor_bytes(predictions["depth"])),
                    "depth_conf": wrap_value(_tensor_bytes(predictions["depth_conf"])),
                    "extrinsic": wrap_value(_tensor_bytes(predictions["extrinsic"])),
                    "intrinsic": wrap_value(_tensor_bytes(predictions["intrinsic"])),
                    "images": wrap_value(_tensor_bytes(predictions["images"])),
                    "glb_file": wrap_value(glb_bytes),
                },
            )

            glb_mb = len(glb_bytes) / (1024 * 1024)
            total_mb = sum(len(unwrap_value(v)) for v in response.data.values()) / (1024 * 1024)
            logging.info("Response size: glb=%.2f MB total=%.2f MB", glb_mb, total_mb)

            return response

        except Exception as e:
            logging.exception("[VGGT] Unhandled exception")
            return _cfg_envelope("error", error=f"{type(e).__name__}: {e}",
                                 runtime=time.time() - start)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run(self, image_list, device, conf_thres):
        received_images = []
        for image_bytes in image_list:
            img = Image.open(io.BytesIO(bytes(image_bytes))).convert("RGB")
            received_images.append(np.array(img))

        # all frames must share height/width (VGGT contract)
        shapes = {img.shape for img in received_images}
        if len(shapes) != 1:
            raise ValueError(
                f"All images must have the same shape, but got: {sorted(shapes)}")

        images = torch.tensor(np.stack(received_images)).permute(0, 3, 1, 2).to(device)
        images = preprocess_images_batch(images.float() / 255)

        if device.startswith("cuda"):
            dtype = (torch.bfloat16
                     if torch.cuda.get_device_capability(device)[0] >= 8
                     else torch.float16)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                predictions = self._model(images, query_points=None)
        else:
            with torch.no_grad():
                predictions = self._model(images, query_points=None)

        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic.squeeze()
        predictions["intrinsic"] = intrinsic.squeeze()

        # Drop bulky intermediates; move everything to CPU for serialization
        predictions.pop("pose_enc_list", None)
        predictions = {k: v.cpu() for k, v in predictions.items()}
        predictions["images"] = images.cpu()   # CHW float tensor (what VGGT saw)

        # predictions_to_glb works on numpy arrays (it calls .astype etc.) —
        # convert just for that call; the response keeps torch tensors.
        glb_pred = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v)
                    for k, v in predictions.items()}
        glb_scene = predictions_to_glb(glb_pred, conf_thres=conf_thres,
                                      target_dir="/tmp")
        glb_bytes = glb_scene.export(file_type="glb")

        return predictions, glb_bytes


# ----------------------------------------
# Server setup and running
# ----------------------------------------

def get_port():
    """Parse the port where the server should listen.

    Returns the port, or None (and logs the problem) if the environment
    variable is missing/invalid.
    """
    try:
        server_port = int(os.getenv(_PORT_ENV_VAR, _PORT_DEFAULT))
        if server_port <= 0:
            logging.error('Port should be greater than 0')
            return None
        return server_port
    except ValueError:
        logging.exception('Unable to parse port')
        return None


def run_server(server):
    """Run the given server on the port defined by the environment variables
    or the default port if it is not defined."""
    port = get_port()
    if not port:
        return

    target = f'[::]:{port}'
    server.add_insecure_port(target)
    server.start()
    logging.info(f'''Server started at {target}''')
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig(
        format='[ %(levelname)s ] %(asctime)s (%(module)s) %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=serve.max_workers()),
        options=serve.server_options(),
        maximum_concurrent_rpcs=serve.max_concurrent_rpcs())

    pipeline_pb2_grpc.add_PipelineServiceServicer_to_server(PipelineService(), server)

    service_names = (
        pipeline_pb2.DESCRIPTOR.services_by_name['PipelineService'].full_name,
        grpc_reflection.SERVICE_NAME
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)
