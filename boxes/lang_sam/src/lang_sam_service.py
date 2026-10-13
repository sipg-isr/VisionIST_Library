import concurrent.futures as futures
import io
import json
import logging
import os
import sys
import threading
import time
import traceback

import torch
import zstandard as zstd
import pickle

from PIL import Image

# Make the protos/ folder importable (same pattern as every other box).
sys.path.append("./protos")
import pipeline_pb2 as lang_sam_pb2  # noqa: E402
import pipeline_pb2_grpc as lang_sam_grpc  # noqa: E402
from aux import wrap_value, unwrap_value
import serve  # noqa: E402

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/lang-segment-anything')
from lang_sam import LangSAM  # noqa: E402

_PORT_ENV_VAR = 'PORT'
_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_IDLE_TIMEOUT = 60  # seconds

_DEFAULT_SAM_TYPE = "sam2.1_hiera_small"

# Canonical box key, plus accepted aliases (legacy pipeline configs used
# "aispgradio" and the flat top-level format when calling this box).
_BOX_KEYS = ("lang_sam", "lang_segm", "aispgradio")


def _encode_results(out_list) -> bytes:
    """Serialize LangSAM outputs for the Envelope (zstd + pickle).

    Clients decode with: pickle.loads(zstd.ZstdDecompressor().decompress(b))
    """
    return zstd.compress(pickle.dumps(out_list))


def _decode_config(request):
    """Parse request.config_json and locate this box's section.

    Returns (box_key, box_config_section) or raises ValueError.
    """
    if not request.config_json:
        raise ValueError("No config JSON")

    config = json.loads(request.config_json)  # may raise JSONDecodeError

    for key in _BOX_KEYS:
        section = config.get(key)
        if isinstance(section, dict):
            return key, section

    # Legacy flat format: {parameters: {...}, text_prompt: [...]}
    if "parameters" in config or "text_prompt" in config:
        return "lang_sam", config

    raise ValueError(f"config JSON has no '{'/'' or '.join(_BOX_KEYS)}' section")


class PipelineService(lang_sam_grpc.PipelineServiceServicer):

    def __init__(self):
        # Always load to CPU first; moved to GPU lazily on request.
        self._model = LangSAM(sam_type=_DEFAULT_SAM_TYPE, device="cpu")
        self._device = "cpu"
        logging.info("LangSAM model loaded on CPU")

        self._last_request_time = time.time()
        self._lock = threading.Lock()

        # Background thread to monitor idle time.
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def _watchdog_loop(self):
        while True:
            time.sleep(10)  # check every 10s
            with self._lock:
                idle_time = time.time() - self._last_request_time
                if idle_time > _IDLE_TIMEOUT and self._device.startswith("cuda"):
                    logging.info("Idle timeout reached: moving model back to CPU")
                    self._set_device_locked("cpu")

    def set_device(self, target: str) -> str:
        """(Re)place the LangSAM model on the requested device."""
        with self._lock:
            self._last_request_time = time.time()
            return self._set_device_locked(target)

    def _release_inference_cache(self) -> None:
        """Free the image embeddings cached by the SAM2 predictor.

        They are plain tensor attributes on the predictor (not module
        parameters/buffers), so ``model.to("cpu")`` and
        ``torch.cuda.empty_cache()`` cannot release them. They are
        recomputed on every ``predict`` call, so dropping them is safe.
        """
        try:
            predictor = getattr(self._model.sam, "predictor", None)
            if predictor is not None and hasattr(predictor, "reset_predictor"):
                predictor.reset_predictor()
        except Exception:
            logging.exception("Failed to release SAM2 predictor cache")

    def _set_device_locked(self, target: str) -> str:
        """Swap the model device. Must be called with self._lock held."""
        target = target.lower()
        if target.startswith("cuda") and not torch.cuda.is_available():
            logging.warning("CUDA requested but not available. Staying on %s.",
                            self._device)
            return self._device

        if target == self._device:
            return self._device

        # Fast path: move the already-loaded torch modules in place. Much
        # cheaper than rebuilding LangSAM, which reloads the SAM 2.1 and
        # Grounding-DINO checkpoints on every switch.
        try:
            device = torch.device(target)
            # Drop the cached image embeddings first, otherwise the last
            # inference's feature maps would keep referencing GPU memory
            # even after moving off CUDA.
            if not target.startswith("cuda"):
                self._release_inference_cache()
            logging.info(f"Moving model to {target}")
            self._model.sam.model.to(device)
            self._model.gdino.model.to(device)
            self._device = target
        except Exception:
            # Slow path (e.g. library internals changed, OOM): rebuild
            # the model on the target device, like before.
            try:
                logging.exception("In-place move failed; rebuilding model on %s",
                                  target)
                new_model = LangSAM(sam_type=_DEFAULT_SAM_TYPE,
                                    device=torch.device(target))
                del self._model  # also drops the old predictor's cache
                self._model = new_model
                self._device = target
                if not target.startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception:
                logging.exception("Failed to move model to %s", target)
                return self._device
        else:
            if not target.startswith("cuda"):
                torch.cuda.empty_cache()
        return self._device

    def Process(self, request, context):
        """Perform text-guided segmentation on image(s).

        Request:
            config_json -> {"lang_sam": {"command": "?",
                                         "parameters": {...},
                                         "text_prompt": [str, ...]}}
            data["images"] -> list of image bytes (JPEG/PNG)

        Response:
            config_json -> {"lang_sam": {"status": "done", "runtime", ...}}
            data["results"] -> zstd-compressed pickled list (one LangSAM
                               predict() output dict per input image)
        """
        box_key = "lang_sam"

        def _status(status, **extra):
            return lang_sam_pb2.Envelope(
                config_json=json.dumps({box_key: {"status": status, **extra}}))

        start_time = time.time()

        try:
            box_key, box_cfg = _decode_config(request)
        except json.JSONDecodeError:
            logging.error("config_json is not valid JSON")
            return lang_sam_pb2.Envelope(
                config_json=json.dumps({"lang_sam": {"status": "error",
                                                     "error": "config_json is not valid JSON"}}))
        except ValueError as e:
            return lang_sam_pb2.Envelope(
                config_json=json.dumps({"lang_sam": {"status": "error", "error": str(e)}}))

        parameters = box_cfg.get("parameters", {}) or {}

        # Stateless box: accept "reset" (client convenience) as a no-op.
        if box_cfg.get("command") == "reset" or parameters.get("reset"):
            return _status("done", action="reset")

        # Device selection: an explicit "device" parameter (if any) wins;
        # otherwise default to the GPU when CUDA is visible (matches the
        # clip / textEmbedding / tapnext boxes and the README's documented
        # "cuda automatically" contract).
        requested_device = (parameters.get("device") or "").strip().lower()
        if requested_device:
            self.set_device(requested_device)
        elif torch.cuda.is_available():
            self.set_device("cuda")

        # --- Extract image(s) ---
        if "images" not in request.data:
            # No payload: some stages (e.g. opencv frame recovery) forward
            # config-only envelopes in the pipeline. Echo the envelope.
            return lang_sam_pb2.Envelope(config_json=request.config_json)

        img_list = unwrap_value(request.data["images"])
        # A lone image arrives as a bare bytes value (Value.b), not a
        # one-element BytesList; iterating it would yield ints.
        if isinstance(img_list, (bytes, bytearray)):
            img_list = [img_list]
        if not img_list:
            return _status("empty_request")
        if not all(isinstance(b, (bytes, bytearray)) for b in img_list):
            return _status("error", error="data.images must be image file bytes "
                           "(send the file contents, not a path string)")

        # --- Extract text prompts ---
        text_prompts = box_cfg.get("text_prompt") or parameters.get("text_prompt") or []
        text_prompts = [str(p) for p in text_prompts]
        if not text_prompts:
            return _status("error", error="No text_prompt in config")

        try:
            # --- Run inference ---
            received_images = []
            for image_bytes in img_list:
                received_images.append(
                    Image.open(io.BytesIO(bytes(image_bytes))).convert("RGB"))

            # LangSAM pairs one prompt with each image (Grounding-DINO
            # batches text and images 1:1), so give every image the same
            # joined prompt.
            text_prompt_str = ". ".join(text_prompts) + "."

            box_threshold = float(parameters.get("box_threshold", 0.3))
            text_threshold = float(parameters.get("text_threshold", 0.25))

            out_list = self._model.predict(
                received_images,
                [text_prompt_str] * len(received_images),
                box_threshold, text_threshold)
            # LangSAM kept the SAM2 predictor's image embeddings for the
            # last image batch; they are re-embedded on the next predict,
            # so release them to keep GPU memory low between requests.
            self._release_inference_cache()
        except Exception as e:
            logging.exception("[lang_sam] inference failed:\n%s", traceback.format_exc())
            return _status("error", error=str(e))

        # --- Build response ---
        runtime = time.time() - start_time
        logging.info(f"Inference completed on {len(received_images)} image(s) "
                     f"in {runtime:.2f}s")
        return lang_sam_pb2.Envelope(
            config_json=json.dumps({box_key: {
                "status": "done",
                "runtime": runtime,
                "num_images": len(received_images),
                "num_prompts": len(text_prompts),
                # Declared payload encoding (generic visionist_client contract):
                # data.results is zstd.compress(pickle.dumps(list)).
                "encoding": "zstd_pickle",
            }}),
            data={"results": wrap_value(_encode_results(out_list))}
        )


# ----------------------------------------
# Server setup and running
# ----------------------------------------

def get_port():
    """Parse the port where the server should listen.

    Returns the port, or None if invalid.
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
    """Run the given server on the port given by the PORT env var or default."""
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
    import grpc
    import grpc_reflection.v1alpha.reflection as grpc_reflection

    logging.basicConfig(
        format='[ %(levelname)s ] %(asctime)s (%(module)s) %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=serve.max_workers()),
        options=serve.server_options(),
        maximum_concurrent_rpcs=serve.max_concurrent_rpcs())

    lang_sam_grpc.add_PipelineServiceServicer_to_server(PipelineService(), server)

    # Add reflection
    service_names = (
        lang_sam_pb2.DESCRIPTOR.services_by_name['PipelineService'].full_name,
        grpc_reflection.SERVICE_NAME
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)
