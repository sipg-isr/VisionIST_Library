import concurrent.futures as futures
import sys
import logging
import os
import time
import json
import io
import threading

sys.path.append("./protos")
import pipeline_pb2
import pipeline_pb2_grpc
from aux import wrap_value, unwrap_value
import serve

import torch
import clip
from PIL import Image


_PORT_ENV_VAR = 'PORT'
_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_IDLE_TIMEOUT = 60  # seconds

_DEFAULT_MODEL = "ViT-B/32"


def _serialize_tensor(t: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(t.detach().cpu(), buf, pickle_protocol=4)
    return buf.getvalue()


class PipelineService(pipeline_pb2_grpc.PipelineServiceServicer):

    def __init__(self):
        # Always load to CPU first; moved to GPU lazily on request (see below).
        self._model, self._preprocess = clip.load(_DEFAULT_MODEL, device="cpu",
                                                  download_root="./model")
        for p in self._model.parameters():
            p.requires_grad = False
        self._device = "cpu"
        logging.info("CLIP model loaded on CPU")

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
                if idle_time > _IDLE_TIMEOUT and self._device == "cuda":
                    logging.info("Idle timeout reached: moving model back to CPU")
                    self._model.to("cpu")
                    torch.cuda.empty_cache()
                    self._device = "cpu"

    def Process(self, request, context):
        start_time = time.time()

        with self._lock:
            self._last_request_time = time.time()
            # If the idle watchdog moved it back to CPU, restore to GPU.
            if self._device == "cpu" and torch.cuda.is_available():
                logging.info("Request received: moving model to GPU")
                self._model.to("cuda")
                self._device = "cuda"

        try:
            if not request.config_json:
                return pipeline_pb2.Envelope(
                    config_json=json.dumps({"clip": {"status": "error", "error": "No config JSON"}}))

            config = json.loads(request.config_json)
            clip_config = config.get("clip", {})
            parameters = clip_config.get("parameters", {}) or {}

            # Stateless box: accept "reset" (client convenience) as a no-op.
            if clip_config.get("command") == "reset" or parameters.get("reset"):
                return pipeline_pb2.Envelope(
                    config_json=json.dumps({"clip": {"status": "done", "action": "reset"}}))

            # parameters.model is accepted for API consistency; the service runs
            # the model loaded at startup (ViT-B/32). Switching checkpoints would
            # require re-loading and is intentionally deferred.
            image_bytes_list = unwrap_value(request.data["images"]) if "images" in request.data else None
            texts_list = unwrap_value(request.data["texts"]) if "texts" in request.data else None
            # A lone value arrives bare (Value.b / Value.s), not as a one-element
            # list; iterating it would yield ints / characters.
            if isinstance(image_bytes_list, (bytes, bytearray)):
                image_bytes_list = [image_bytes_list]
            if isinstance(texts_list, str):
                texts_list = [texts_list]

            if not image_bytes_list:
                return pipeline_pb2.Envelope(
                    config_json=json.dumps({"clip": {"status": "empty_request"}}))
            if not texts_list:
                return pipeline_pb2.Envelope(
                    config_json=json.dumps({"clip": {"status": "error", "error": "No texts in data"}}))
            if not all(isinstance(b, (bytes, bytearray)) for b in image_bytes_list):
                return pipeline_pb2.Envelope(
                    config_json=json.dumps({"clip": {
                        "status": "error",
                        "error": "data.images must be image file bytes (send the "
                                 "file contents, not a path string)"}}))

            image_features, text_features, logits_per_image = self._encode(
                image_bytes_list, texts_list)

            response_data = {
                "image_emb": wrap_value(_serialize_tensor(image_features)),
                "text_emb": wrap_value(_serialize_tensor(text_features)),
                "similarity": wrap_value(_serialize_tensor(logits_per_image)),
            }

            return pipeline_pb2.Envelope(
                config_json=json.dumps({
                    "clip": {
                        "status": "done",
                        "runtime": time.time() - start_time,
                        "num_images": len(image_bytes_list),
                        "num_texts": len(texts_list),
                        # Declared payload encoding (generic visionist_client contract):
                        # all responses are torch.save()-format tensor bytes.
                        "encoding": {
                            "image_emb": "torch",
                            "text_emb": "torch",
                            "similarity": "torch",
                        }
                    }
                }),
                data=response_data
            )

        except Exception as e:
            logging.exception(f"Error in Process: {e}")
            return pipeline_pb2.Envelope(
                config_json=json.dumps({"clip": {"status": "error", "error": str(e)}}))

    def _encode(self, image_bytes_list, texts_list):
        received_images = []
        for image_bytes in image_bytes_list:
            img = Image.open(io.BytesIO(bytes(image_bytes))).convert("RGB")
            received_images.append(img)

        device = self._device
        images = torch.stack([self._preprocess(im) for im in received_images]).to(device)
        text = clip.tokenize(list(texts_list)).to(device)

        with torch.no_grad():
            image_features = self._model.encode_image(images)
            text_features = self._model.encode_text(text)
            logits_per_image, _logits_per_text = self._model(images, text)

        return image_features, text_features, logits_per_image


def get_port():
    """Parse the port where the server should listen.

    Exits the program if the environment variable is not a positive int.

    Returns:
        The port where the server should listen, or None if an error occurred.
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

    pipeline_pb2_grpc.add_PipelineServiceServicer_to_server(PipelineService(), server)

    service_names = (
        pipeline_pb2.DESCRIPTOR.services_by_name['PipelineService'].full_name,
        grpc_reflection.SERVICE_NAME
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)
