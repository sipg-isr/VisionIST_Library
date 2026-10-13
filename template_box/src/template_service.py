"""template box - a minimal, working box to copy.

It runs as-is: send it images and it reports their decoded sizes. Everything
here is contract, not content - replace the one function that does the work
and you have your own box.

The shared-envelope rules this file demonstrates:

* ONE RPC, ``Process``, taking and returning an ``Envelope``.
* The request's work is described by ``config_json``, in a section named after
  this box: ``{"template": {"command": ..., "parameters": {...}}}``.
* The reply's config is namespaced the same way and carries a ``status`` of
  ``done`` | ``empty_request`` | ``error`` - never a bare top-level status.
* ``reset`` is accepted by every box, even if there is no state to clear.
* Binary payloads declare their ``encoding`` so a client knows how to decode
  them without knowing this box: here ``numpy``, an ``np.save`` blob.
* Port 8061 (AI4EU spec), overridable with the ``PORT`` env var.
* Server limits come from ``serve`` (contract/serve.py), so the message cap
  and the worker count are set at launch rather than written into the image.
  Declare your box's own defaults under ``limits:`` in box.yaml.
"""

import concurrent.futures as futures
import io
import json
import logging
import os
import sys
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

#: The config section this box answers to. Must equal `key` in box.yaml, and
#: this file must be named <key>_service.py - the Dockerfile resolves it.
BOX_KEY = "template"

_DEFAULTS = {
    "scale": 1.0,
}


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


def np_to_bytes(arr: np.ndarray) -> bytes:
    """np.save keeps dtype and shape in the blob, so the client's `numpy`
    codec restores the array exactly."""
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr))
    return buf.getvalue()


class PipelineService(folder_wd_pb2_grpc.PipelineServiceServicer):

    @staticmethod
    def _decode_images(blobs):
        frames = []
        for i, raw in enumerate(blobs):
            arr = np.frombuffer(bytes(raw), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"could not decode image #{i + 1} of {len(blobs)}")
            frames.append(img)
        return frames

    def Process(self, request, context):
        start = time.time()
        try:
            if not request.config_json:
                return _error("No config JSON")

            config = json.loads(request.config_json)
            if not isinstance(config, dict) or not isinstance(config.get(BOX_KEY), dict):
                return _error(f"config section {BOX_KEY!r} missing or not an object")
            section = config[BOX_KEY]
            command = section.get("command") or "run"
            parameters = section.get("parameters", {})
            if not isinstance(parameters, dict):
                return _error("parameters must be an object")

            if command == "reset":
                return _done({"action": "reset"})
            if command != "run":
                return _error(f"unknown command {command!r} "
                              f"(expected 'run' or 'reset')")

            images = unwrap_value(request.data["images"]) \
                if "images" in request.data else None
            if isinstance(images, (bytes, bytearray)):
                images = [images]
            if not images:
                return _empty_request()

            try:
                scale = float(parameters.get("scale", _DEFAULTS["scale"]))
            except (TypeError, ValueError):
                return _error("parameters.scale must be a number")

            frames = self._decode_images(images)

            # ---- the only part that is this box's own work ----------------
            sizes = np.array([[f.shape[1] * scale, f.shape[0] * scale]
                              for f in frames], dtype=np.float32)
            # ---------------------------------------------------------------

            return _done(
                {"num_images": len(frames), "scale": scale,
                 "runtime": time.time() - start},
                data={"sizes": wrap_value(np_to_bytes(sizes))},
                encoding={"sizes": "numpy"})
        except Exception as e:                              # noqa: BLE001
            logging.exception("Error in Process")
            return _error(str(e))


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
        futures.ThreadPoolExecutor(max_workers=serve.max_workers()),
        options=serve.server_options(),
        maximum_concurrent_rpcs=serve.max_concurrent_rpcs())
    folder_wd_pb2_grpc.add_PipelineServiceServicer_to_server(
        PipelineService(), server)

    grpc_reflection.enable_server_reflection(
        (folder_wd_pb2.DESCRIPTOR.services_by_name["PipelineService"].full_name,
         grpc_reflection.SERVICE_NAME), server)

    run_server(server)
