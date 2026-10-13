"""open_clip box - image and text embeddings from any OpenCLIP model.

Wraps https://github.com/mlfoundations/open_clip. Where the `clip` box is fixed
to OpenAI's ViT-B/32, this one lets the caller pick any (model, pretrained) pair
that open_clip knows - ViT-B-32/laion2b_s34b_b79k, ViT-L-14/datacomp_xl_s13b_b90k,
ViT-SO400M-14-SigLIP/webli, an `hf-hub:` repo, ... - per request. Models are
loaded on first use and kept in a small LRU cache.

Shared-envelope rules this file follows:

* ONE RPC, ``Process``; the work is described by ``config_json`` under the
  ``open_clip`` key; the reply is namespaced the same way and carries a
  ``status`` of ``done`` | ``empty_request`` | ``error``.
* ``reset`` is accepted (it unloads every cached model and frees the memory).
* Payloads declare their ``encoding`` (``numpy``: an ``np.save`` blob).
* Port 8061 (AI4EU spec), overridable with ``PORT``.

Environment:
  OPEN_CLIP_MODEL / OPEN_CLIP_PRETRAINED   defaults when a request names none
  OPEN_CLIP_CACHE_SIZE                     models kept loaded (default 1)
  OPEN_CLIP_IDLE_SECONDS                   idle time before GPU models go back
                                           to CPU (default 60; 0 disables)
"""

import collections
import concurrent.futures as futures
import io
import json
import logging
import os
import sys
import threading
import time

sys.path.append("./protos")
import pipeline_pb2  # noqa: E402
import pipeline_pb2_grpc  # noqa: E402
from aux import wrap_value, unwrap_value
import serve  # noqa: E402

import numpy as np  # noqa: E402
import open_clip  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_PORT_ENV_VAR = "PORT"

#: The config section this box answers to. Must equal `key` in box.yaml, and
#: this file must be named <key>_service.py - the Dockerfile resolves it.
BOX_KEY = "open_clip"

DEFAULT_MODEL = os.getenv("OPEN_CLIP_MODEL", "ViT-B-32")
DEFAULT_PRETRAINED = os.getenv("OPEN_CLIP_PRETRAINED", "laion2b_s34b_b79k")
CACHE_SIZE = max(1, int(os.getenv("OPEN_CLIP_CACHE_SIZE", "1")))
IDLE_SECONDS = float(os.getenv("OPEN_CLIP_IDLE_SECONDS", "60"))

_COMMANDS = ("encode", "models", "reset")
_DEVICES = ("auto", "cuda", "cpu")
_HUB_PREFIXES = ("hf-hub:", "local-dir:")


# --------------------------------------------------------------------------
# envelope helpers
# --------------------------------------------------------------------------

def _done(extra, data=None, encoding=None):
    section = {"status": "done", **extra}
    if encoding:
        section["encoding"] = encoding
    return pipeline_pb2.Envelope(
        config_json=json.dumps({BOX_KEY: section}), data=data or {})


def _empty_request():
    return pipeline_pb2.Envelope(
        config_json=json.dumps({BOX_KEY: {"status": "empty_request"}}))


def _error(message):
    return pipeline_pb2.Envelope(
        config_json=json.dumps({BOX_KEY: {"status": "error", "error": message}}))


class RequestError(ValueError):
    """A problem with the caller's request: reported as status=error."""


def np_to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr))
    return buf.getvalue()


# --------------------------------------------------------------------------
# model registry
# --------------------------------------------------------------------------

def _norm_model(name: str) -> str:
    """OpenAI spelled it ViT-B/32; open_clip spells it ViT-B-32."""
    return name if name.startswith(_HUB_PREFIXES) else name.replace("/", "-")


def _pretrained_by_model() -> dict:
    out = collections.OrderedDict()
    for model, tag in open_clip.list_pretrained():
        out.setdefault(model, []).append(tag)
    return out


def resolve(model, pretrained):
    """Validate a request's (model, pretrained) and return the canonical pair.

    Unknown names would otherwise surface as a stack trace from deep inside
    open_clip (or a silent attempt to download); say what IS available instead.
    """
    model = _norm_model(str(model or DEFAULT_MODEL).strip())
    if model.startswith(_HUB_PREFIXES):
        return model, None                       # the repo/dir carries the weights
    if pretrained in (None, ""):
        pretrained = DEFAULT_PRETRAINED if model == _norm_model(DEFAULT_MODEL) else None
        if pretrained is None:
            raise RequestError(
                f"model {model!r} needs a 'pretrained' tag; "
                f"see command 'models' with parameters.model={model!r}")
    pretrained = str(pretrained)
    if os.path.isfile(pretrained):               # a local checkpoint file
        return model, pretrained
    known = _pretrained_by_model()
    if model not in known:
        raise RequestError(
            f"unknown model {model!r}; run command 'models' for the list, or use "
            f"'hf-hub:<org>/<repo>'")
    tags = {t.lower(): t for t in known[model]}
    if pretrained.lower() not in tags:
        raise RequestError(
            f"unknown pretrained {pretrained!r} for {model!r}; available: "
            f"{', '.join(known[model])}")
    return model, tags[pretrained.lower()]


class _Loaded:
    def __init__(self, model, preprocess, tokenizer):
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.device = "cpu"

    def to(self, device):
        if device != self.device:
            self.model.to(device)
            self.device = device


def load_model(model, pretrained):
    """The one place that touches open_clip's loaders (tests replace it)."""
    net, _train_tf, preprocess = open_clip.create_model_and_transforms(
        model, pretrained=pretrained, device="cpu")
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    return _Loaded(net, preprocess, open_clip.get_tokenizer(model))


# --------------------------------------------------------------------------
# the service
# --------------------------------------------------------------------------

class PipelineService(pipeline_pb2_grpc.PipelineServiceServicer):

    def __init__(self):
        self._cache = collections.OrderedDict()      # (model, pretrained) -> _Loaded
        self._lock = threading.RLock()
        self._last_used = time.time()
        if IDLE_SECONDS > 0:
            threading.Thread(target=self._watchdog, daemon=True).start()

    # -- lifecycle ---------------------------------------------------------

    def _watchdog(self):
        while True:
            time.sleep(min(10, max(1, IDLE_SECONDS / 2)))
            with self._lock:
                if time.time() - self._last_used > IDLE_SECONDS:
                    moved = False
                    for entry in self._cache.values():
                        if entry.device != "cpu":
                            entry.to("cpu")
                            moved = True
                    if moved:
                        torch.cuda.empty_cache()
                        logging.info("idle: models moved back to CPU")

    def _get(self, model, pretrained):
        key = (model, pretrained)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        logging.info("loading %s / %s", model, pretrained)
        entry = load_model(model, pretrained)
        self._cache[key] = entry
        while len(self._cache) > CACHE_SIZE:
            old_key, old = self._cache.popitem(last=False)
            old.to("cpu")
            logging.info("evicted %s / %s", *old_key)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return entry

    # -- request parsing ---------------------------------------------------

    @staticmethod
    def _device(requested):
        requested = str(requested or "auto").lower()
        if requested not in _DEVICES:
            raise RequestError(f"parameters.device must be one of {list(_DEVICES)}")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RequestError("parameters.device is 'cuda' but no GPU is visible")
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return requested

    @staticmethod
    def _as_list(value, what):
        if value is None:
            return []
        if isinstance(value, (bytes, bytearray, str)):
            return [value]
        try:
            return list(value)
        except TypeError:
            raise RequestError(f"data.{what} must be a list")

    @staticmethod
    def _decode_images(blobs):
        out = []
        for i, raw in enumerate(blobs):
            try:
                out.append(Image.open(io.BytesIO(bytes(raw))).convert("RGB"))
            except Exception:                            # noqa: BLE001
                raise RequestError(f"could not decode image #{i + 1} of {len(blobs)}")
        return out

    # -- commands ----------------------------------------------------------

    def _models(self, parameters):
        known = _pretrained_by_model()
        asked = parameters.get("model")
        if asked:
            name = _norm_model(str(asked))
            if name not in known:
                raise RequestError(f"unknown model {name!r}")
            listing = {name: known[name]}
        else:
            listing = dict(known)
        return _done({
            "default": {"model": _norm_model(DEFAULT_MODEL),
                        "pretrained": DEFAULT_PRETRAINED},
            "loaded": [{"model": m, "pretrained": p} for (m, p) in self._cache],
            "models": listing,
            "num_models": len(listing),
            "open_clip_version": getattr(open_clip, "__version__", None),
        })

    def _encode(self, parameters, images_in, texts_in, start):
        model, pretrained = resolve(parameters.get("model"), parameters.get("pretrained"))
        device = self._device(parameters.get("device"))
        normalize = parameters.get("normalize", True)
        if not isinstance(normalize, bool):
            raise RequestError("parameters.normalize must be true or false")
        try:
            batch_size = int(parameters.get("batch_size", 32))
        except (TypeError, ValueError):
            raise RequestError("parameters.batch_size must be an integer")
        if batch_size < 1:
            raise RequestError("parameters.batch_size must be >= 1")

        images = self._decode_images(images_in)
        texts = [t.decode() if isinstance(t, (bytes, bytearray)) else str(t)
                 for t in texts_in]

        with self._lock:
            self._last_used = time.time()
            entry = self._get(model, pretrained)
            entry.to(device)
            image_emb = text_emb = None
            with torch.no_grad():
                if images:
                    chunks = []
                    for i in range(0, len(images), batch_size):
                        batch = torch.stack([entry.preprocess(im)
                                             for im in images[i:i + batch_size]]).to(device)
                        chunks.append(entry.model.encode_image(batch).float().cpu())
                    image_emb = torch.cat(chunks)
                if texts:
                    chunks = []
                    for i in range(0, len(texts), batch_size):
                        tok = entry.tokenizer(texts[i:i + batch_size]).to(device)
                        chunks.append(entry.model.encode_text(tok).float().cpu())
                    text_emb = torch.cat(chunks)
                logit_scale = float(entry.model.logit_scale.exp()) \
                    if hasattr(entry.model, "logit_scale") else None
            self._last_used = time.time()

        unit = lambda t: torch.nn.functional.normalize(t, dim=-1)       # noqa: E731
        data, encoding, extra = {}, {}, {}
        for name, emb in (("image_emb", image_emb), ("text_emb", text_emb)):
            if emb is not None:
                out = unit(emb) if normalize else emb
                data[name] = wrap_value(np_to_bytes(out.numpy()))
                encoding[name] = "numpy"
                extra["embedding_dim"] = int(emb.shape[-1])
        if image_emb is not None and text_emb is not None:
            cos = unit(image_emb) @ unit(text_emb).T
            data["similarity"] = wrap_value(np_to_bytes(cos.numpy()))
            encoding["similarity"] = "numpy"
            if logit_scale is not None:
                probs = (logit_scale * cos).softmax(dim=-1)
                data["probs"] = wrap_value(np_to_bytes(probs.numpy()))
                encoding["probs"] = "numpy"
                extra["logit_scale"] = logit_scale

        return _done({
            "model": model, "pretrained": pretrained, "device": device,
            "normalized": normalize,
            "num_images": len(images), "num_texts": len(texts),
            "runtime": time.time() - start, **extra,
        }, data=data, encoding=encoding)

    # -- the RPC -----------------------------------------------------------

    def Process(self, request, context):
        start = time.time()
        try:
            if not request.config_json:
                return _error("No config JSON")
            config = json.loads(request.config_json)
            if not isinstance(config, dict) or not isinstance(config.get(BOX_KEY), dict):
                return _error(f"config section {BOX_KEY!r} missing or not an object")
            section = config[BOX_KEY]
            command = section.get("command") or "encode"
            parameters = section.get("parameters") or {}
            if not isinstance(parameters, dict):
                return _error("parameters must be an object")

            if command == "reset":
                with self._lock:
                    unloaded = len(self._cache)
                    for entry in self._cache.values():
                        entry.to("cpu")
                    self._cache.clear()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                return _done({"action": "reset", "unloaded": unloaded})
            if command == "models":
                return self._models(parameters)
            if command != "encode":
                return _error(f"unknown command {command!r} "
                              f"(expected one of {list(_COMMANDS)})")

            images = self._as_list(unwrap_value(request.data["images"])
                                   if "images" in request.data else None, "images")
            texts = self._as_list(unwrap_value(request.data["texts"])
                                  if "texts" in request.data else None, "texts")
            if not images and not texts:
                return _empty_request()
            return self._encode(parameters, images, texts, start)
        except RequestError as e:
            return _error(str(e))
        except Exception as e:                              # noqa: BLE001
            logging.exception("Error in Process")
            return _error(str(e))


# --------------------------------------------------------------------------
# server plumbing (identical in every box)
# --------------------------------------------------------------------------

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
    pipeline_pb2_grpc.add_PipelineServiceServicer_to_server(PipelineService(), server)

    grpc_reflection.enable_server_reflection(
        (pipeline_pb2.DESCRIPTOR.services_by_name["PipelineService"].full_name,
         grpc_reflection.SERVICE_NAME), server)

    run_server(server)
