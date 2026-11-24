"""internvl box - InternVL3 visual question answering.

Images and a question in, text answers out. Wraps the InternVL3 chat models
(OpenGVLab/InternVL3-*) through their own `chat` / `batch_chat` methods, so a
checkpoint fine-tuned with the InternVL training code - OpenFACADES' building
models, for instance - loads exactly like the upstream ones.

What this box takes from OpenFACADES (src/openfacades/vlm/) and what it changes:

* the image tiling (`dynamic_preprocess`) is the same algorithm, but it works
  on decoded bytes in memory - images never touch the server's disk - and the
  normalisation is done in numpy, so torchvision is not a dependency;
* the model is loaded ONCE, at startup, and kept warm. OpenFACADES rebuilt it
  on every `building_vlm()` call;
* the device and precision are chosen, not hard-coded: no `.cuda()` after
  `device_map='auto'` (they conflict), bf16 only where the GPU has it
  (compute capability >= 8.0, i.e. Ampere and newer), fp16 on older cards such
  as an RTX 2080, fp32 on CPU;
* flash-attn is optional (INTERNVL_FLASH_ATTN=1). The default is the model's
  own eager attention, which runs on every GPU and on CPU;
* answers come back in the reply rather than being appended to a JSON file.

The weights are baked into the image at build time (docker/fetch_weights.py);
the box never reaches the network at request time (HF_HUB_OFFLINE=1).

Environment (read at startup):
  INTERNVL_MODEL_DIR   where the checkpoint lives        (/weights/model)
  INTERNVL_MODEL_ID    the name reported in replies      (from weights/MODEL_ID)
  INTERNVL_DEVICE      auto | cpu | cuda | cuda:N | split (auto)
                       `split` spreads the language model over every visible
                       GPU, for checkpoints too big for one card
  INTERNVL_DTYPE       auto | bf16 | fp16 | fp32         (auto)
  INTERNVL_FLASH_ATTN  0 | 1                             (0)
  INTERNVL_PRELOAD     1 loads the model at startup, 0 on first request (1)
"""

import concurrent.futures as futures
import io
import json
import logging
import math
import os
import re
import sys
import threading
import time

sys.path.append("./protos")
import pipeline_pb2 as folder_wd_pb2  # noqa: E402
import pipeline_pb2_grpc as folder_wd_pb2_grpc  # noqa: E402
from aux import wrap_value, unwrap_value, as_list  # noqa: E402
import serve  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

# torch and transformers are imported lazily, inside InternVLEngine, so the
# request handling and the tiling can be tested without either installed.

_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_PORT_ENV_VAR = "PORT"

#: The config section this box answers to. Must equal `key` in box.yaml.
BOX_KEY = "internvl"

COMMANDS = ("ask", "info", "reset")
#: Accepted as `ask`, so a caller written against the template still works.
ALIASES = {"run": "ask"}

DEFAULT_QUESTION = "Describe this image in detail."

_DEFAULTS = {
    "question": None,          # falls back to data `texts`, then DEFAULT_QUESTION
    "mode": "each",            # each | together
    "max_tiles": 12,
    "use_thumbnail": True,
    "max_new_tokens": 512,
    "do_sample": False,
    "temperature": 0.7,
    "top_p": 0.9,
    "batch_size": 4,
    "parse_json": False,
}

_LIMITS = {
    "max_tiles": (1, 40),
    "max_new_tokens": (1, 8192),
    "batch_size": (1, 64),
}

IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


# ---------------------------------------------------------------- envelope
def _done(extra, data=None, encoding=None):
    section = {"status": "done", **extra}
    if encoding:
        section["encoding"] = encoding
    return folder_wd_pb2.Envelope(
        config_json=json.dumps({BOX_KEY: section}), data=data or {})


def _empty_request(reason):
    return folder_wd_pb2.Envelope(config_json=json.dumps(
        {BOX_KEY: {"status": "empty_request", "reason": reason}}))


def _error(message):
    return folder_wd_pb2.Envelope(
        config_json=json.dumps({BOX_KEY: {"status": "error", "error": message}}))


def _json_bytes(obj) -> bytes:
    """The client's `json` codec decodes UTF-8 bytes."""
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------- tiling
# InternVL's dynamic-resolution preprocessing, as published on the model cards
# and copied into OpenFACADES' vlm/base.py. Kept algorithm-for-algorithm so a
# fine-tuned checkpoint sees exactly the tiles it was trained on.
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
                              image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448,
                       use_thumbnail=False):
    """Split a PIL image into image_size x image_size tiles on the grid whose
    aspect ratio is closest to the image's, plus an optional thumbnail."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    cols = target_width // image_size

    resized_img = image.resize((target_width, target_height))
    tiles = []
    for i in range(blocks):
        box = ((i % cols) * image_size,
               (i // cols) * image_size,
               ((i % cols) + 1) * image_size,
               ((i // cols) + 1) * image_size)
        tiles.append(resized_img.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def tiles_to_array(tiles, image_size):
    """Tiles -> float32 (N, 3, S, S), normalised with ImageNet statistics.

    Equivalent to torchvision's Resize(BICUBIC) + ToTensor + Normalize for
    tiles that dynamic_preprocess has already sized; resizing again is a no-op
    then, but is kept for any tile that is not."""
    out = np.empty((len(tiles), 3, image_size, image_size), dtype=np.float32)
    for k, tile in enumerate(tiles):
        if tile.mode != "RGB":
            tile = tile.convert("RGB")
        if tile.size != (image_size, image_size):
            tile = tile.resize((image_size, image_size), Image.BICUBIC)
        arr = np.asarray(tile, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        out[k] = arr.transpose(2, 0, 1)
    return out


def decode_image(raw, index, total):
    try:
        img = Image.open(io.BytesIO(bytes(raw)))
        img.load()
    except Exception as e:                                  # noqa: BLE001
        raise ValueError(
            f"could not decode image #{index + 1} of {total}: {e}") from None
    return img.convert("RGB")


def parse_json_answer(text):
    """Best effort: the answer as JSON, tolerating a ```json fence or prose
    around one object/array. None when there is nothing parseable."""
    if not isinstance(text, str):
        return None
    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.S)
    if fence:
        candidate = fence.group(1).strip()
    try:
        return json.loads(candidate)
    except ValueError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = candidate.find(open_c), candidate.rfind(close_c)
        if 0 <= start < end:
            try:
                return json.loads(candidate[start:end + 1])
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------- model
def split_device_map(num_layers, world_size):
    """Spread the LLM layers over GPUs, keeping GPU 0 light because it also
    holds the vision tower. From InternVL's model cards / OpenFACADES base.py,
    generalised to read the layer count from the checkpoint's config."""
    device_map = {}
    per_gpu = math.ceil(num_layers / (world_size - 0.5))
    counts = [per_gpu] * world_size
    counts[0] = math.ceil(per_gpu * 0.5)
    layer = 0
    for gpu, n in enumerate(counts):
        for _ in range(n):
            if layer >= num_layers:
                break
            device_map[f"language_model.model.layers.{layer}"] = gpu
            layer += 1
    for name in ("vision_model", "mlp1",
                 "language_model.model.tok_embeddings",
                 "language_model.model.embed_tokens",
                 "language_model.output", "language_model.model.norm",
                 "language_model.model.rotary_emb",
                 "language_model.lm_head",
                 f"language_model.model.layers.{num_layers - 1}"):
        device_map[name] = 0
    return device_map


class InternVLEngine:
    """Owns the model. Everything torch-specific lives here."""

    def __init__(self, model_dir=None, model_id=None, device=None, dtype=None,
                 flash_attn=None):
        self.model_dir = model_dir or os.getenv("INTERNVL_MODEL_DIR",
                                                "/weights/model")
        self.model_id = model_id or os.getenv("INTERNVL_MODEL_ID") \
            or self._read_model_id() or os.path.basename(self.model_dir)
        self.device_req = (device or os.getenv("INTERNVL_DEVICE", "auto")).lower()
        self.dtype_req = (dtype or os.getenv("INTERNVL_DTYPE", "auto")).lower()
        self.flash_attn = flash_attn if flash_attn is not None else \
            os.getenv("INTERNVL_FLASH_ATTN", "0").strip().lower() in ("1", "true", "yes", "on")
        self.model = None
        self.tokenizer = None
        self.image_size = 448
        self.device = None          # where pixel_values go
        self.dtype_name = None
        self.load_error = None
        self._load_lock = threading.Lock()
        self._gen_lock = threading.Lock()

    def _read_model_id(self):
        try:
            with open(os.path.join(self.model_dir, "MODEL_ID")) as f:
                return f.read().strip() or None
        except OSError:
            return None

    # -- loading -----------------------------------------------------------
    def ensure_loaded(self):
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            if self.load_error is not None:
                raise RuntimeError(self.load_error)
            try:
                self._load()
            except Exception as e:                          # noqa: BLE001
                logging.exception("Model load failed")
                self.load_error = f"model failed to load from {self.model_dir}: {e}"
                raise RuntimeError(self.load_error) from None

    def _pick(self, torch):
        cuda = torch.cuda.is_available()
        dev = self.device_req
        if dev == "auto":
            dev = "cuda" if cuda else "cpu"
        if (dev.startswith("cuda") or dev == "split") and not cuda:
            logging.warning("%s requested but CUDA is not available; using CPU",
                            dev)
            dev = "cpu"
        if dev == "split" and torch.cuda.device_count() < 2:
            dev = "cuda"

        name = self.dtype_req
        if name == "auto":
            if dev == "cpu":
                name = "fp32"
            else:
                idx = 0 if dev in ("cuda", "split") else int(dev.split(":")[1])
                major, _ = torch.cuda.get_device_capability(idx)
                # Native bf16 is Ampere (sm_80) and newer. Turing (RTX 20xx,
                # T4) and Volta emulate it slowly or not at all.
                name = "bf16" if major >= 8 else "fp16"
        dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16,
                  "fp32": torch.float32}
        if name not in dtypes:
            raise ValueError(f"INTERNVL_DTYPE must be auto|bf16|fp16|fp32, got {name!r}")
        return dev, name, dtypes[name]

    def _load(self):
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                f"{self.model_dir} does not exist - build the image with its "
                f"weights, or mount a checkpoint there")

        dev, dtype_name, dtype = self._pick(torch)
        t0 = time.time()
        kwargs = dict(torch_dtype=dtype, low_cpu_mem_usage=True,
                      trust_remote_code=True, use_flash_attn=self.flash_attn)
        if dev == "split":
            cfg = AutoConfig.from_pretrained(self.model_dir, trust_remote_code=True)
            n_layers = cfg.llm_config.num_hidden_layers
            kwargs["device_map"] = split_device_map(n_layers,
                                                    torch.cuda.device_count())
            model = AutoModel.from_pretrained(self.model_dir, **kwargs).eval()
            self.device = "cuda:0"
        else:
            model = AutoModel.from_pretrained(self.model_dir, **kwargs).eval()
            model = model.to(dev)
            self.device = dev
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, trust_remote_code=True, use_fast=False)

        cfg = getattr(model, "config", None)
        size = getattr(cfg, "force_image_size", None) or \
            getattr(getattr(cfg, "vision_config", None), "image_size", None)
        self.image_size = int(size) if size else 448

        self.model, self.tokenizer = model, tokenizer
        self.dtype_name = dtype_name
        self._dtype = dtype
        self._device_label = dev
        logging.info("InternVL %s loaded on %s (%s, flash_attn=%s) in %.1fs",
                     self.model_id, dev, dtype_name, self.flash_attn,
                     time.time() - t0)

    # -- metadata ----------------------------------------------------------
    def describe(self):
        return {"model": self.model_id,
                "loaded": self.model is not None,
                "device": getattr(self, "_device_label", None),
                "dtype": self.dtype_name,
                "flash_attn": self.flash_attn,
                "image_size": self.image_size,
                **({"load_error": self.load_error} if self.load_error else {})}

    # -- inference ---------------------------------------------------------
    def _to_tensor(self, arrays):
        import torch
        return torch.from_numpy(np.concatenate(arrays, axis=0)).to(
            device=self.device, dtype=self._dtype)

    def _generation_config(self, p):
        cfg = {"max_new_tokens": p["max_new_tokens"], "do_sample": p["do_sample"]}
        if p["do_sample"]:
            cfg.update(temperature=p["temperature"], top_p=p["top_p"])
        return cfg

    def answer_each(self, arrays, questions, p):
        """One answer per image, in batches of p['batch_size']."""
        import torch
        gen = self._generation_config(p)
        answers = []
        with self._gen_lock, torch.inference_mode():
            for start in range(0, len(arrays), p["batch_size"]):
                chunk = arrays[start:start + p["batch_size"]]
                qs = questions[start:start + p["batch_size"]]
                pixel_values = self._to_tensor(chunk)
                if len(chunk) == 1:
                    answers.append(self.model.chat(
                        self.tokenizer, pixel_values, qs[0], gen))
                else:
                    answers.extend(self.model.batch_chat(
                        self.tokenizer, pixel_values,
                        num_patches_list=[a.shape[0] for a in chunk],
                        questions=qs, generation_config=gen))
        return answers

    def answer_together(self, arrays, question, p):
        """One answer about all the images, each tagged Image-k."""
        import torch
        gen = self._generation_config(p)
        if "<image>" not in question:
            if len(arrays) == 1:
                question = "<image>\n" + question
            else:
                question = "".join(f"Image-{k + 1}: <image>\n"
                                   for k in range(len(arrays))) + question
        with self._gen_lock, torch.inference_mode():
            pixel_values = self._to_tensor(arrays)
            return [self.model.chat(
                self.tokenizer, pixel_values, question, gen,
                num_patches_list=[a.shape[0] for a in arrays])]

    @staticmethod
    def is_oom(exc):
        try:
            import torch
            return isinstance(exc, torch.cuda.OutOfMemoryError)
        except Exception:                                   # noqa: BLE001
            return False

    @staticmethod
    def free_cache():
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                                   # noqa: BLE001
            pass


# ---------------------------------------------------------------- service
def _parameters(raw):
    """Merge with defaults and check types/ranges. Raises ValueError."""
    if not isinstance(raw, dict):
        raise ValueError("parameters must be an object")
    p = dict(_DEFAULTS)
    p.update({k: v for k, v in raw.items() if v is not None})

    for name, (lo, hi) in _LIMITS.items():
        try:
            value = int(p[name])
        except (TypeError, ValueError):
            raise ValueError(f"parameters.{name} must be an integer") from None
        if not lo <= value <= hi:
            raise ValueError(f"parameters.{name} must be in [{lo}, {hi}], got {value}")
        p[name] = value
    for name in ("temperature", "top_p"):
        try:
            p[name] = float(p[name])
        except (TypeError, ValueError):
            raise ValueError(f"parameters.{name} must be a number") from None
    for name in ("use_thumbnail", "do_sample", "parse_json"):
        value = p[name]
        # The web console's select widget and MATLAB callers send "true" /
        # "false" as strings; take those, refuse anything else.
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            value = value.strip().lower() == "true"
        if not isinstance(value, bool):
            raise ValueError(f"parameters.{name} must be true or false")
        p[name] = value
    if p["mode"] not in ("each", "together"):
        raise ValueError(f"parameters.mode must be 'each' or 'together', got {p['mode']!r}")
    if p["question"] is not None and not isinstance(p["question"], str):
        raise ValueError("parameters.question must be a string")
    return p


def _questions(texts, p, num_images):
    """Pair questions with images.

    `each`: no question -> the default for every image; one -> shared by every
    image; one per image -> paired in order. `together`: exactly one."""
    if texts and p["question"] is not None:
        raise ValueError("give the question either as data `texts` or as "
                         "parameters.question, not both")
    if not texts:
        texts = [p["question"] or DEFAULT_QUESTION]
    if any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("every question must be a non-empty string")
    if p["mode"] == "together":
        if len(texts) != 1:
            raise ValueError(f"mode 'together' takes one question, got {len(texts)}")
        return texts
    if len(texts) == 1:
        return texts * num_images
    if len(texts) != num_images:
        raise ValueError(f"{len(texts)} questions for {num_images} images: send "
                         f"one question for all, or exactly one per image")
    return texts


class PipelineService(folder_wd_pb2_grpc.PipelineServiceServicer):

    def __init__(self, engine=None):
        self.engine = engine or InternVLEngine()

    def Process(self, request, context):
        start = time.time()
        try:
            if not request.config_json:
                return _error("No config JSON")
            config = json.loads(request.config_json)
            if not isinstance(config, dict) or not isinstance(config.get(BOX_KEY), dict):
                return _error(f"config section {BOX_KEY!r} missing or not an object")
            section = config[BOX_KEY]
            command = section.get("command") or "ask"
            command = ALIASES.get(command, command)

            if command == "reset":
                return _done({"action": "reset"})
            if command == "info":
                return _done({"command": "info", **self.engine.describe()})
            if command != "ask":
                return _error(f"unknown command {command!r} (expected one of "
                              f"{', '.join(COMMANDS)})")

            try:
                p = _parameters(section.get("parameters", {}))
            except ValueError as e:
                return _error(str(e))

            images = as_list(unwrap_value(request.data["images"])
                             if "images" in request.data else None)
            if not images:
                return _empty_request("no images")
            texts = as_list(unwrap_value(request.data["texts"])
                            if "texts" in request.data else None)
            try:
                questions = _questions(texts, p, len(images))
            except ValueError as e:
                return _error(str(e))

            self.engine.ensure_loaded()

            size = self.engine.image_size
            arrays = []
            for i, raw in enumerate(images):
                try:
                    img = decode_image(raw, i, len(images))
                except ValueError as e:
                    return _error(str(e))
                tiles = dynamic_preprocess(img, image_size=size,
                                           max_num=p["max_tiles"],
                                           use_thumbnail=p["use_thumbnail"])
                arrays.append(tiles_to_array(tiles, size))

            try:
                if p["mode"] == "together":
                    answers = self.engine.answer_together(arrays, questions[0], p)
                else:
                    answers = self.engine.answer_each(arrays, questions, p)
            except Exception as e:                          # noqa: BLE001
                if self.engine.is_oom(e):
                    self.engine.free_cache()
                    return _error("GPU out of memory - lower parameters.max_tiles "
                                  "or parameters.batch_size, or send fewer images")
                raise

            data = {"answers": wrap_value(_json_bytes(answers))}
            encoding = {"answers": "json"}
            extra = {}
            if p["parse_json"]:
                parsed = [parse_json_answer(a) for a in answers]
                data["parsed"] = wrap_value(_json_bytes(parsed))
                encoding["parsed"] = "json"
                extra["num_parsed"] = sum(x is not None for x in parsed)

            info = self.engine.describe()
            return _done(
                {"command": "ask", "mode": p["mode"],
                 "model": info["model"], "device": info["device"],
                 "dtype": info["dtype"],
                 "num_images": len(images), "num_answers": len(answers),
                 "tiles": [int(a.shape[0]) for a in arrays],
                 "max_new_tokens": p["max_new_tokens"], **extra,
                 "runtime": time.time() - start},
                data=data, encoding=encoding)
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


def _try_load(engine):
    try:
        engine.ensure_loaded()
    except Exception:                                       # noqa: BLE001
        pass  # already logged; requests report engine.load_error


if __name__ == "__main__":
    import grpc
    import grpc_reflection.v1alpha.reflection as grpc_reflection

    logging.basicConfig(
        format="[ %(levelname)s ] %(asctime)s (%(module)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    service = PipelineService()
    if os.getenv("INTERNVL_PRELOAD", "1").strip().lower() in ("1", "true", "yes", "on"):
        # Load in the background so the port opens at once: `reset`, `info`
        # and the conformance test answer while the weights are still
        # loading, and the first `ask` waits for them.
        threading.Thread(target=lambda: _try_load(service.engine),
                         daemon=True).start()

    # One worker by default: generation is serialised on the GPU anyway.
    workers = serve.max_workers(1)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=workers),
        options=serve.server_options(),
        maximum_concurrent_rpcs=serve.max_concurrent_rpcs(workers))
    folder_wd_pb2_grpc.add_PipelineServiceServicer_to_server(service, server)

    grpc_reflection.enable_server_reflection(
        (folder_wd_pb2.DESCRIPTOR.services_by_name["PipelineService"].full_name,
         grpc_reflection.SERVICE_NAME), server)

    run_server(server)
