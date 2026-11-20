# internvl Box

A vision-language model behind the standard envelope: send images and a
question, get text answers back. It wraps the
[InternVL3](https://github.com/OpenGVLab/InternVL) chat models through their
own `chat` / `batch_chat`, so any InternVL3 checkpoint loads the same way,
including ones fine-tuned with the InternVL training code, such as the
[OpenFACADES](https://github.com/seshing/OpenFACADES) building models.
Use it for captions, attribute questions ("what is the facade made of?"), or a
JSON record per image (`parse_json`).

| Command | What it does | State |
|---|---|---|
| `ask` (default; `run` is accepted too) | one answer per image, or one answer about several images | none |
| `info` | which model is loaded, on what device and in what precision; needs no data | none |
| `reset` | standard no-op (this box is stateless) | — |

## Build

The checkpoint is chosen when the image is built and is baked into it: the box
never reaches the network at request time. The default is
`OpenGVLab/InternVL3-1B` (~2 GB).

```bash
cd boxes/internvl
docker build --tag sipgisr/visionist-internvl --build-arg SERVICE_NAME=internvl -f docker/Dockerfile .
```

Another size, or your own fine-tuned checkpoint on the Hugging Face Hub, pinned
to a commit:

```bash
cd boxes/internvl
docker build --tag sipgisr/visionist-internvl --build-arg SERVICE_NAME=internvl \
    --build-arg MODEL_ID=OpenGVLab/InternVL3-8B --build-arg MODEL_REVISION=<commit> \
    -f docker/Dockerfile .
```

`--build-arg DOWNLOAD_WEIGHTS=false` builds a thin image; mount a checkpoint at
`/weights/model` when you run it (`-v /path/to/checkpoint:/weights/model:ro`).

Run it:

```bash
docker run --rm --gpus all -p 9081:8061 -e PORT=8061 sipgisr/visionist-internvl
```

Or run the published image without building - `docker pull
sipgisr/visionist-internvl` is the whole upgrade procedure.

The port opens at once; the weights load in the background, and the first
`ask` waits for them.

### GPUs

The image is built on PyTorch 2.7 / CUDA 12.8, which has kernels for every
NVIDIA generation from Turing (RTX 20xx) to Blackwell (RTX 50xx, B200).
Precision is chosen per card: **bf16** on Ampere and newer (RTX 30xx/40xx/50xx,
A100, H100), **fp16** on older cards such as the RTX 2080, which have no native
bf16, and **fp32** on CPU. Rough GPU memory for the weights alone, in 16-bit:
1B ≈ 2 GB, 2B ≈ 4 GB, 8B ≈ 16 GB. Activations grow with `max_tiles` ×
`batch_size`.

| Environment | Default | Meaning |
|---|---|---|
| `INTERNVL_DEVICE` | `auto` | `cpu`, `cuda`, `cuda:N`, or `split` to spread a big model over all visible GPUs |
| `INTERNVL_DTYPE` | `auto` | `bf16`, `fp16` or `fp32` |
| `INTERNVL_FLASH_ATTN` | `0` | `1` uses flash-attn; the image must be built with `--build-arg INSTALL_FLASH_ATTN=true`, Ampere or newer only |
| `INTERNVL_PRELOAD` | `1` | `0` loads the model on the first `ask` instead of at start |
| `INTERNVL_MODEL_DIR` | `/weights/model` | where the checkpoint is |

## Request

```jsonc
{
  "internvl": {
    "command": "ask",                 // "ask" (default) | "info" | "reset"
    "parameters": {
      "question": "Describe this image in detail.",   // or send data `texts`
      "mode": "each",                 // "each" | "together"
      "max_tiles": 12,                // 1-40 tiles of 448x448 per image
      "use_thumbnail": true,
      "max_new_tokens": 512,
      "do_sample": false,             // true -> temperature, top_p apply
      "temperature": 0.7,
      "top_p": 0.9,
      "batch_size": 4,                // images per forward pass, mode "each"
      "parse_json": false             // also return each answer parsed as JSON
    }
  }
}
```

`data`:

| field | kind | meaning |
|---|---|---|
| `images` | `bb` | one or more images (JPEG/PNG bytes). Required. |
| `texts` | `ss` | the question(s). Mode `each`: one question for every image, or one per image in order. Mode `together`: exactly one. Absent → `parameters.question`. |

## Response

```json
{
  "internvl": {
    "status": "done",
    "command": "ask",
    "mode": "each",
    "model": "OpenGVLab/InternVL3-1B",
    "device": "cuda",
    "dtype": "bf16",
    "num_images": 2,
    "num_answers": 2,
    "tiles": [13, 7],
    "max_new_tokens": 512,
    "runtime": 1.84,
    "encoding": { "answers": "json" }
  }
}
```

Status vocabulary per the shared contract: `done`, `empty_request` (no
images), `error` (reason in `"error"`).

`data`:

| field | kind | description |
|---|---|---|
| `answers` | `b` (`json`) | list of strings: one per image (`each`), or one (`together`) |
| `parsed` | `b` (`json`) | only with `parse_json`: each answer parsed as JSON, `null` where it does not parse. A ```` ```json ```` fence or prose around the object is tolerated. `num_parsed` in the config counts the successes. |

## Semantics worth knowing

* **The model is fixed per image.** There is no `model` parameter: switching
  checkpoints at request time would mean downloading at request time. Build one
  image per model and run them on different ports. `info` says which one is in
  a running box.
* **One question source.** Data `texts` or `parameters.question`, not both.
  In mode `each`, 2 questions for 3 images is an error rather than a guess.
* **Tiling.** Each image is resized to the grid of 448×448 tiles closest to
  its aspect ratio, with at most `max_tiles` tiles, plus a thumbnail. `tiles`
  in the reply says how many each image used. A 4:3 photo becomes 12 + 1 tiles
  with the default and 1 tile with `max_tiles: 4`; a 3:1 crop of a panorama
  becomes 3×1. Lower `max_tiles` first when you run out of GPU memory.
* **Mode `together`** shows the model all the images in one conversation,
  labelled `Image-1`, `Image-2`, ... ; refer to them that way in the question.
  If the question contains `<image>` placeholders itself, they are used as
  given.
* **Determinism.** `do_sample: false` (the default) is greedy decoding: the
  same image and question give the same answer.
* **CPU** works, in fp32, and is slow; use it for a quick check, not a batch.

## Call with visionist_client

```python
from visionist_client import Visionist
import pathlib

b = Visionist("localhost:9081")
print(b.run(data={}, config={"internvl": {"command": "info"}}).config["internvl"])

# one question for every image
res = b.run(data   = {"images": [pathlib.Path("facade1.jpg"), pathlib.Path("facade2.jpg")],
                      "texts":  ["What is the main material of this building's facade?"]},
            config = {"internvl": {"parameters": {"max_tiles": 6}}})
print(res.answers)          # already decoded: ['Red brick.', 'Glass and steel.']

# a JSON record per image, OpenFACADES style
q = ('Return JSON only: {"building_type": ..., "surface_material": ..., '
     '"floors": <int>, "construction_era": ...}')
res = b.run(data   = {"images": [pathlib.Path("facade1.jpg")]},
            config = {"internvl": {"parameters": {"question": q, "parse_json": True}}})
print(res.parsed[0])

# one answer about two images
res = b.run(data   = {"images": [pathlib.Path("before.jpg"), pathlib.Path("after.jpg")]},
            config = {"internvl": {"parameters": {"mode": "together",
                       "question": "What changed between Image-1 and Image-2?"}}})
print(res.answers[0])
```

## Testing

```bash
cd boxes/internvl
python3 test/test_internvl.py                       # no model, no GPU: tiling, pairing, envelope
python3 test/test_internvl.py --host localhost:9081 # a running box: real answers

# the shared contract test, which every box must pass
python3 ../../contract/conformance/test_conformance.py --box internvl --host localhost:9081
```

The first form serves the box in-process with a stand-in for the model, so it
needs neither torch nor the weights; it needs `grpcio-tools` to compile the
protos.
