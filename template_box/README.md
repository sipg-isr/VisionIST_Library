# template Box

One paragraph: what this box computes, and what it wraps. The per-box README
is the **authoritative** request shape — the manifest is a summary of it, and
clients are written against this page.

| Command | What it does | State |
|---|---|---|
| `run` (default) | the thing this box is for | none |
| `reset` | standard no-op (this box is stateless) | — |

## Build

```bash
cd boxes/template
docker build --tag sipgisr/visionist-template --build-arg SERVICE_NAME=template -f docker/Dockerfile .
```

Or pull the published image:

```bash
docker run --rm -p 8061:8061 -e PORT=8061 docker.io/sipgisr/visionist-template:latest
```

## Request

```jsonc
{
  "template": {
    "command": "run",          // "run" (default) | "reset"
    "parameters": {
      "scale": 1.0
    }
  }
}
```

`data`:

| field | kind | meaning |
|---|---|---|
| `images` | `bb` | one or more images (JPEG/PNG bytes) |

## Response

`config_json` carries a **namespaced status** (never flat):

```json
{
  "template": {
    "status": "done",
    "num_images": 2,
    "scale": 1.0,
    "runtime": 0.01,
    "encoding": { "sizes": "numpy" }
  }
}
```

Status vocabulary per the shared contract: `done`, `empty_request` (no usable
input), `error` (reason in `"error"`).

`data`:

| field | kind | description |
|---|---|---|
| `sizes` | `b` (`numpy`) | `np.save` blob, `(num_images, 2)`: width and height of each decoded image, times `scale` |

## Semantics worth knowing

Put here the things a caller gets wrong otherwise: how many images a command
needs, which fields are absent rather than empty in some cases, whether
anything is cumulative across calls, what an invalid pixel looks like.

## Call with visionist_client

```python
from visionist_client import Visionist
import pathlib

b = Visionist("localhost:8061")
res = b.run(data   = {"images": [pathlib.Path("test/sample.jpg")]},
            config = {"template": {"command": "run",
                                   "parameters": {"scale": 2.0}}})
print(res.config["template"])
print(res.sizes)            # already decoded: np.ndarray (1, 2)
```

## Testing

```bash
python3 test/test_template.py
BOX_HOST=10.0.0.5:8061 python3 test/test_template.py

# the shared contract test, which every box must pass
python3 ../../contract/conformance/test_conformance.py --box template --host localhost:8061
```
