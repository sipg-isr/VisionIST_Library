# UniMatch Box

Unified dense matching behind the **shared envelope** interface: optical flow,
stereo disparity and multi-view depth from one model family.

| Command | What it does | Images in |
|---|---|---|
| `flow` (default) | per-pixel motion between two frames | exactly 2 |
| `stereo` | per-pixel disparity from a rectified pair | exactly 2 |
| `depth` | metric depth, given camera intrinsics | 1 or 2 |
| `reset` | standard no-op (this box is stateless) | — |

Only the **flow** checkpoint ships inside the image. `stereo` and `depth` need
`parameters.model` pointed at a checkpoint — a name, a local path, or an
`http(s)` URL that is fetched once on first use.

## Build

```bash
cd boxes/unimatch
docker build --tag sipgisr/visionist-unimatch --build-arg SERVICE_NAME=unimatch -f docker/Dockerfile .
```

Or pull the published image:

```bash
docker run --rm --gpus all -p 8061:8061 -e PORT=8061 docker.io/sipgisr/visionist-unimatch:latest
```

## Request

```jsonc
{
  "unimatch": {
    "command": "flow",                 // flow | stereo | depth | reset
    "parameters": {
      "model": "…gmstereo-…pth",       // required for stereo / depth
      "inference_size": [800, 1120],   // force the internal resolution
      "padding_factor": 16,
      "num_scales": 1,
      "attn_type": "swin",
      "attn_splits_list": [2],         // per-scale; length must equal num_scales
      "corr_radius_list": [-1],
      "prop_radius_list": [-1],
      "reg_refine": false,             // needs num_scales <= 2
      "device": "cuda"
    }
  }
}
```

`flow` also takes `pred_bidir_flow` (adds the backward flow) and
`fwd_bwd_check` (occlusion masks; requires `pred_bidir_flow`).

`depth` also takes `intrinsics` (**required**: 3×3, 4×4, or
`[fx, fy, cx, cy]`), `pose` (4×4 relative, reference → target) or
`pose_ref` + `pose_tgt`, `min_depth` (0.5), `max_depth` (10.0),
`num_depth_candidates` (64) and `pred_bidir_depth`.

`data`:

| field | kind | meaning |
|---|---|---|
| `images` | `bb` | `flow`/`stereo`: exactly 2. `depth`: 1 (monocular, identity pose) or 2 |

## Response

```json
{
  "unimatch": {
    "status": "done",
    "command": "flow",
    "runtime": 1.84,
    "auto_resized": false,
    "encoding": { "flow": "numpy" }
  }
}
```

Status vocabulary per the shared contract: `done`, `empty_request`, `error`
(reason in `"error"`).

`data` — every field is an `np.save` blob declared `numpy`, at the **input**
resolution:

| field | shape | when | meaning |
|---|---|---|---|
| `flow` | `(H, W, 2)` | `flow` | where each pixel of image 1 went in image 2, in pixels: `(..., 0)` is horizontal, `(..., 1)` vertical |
| `bwd_flow` | `(H, W, 2)` | `pred_bidir_flow` | the same, image 2 → image 1 |
| `occ_fwd`, `occ_bwd` | `(H, W)` | `fwd_bwd_check` | occlusion masks |
| `disparity` | `(H, W)` | `stereo` | left-view disparity in pixels; bigger = closer |
| `depth` | `(H, W)` | `depth` | metres |

## Semantics worth knowing

- **Large inputs are downscaled inside the box.** Anything much above ~1.6 MP
  is resized for inference and the result is scaled back to the input
  resolution; the reply sets `auto_resized: true` so you know it happened.
  `inference_size` overrides the automatic choice.
- **Disparity to metric depth** needs a calibrated rig:
  `depth_m ≈ baseline_m * fx / disparity`.
- **Per-scale lists** (`attn_splits_list`, `corr_radius_list`,
  `prop_radius_list`) must each have exactly `num_scales` entries.
- Stateless: `reset` is the standard no-op.

## Call with visionist_client

```python
from visionist_client import Visionist
import pathlib, numpy as np

b = Visionist("localhost:9070")

res = b.run(data   = {"images": [pathlib.Path("test/flow_0.jpg"),
                                 pathlib.Path("test/flow_1.jpg")]},
            config = {"unimatch": {"command": "flow"}})

flow = res.flow                       # (H, W, 2) ndarray
print(f"mean shift {np.linalg.norm(flow, axis=2).mean():.2f} px")
```

Stereo, fetching the checkpoint on first use:

```python
res = b.run(
    data   = {"images": [left, right]},
    config = {"unimatch": {"command": "stereo", "parameters": {
        "model": "https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/"
                 "pretrained/gmstereo-scale1-sceneflow-124a438f.pth",
        "inference_size": [800, 1120]}}})
d = res.disparity
```

## Testing

```bash
python3 ../../tools/fetch_assets.py unimatch   # the stereo pair is a declared asset
python3 test/test_unimatch.py
BOX_HOST=10.0.0.5:8061 python3 test/test_unimatch.py
```

Upstream: [autonomousvision/unimatch](https://github.com/autonomousvision/unimatch).
