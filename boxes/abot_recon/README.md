# ABot-Recon Box

[ABot-Recon](https://github.com/amap-cvlab/ABot-Recon) streaming 3D
reconstruction behind the **shared envelope** interface — a "deliberately
local" model that reconstructs long video streams with a fixed 12-frame local
context, composing current-frame geometry and adjacent relative poses into a
global reconstruction.

Unlike the fixed-length clip models in this registry (d4rt), this one is
built for *long* inputs: it never holds persistent learned long-range state,
so per-frame cost and model memory do not grow with the sequence. The paper
reports ~24 FPS at 504×280 on one H100 with 6.7 GiB.

| Command | What it does | State |
|---|---|---|
| `reconstruct` (default) | camera poses, adjacent relative poses, point maps, confidence | none |
| `trajectory` | the poses only (no point maps — cheaper and small) | none |
| `reset` | standard no-op (this box is stateless) | — |

## Build

```bash
cd boxes/abot_recon
docker build --tag sipgisr/visionist-abot-recon --build-arg SERVICE_NAME=abot_recon -f docker/Dockerfile .
```

The build clones ABot-Recon at a **pinned commit** and bakes in the
checkpoint (4.0 GB, Hugging Face `acvlab/ABot-Recon`), so the image is
self-contained and a fleet never waits on a download. That makes it one of
the heavier images in the registry (~11 GB). For a thin image with the
weights mounted instead:

```bash
cd boxes/abot_recon
docker build --tag sipgisr/visionist-abot-recon --build-arg DOWNLOAD_WEIGHTS=false --build-arg SERVICE_NAME=abot_recon -f docker/Dockerfile .
docker run --rm --gpus all -p 8061:8061 -e PORT=8061 -v /path/to:/weights sipgisr/visionist-abot-recon
```

`/weights` must then hold `abot_recon.safetensors` (and may hold `config.json`).

Optional flags:

- `--build-arg LOOP_ASSETS=true` adds the ~450 MB of loop-closure retrieval
  assets (DINOv2-SALAD + DINOv2) to `/weights/loop`. That is what
  `loop_closure: true` in a request needs. The *code* for it is always
  installed, so mounting the two files is an equivalent path.
- `ABOT_RECON_AMP_DTYPE` (default `bf16`) and `ABOT_RECON_ATTENTION_BACKEND`
  (default `auto`) are set per launch, not per request. The `paged` backend
  needs `flashinfer-python`, which the stock image does not carry; `auto`
  falls back to PyTorch SDPA, the upstream's documented no-dependency path.

## Run

```bash
docker run --rm --gpus all -p 8061:8061 -e PORT=8061 sipgisr/visionist-abot-recon
```

## Service usage

### Request

```jsonc
{
  "abot_recon": {
    "command": "reconstruct",      // reconstruct | trajectory | reset
    "parameters": {
      "max_frames": 256,           // cap on accepted frames (>= 3; model allows up to 22000)
      "frame_step": 1,             // keep every Nth frame
      "dense_stride": 1,           // save dense maps every Nth frame
      "loop_closure": false,       // needs the loop assets; refinement of the trajectory
      "output_local_points": true, // reconstruct
      "output_world_points": true, // reconstruct
      "output_confidence": true,   // reconstruct
      "confidence_threshold": 0.0, // reconstruct, [0, 1]
      "device": "cuda"
    }
  }
}
```

`data`:

| field | kind | meaning |
|---|---|---|
| `images` | `bb` | ordered frames, all the same size |
| `video` | `b` | one video file (mp4/avi/webm/mov) |

`images` and `video` are mutually exclusive. Frame order is the input order:
do not send a shuffled sequence.

### Response

```json
{
  "abot_recon": {
    "status": "done",
    "command": "reconstruct",
    "num_frames": 256,
    "num_dense_frames": 128,
    "loop_closure_used": false,
    "input_hw": [1080, 1920],
    "model_hw": [280, 504],
    "device": "cuda",
    "runtime": 21.4,
    "metadata": { "num_frames": 256, "attention_backend": "sdpa", "…": "…" },
    "encoding": { "camera_poses": "numpy", "…": "numpy" }
  }
}
```

Status vocabulary per the shared contract: `done`, `empty_request`, `error`
(reason in `"error"`).

`data` — every field is an `np.save` blob declared `numpy` (T = accepted
frames, Td = frames kept by `dense_stride`):

| field | shape | when | meaning |
|---|---|---|---|
| `camera_poses` | `(T, 4, 4)` | both | camera-to-world per frame; refined when loop closure is used |
| `relative_poses` | `(T-1, 4, 4)` | both | adjacent relative poses, frame i to i+1 |
| `camera_poses_noloop` | `(T, 4, 4)` | both | the raw streaming trajectory |
| `relative_poses_noloop` | `(T-1, 4, 4)` | both | adjacent poses of the raw trajectory |
| `camera_poses_loop` / `relative_poses_loop` | `(T, 4, 4)` / `(T-1, 4, 4)` | loop closure used | the refined trajectory, kept separate |
| `local_points` | `(Td, 280, 504, 3)` | reconstruct | point map per frame, in that frame's camera frame |
| `world_points` | `(Td, 280, 504, 3)` | reconstruct | the maps transformed by the final trajectory — the global cloud |
| `confidence` | `(Td, 280, 504)` | reconstruct | per-pixel confidence |
| `confidence_mask` | `(Td, 280, 504)` | reconstruct | boolean, above `confidence_threshold` |

## Semantics worth knowing

- **Point maps are at a fixed 280×504 grid**, whatever the input resolution —
  the model sees a resized view. Sending denser pixels does not buy
  output pixels; it buys input information.
- **`local_points` vs `world_points`.** Local maps are per camera frame, so a
  static scene appears to move as the camera moves. For one global cloud,
  use `world_points` (the maps carried by the final trajectory).
- **`trajectory` is the cheap verb.** It asks the model for no point maps at
  all; the reply is six small matrices. Run it to check a sequence before
  paying for `reconstruct`.
- **Reply size.** A dense frame costs ~1.7 MB in local points + 1.7 MB in
  world points + 0.56 MB of confidence (float32, 280×504). 256 frames is ~700
  MB; the default message cap is 1024 MB. For long videos prefer
  `trajectory`, or `dense_stride`, or a smaller `max_frames`.
- **Scale.** These are the model's camera-frame estimates; the paper's
  evaluations align trajectories to ground truth before scoring. Do not read
  them as a metric reconstruction without your own scale/alignment.
- **Loop closure is opt-in and asset-gated.** Off by default, and the assets
  are not in the stock image; asking for it without them returns an error
  that names the two missing files.
- **One GPU, one request at a time in effect.** The model is large and the
  work is sequential, so the default worker count is 2. The model loads
  lazily and a watchdog parks it back on CPU after ~120 s idle, so an idle
  box holds no VRAM.
- **GPU in practice.** The code path exists for CPU, but at ~1.0 B
  parameters that is not usable.

## Call with visionist_client

```python
from visionist_client import Visionist
import pathlib, numpy as np

b = Visionist("localhost:8061", timeout=3600)

# --- full reconstruction of a clip --------------------------------------
res = b.run(data   = {"video": pathlib.Path("clip.mp4")},
            config = {"abot_recon": {"command": "reconstruct",
                                     "parameters": {"max_frames": 256,
                                                    "dense_stride": 4}}})
poses = res.camera_poses          # (T, 4, 4), camera-to-world
cloud = res.world_points          # (Td, 280, 504, 3)
conf  = res.confidence            # (Td, 280, 504)
mask  = conf > 0.5
xyz   = cloud[mask[..., None]].reshape(-1, 3)
print(f"{xyz.shape[0]} confident world points; "
      f"trajectory length {np.linalg.norm(np.diff(poses[:, :3, 3], axis=0).sum()):.3f}")

# --- poses only (cheap) --------------------------------------------------
traj = b.run(data  = {"images": [pathlib.Path(f"frame_{i:04d}.jpg") for i in range(64)]},
             config = {"abot_recon": {"command": "trajectory"}})
rel = traj.relative_poses         # (T-1, 4, 4), adjacent relative poses
```

## Testing

```bash
# contract only, no GPU and no weights: the model is stubbed out
cd boxes/abot_recon && python3 test/smoke_inprocess.py

# against a running box
python3 test/test_abot_recon.py
BOX_HOST=10.0.0.5:8061 python3 test/test_abot_recon.py

# the shared contract test every box must pass
python3 ../../contract/conformance/test_conformance.py --box abot_recon --host localhost:8061
```

## Upstream

[amap-cvlab/ABot-Recon](https://github.com/amap-cvlab/ABot-Recon), pinned by
commit in `docker/Dockerfile`. This box uses the documented public API
(`ABotRecon.from_pretrained`, `.infer`), so there is no private-helper pin
risk — re-test the box when you move the pin anyway.

Checkpoint: [`acvlab/ABot-Recon`](https://huggingface.co/acvlab/ABot-Recon)
(`abot_recon.safetensors`, 4.0 GB). Paper and project page:
[arxiv.org/abs/2608.27529](https://arxiv.org/abs/2608.27529),
[amap-cvlab.github.io/ABot-Recon-html](https://amap-cvlab.github.io/ABot-Recon-html/).
Licence: Apache-2.0 (code), model weights per the repo's MODEL_LICENSE.
