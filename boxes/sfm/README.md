# SfM Box (depth-augmented projective structure-from-motion, missing data)

A gRPC box that recovers **camera poses and the 3D point cloud** from feature
tracks plus monocular depth — **tracks may be partial** (a point needs not be
seen in every frame). Monocular depth is only reliable up to a per-frame
affine `Z = d·λ + o`; the box resolves it with multi-view consistency while a
rank-4 matrix completion fills the unobserved entries.

Three solvers (`parameters.solver`, see [Solvers](#solvers)): `completion`
(default, below), `pairs` and `linear`.

The `completion` algorithm is `src/projective_reconstruction.py`, ported from
`~/trackers/src/projective_reconstruction.py` (+ the pieces of
`src/mat_compl.py` / `src/ortho_factorization.py` it runs; dead experiment
branches and the k3d plot left out — the port reproduces the original
bit-for-bit on the synthetic tests). `src/sfm_service.py` is the envelope
wrapper. Only dependency: `torch`.

1. **visibility filter** — iteratively drop frames / points with < 2 observations
2. **`calibrate_with_completion`** — ALS rank-4 completion on the observed
   entries of `(λ + o_f)·W`, **RANSAC track rejection** at iterations
   10/20/30/40, masked per-frame affine depth fit after a 40-iteration warm-up
3. **scale correction** — repeated SVD factorization + metric upgrade
   (Tran & Hartley; det = +1 per Marques & Costeira CVIU 2009) on the
   completed matrix; per-frame scales from the Procrustes singular values
4. **final factorization** → `[R | t]` per frame, aligned to the first
   surviving camera

With complete tracks the completion is exact and this reduces to plain
affine-calibration + factorization.

It speaks the shared **envelope** interface:

```protobuf
service PipelineService {
  rpc Process( Envelope ) returns ( Envelope );
}
```

**CPU-only and stateless** (~0.1–0.6 s for 8–12 frames × 100–1600 points);
`reset` is accepted as a no-op.

## Directory structure

```
sfm_box/
├── docker/
│   └── Dockerfile                     # python:3.10-slim + CPU torch wheel
├── protos/                            # shared proto + generated stubs + aux.py
├── src/
│   ├── projective_reconstruction.py   # completion solver (ported) + warm-start hooks
│   ├── pairs_reconstruction.py        # pairs solver: chained 2-frame solves -> global ping-pong
│   ├── linear_reconstruction.py       # linear solver: 2-frame rotations -> global LM
│   └── sfm_service.py                 # PipelineService.Process(Envelope)
├── test/
│   ├── test_sfm.py                    # live-box test: complete / missing / tracks, reset, errors
│   ├── test_projective_reconstruction.py  # completion core (no box): complete, missing, noise
│   ├── test_linear_reconstruction.py      # linear core (no box): exact recovery + noise
│   └── data/                          # W_mat.npy / lambda_mat.npy (synthetic scene)
├── requirements.txt
└── README.md
```

## Build

```bash
cd boxes/sfm
docker build --tag sipgisr/visionist-sfm:0.1.0 --build-arg SERVICE_NAME=sfm -f docker/Dockerfile .
```

## Run

```bash
docker run --rm -p 8061:8061 -e PORT=8061 sipgisr/visionist-sfm:0.1.0
```

Host ports are assigned by the fleet generator (`tools/make_fleet.py`).

## Service usage

### Request

`config["sfm"]`:

- `command` — `"reconstruct"` (default) or `"reset"` (no-op; the box is
  stateless, but `reset_first` from `visionist_client` stays safe).
- `parameters` (all optional):

| key | default | meaning |
|---|---|---|
| `solver` | `"completion"` | `"completion"`, `"pairs"` or `"linear"` (see [Solvers](#solvers)) |
| `iters` | `100` | completion iterations (`calibrate_with_completion`) / LM iterations (`linear`) |
| `num_scale_iters` | `4` | scale-correction iterations |
| `rank` | `4` | rank of the completion model |
| `removal_iters` | `[10, 20, 30, 40]` | iterations at which RANSAC rejects tracks (`[]` = never; `pairs`: `[]`) |
| `min_obs` | `2` | min observations per point / frame after a removal |
| `offset_mode` | `"normalize"` | `"normalize"` (o − o₀) / `"estimate"` / `"zero"` (`pairs`: `"estimate"`) |
| `seed` | `42` | RANSAC sampling seed (results are reproducible) |
| `intrinsics_normalized` | `false` | mode A: `intrinsics` are normalized (MoGe-style: fx, cx in image widths; fy, cy in image heights) and get scaled by the depth-map size |
| `dtype` | `"float32"` | `"float32"` or `"float64"` |
| `min_points` | `8` | error out if fewer points survive |

`data` — every array is an **`np.save` (`.npy`) blob** (a `torch.save`
tensor is accepted too). Two input modes; send one of them:

**Mode A — tracks + depth maps** (the box builds the rays and samples the depths)

| field | shape | description |
|---|---|---|
| `tracks` | `(2F, P)` | pixel coords, rows `[u0, v0, u1, v1, …]` (x then y per frame); **`NaN` where a point is not seen** |
| `depths` | `(F, H, W)` | monocular depth maps; bilinear-sampled at the tracks |
| `intrinsics` | `(3, 3)` or `(F, 3, 3)` | camera matrix, shared or per frame |

This is the shape `visionist_client.track_stream` (lightglue) produces as
`obs_matrix` (any `min_alive`), and the depth maps a depth box (moge,
unimatch) produces. Note **tapnext** returns tracks as `(y, x)` — swap to
`(x, y)` rows first.

**Mode B — prebuilt matrices** (straight into `run_projective_reconstruction`)

| field | shape | description |
|---|---|---|
| `W_mat` | `(3F, P)` | homogeneous rays `[x, y, 1]` per frame, `x = (u − cx)/fx` |
| `lambda_mat` | `(F, P)` | monocular depth at each track; **`NaN` = missing** |

If both are sent, mode B wins.

**Missing entries.** Entry (frame f, point p) is missing when its track
coordinate is `NaN`, it falls outside the depth map, or its depth sample is
non-finite (e.g. invalid MoGe pixels). Missing entries are **completed**, not
dropped — the pipeline itself removes under-observed frames/points and RANSAC
outlier tracks; `frame_ids` / `point_ids` map what survived back to the input.

### Response

`config_json`:

```json
{"sfm": {"status": "done", "runtime": 0.3, "solver": "completion", "input_mode": "tracks",
         "num_frames_in": 8, "num_points_in": 755, "missing_in": 0.29,
         "num_frames": 8, "num_points": 436, "missing": 0.30,
         "reprojection_error": {"median": 3.0, "mean": 4.1, "p90": 9.2, "unit": "px"},
         "iterations": 100,
         "removed": [{"iter": 10, "points": 310, "frames": 0}, ...],
         "depth_model": "Z = depth_scales * lambda + depth_offsets",
         "encoding": {"cameras": "numpy", ...}}}
```

`status` is `done | empty_request | error` (`error` carries the message).
`missing_in` / `missing` are the unobserved fractions of the input and of the
surviving (frames × points) block; `removed` lists the RANSAC / visibility
removals per completion iteration (`completion` / `pairs`);
`reprojection_error` projects `points` with `cameras` onto the observed
entries (px in mode A, normalized units in mode B) -- the common yardstick of
the three solvers; `depth_model` states how the per-frame depth correction
is applied. `pairs` adds `pairs_fallback` (pairs whose 2-frame solve failed
and were bridged), `linear` adds `converged`, `outliers`, `pairs_fallback`.

`data` (all `np.save` blobs, declared `numpy`); `F'` / `P'` = surviving
frames / points:

| field | shape | description |
|---|---|---|
| `cameras` | `(F', 3, 4)` | per-frame `[R \| t]` (world → camera); the first surviving frame is the reference (`R = I`, `t = 0`) |
| `points` | `(P', 3)` | the 3D points in the reference camera's coordinates |
| `frame_ids` | `(F',)` int64 | input frame of each camera |
| `point_ids` | `(P',)` int64 | input column of each point |
| `observed` | `(F', P')` bool | which entries were observed (the rest were completed) |
| `completed_matrix` | `(3F', P')` | the completed, depth-corrected `Z·W` that is factorized (`Z` as below); `linear`: the model's `R X + t` |
| `depth_scales` | `(F',)` | per-frame slope `d` of the depth correction |
| `depth_offsets` | `(F',)` | per-frame offset `o` of the depth correction (`o₀ = 0`) |

**Depth correction.** The corrected depth of frame `f` is

```
Z = depth_scales[f] * lambda + depth_offsets[f]        # multiply, then add
```

where `lambda` is the raw monocular depth (e.g. a MoGe map); back-project
with `X_cam = Z · K⁻¹ [u, v, 1]` and `X = Rᵀ (X_cam − t)` to land in the
`points` / `cameras` frame. Gauges (the global scale is free — `d ≥ 1`
does not mean the depth is underestimated):

- `completion`: frame 0's offset is anchored to 0, the **smallest scale is
  ≈ 1**. Internally the pipeline estimates `(lambda + o) / s`; the box
  reports `d = 1/s`, `o' = o/s`.
- `pairs`: smallest scale ≈ 1, frame 0's offset is estimated (not anchored).
- `linear`: `d₀ = 1`, frame 0's offset is estimated.

The reconstruction is defined **up to a similarity** (global scale, rotation,
translation, plus one depth offset) — the natural gauge of the problem.

## Solvers

All three take the same inputs and return the same fields.

- **`completion`** (default) — the ported pipeline above: cold start (SVD
  of the column-mean fill), rank-4 ALS completion + affine fit ("ping-pong"),
  RANSAC, scale correction, factorization. Exact on complete tracks; on
  heavily partial (banded) tracks the cold start is far off and it breaks
  down (camera flips, 100s of px).
- **`pairs`** (`src/pairs_reconstruction.py`) — every consecutive 2-frame
  pair is reconstructed with the same pipeline (no missing entries inside a
  pair -> reliable), the pairs are chained through their shared camera
  (pose of the pair's frame + depth-ratio scale: a similarity chain, no 4×4
  homography), and the chain warm-starts ONE global ping-pong over all
  frames. Defaults: `metric` (keep the completion factors rigid — without it
  the global pass drifts away from the good start), `offset_mode
  "estimate"` (the o₀ = 0 anchor biases it), `offset_ridge 1.0` (the
  ping-pong's algebraic affine fit is not scale-invariant: on real video the
  offsets run off to a degenerate solution without a strong ridge — which
  in turn biases scenes with large true offsets, ~1.5° on the test scene).
  Extra parameters: `pair_removal_iters` (`[20]`), `metric` (`true`),
  `offset_ridge` (`1.0`).
- **`linear`** (`src/linear_reconstruction.py`) — rotations from chained
  2-frame solves, an exact linear solve for translations + points, then
  joint Levenberg-Marquardt over `(d, o, R, t, X)` on all observations (no
  completion) with the residual `w − (R X + t) / (d λ + o)` (scale-invariant,
  ~ reprojection error), Huber IRLS, a weak prior on the offsets. Extra
  parameters: `depth_weight` (`1.0`, weight of the depth row vs the image
  rows), `offset_prior` (`0.03`), `loss` (`"huber"` / `null`),
  `loss_scale` (`"auto"`), `loss_k` (`2.5`), `min_frame_obs` (`6`),
  `pair_min_points` (`8`).

Measured (20 frames; synthetic video with 77 % banded missing entries, 2 %
depth + 0.002 ray noise; `cozinha.mp4` with lightglue + MoGe, every 8th
frame, `min_alive` 2–4 = 52–68 % missing):

| solver | synthetic: rotation err / camera-path err | cozinha: reprojection median | cozinha: camera path |
|---|---|---|---|
| `completion` | 82° / 99 % | 38–2467 px | flips (up to 175° per step) |
| `pairs` | 0.7° / 26 % | 14 px | smooth, ~32.5° at every `min_alive` |
| `linear` | 0.8° / 0.8 % | 4–5 px | smooth, ~35° at every `min_alive` |

Runtime: `completion` < 1 s, `pairs` 1–2 s, `linear` 5–10 s (20 frames,
1–3k tracks, CPU).

## Call with visionist_client

```python
import pathlib
from visionist_client import Visionist

b = Visionist("localhost:9071")
res = b.run(
    data={"W_mat":      pathlib.Path("test/data/W_mat.npy"),
          "lambda_mat": pathlib.Path("test/data/lambda_mat.npy")},
    config={"sfm": {"command": "reconstruct",
                    "parameters": {"removal_iters": [10, 20]}}},
)
print(res.config["sfm"]["status"], res.cameras.shape, res.points.shape)
```

Mode A from in-memory arrays (e.g. a lightglue `obs_matrix` + moge depths):

```python
import io, numpy as np

def npy(a):
    buf = io.BytesIO(); np.save(buf, np.asarray(a)); return buf.getvalue()

res = b.run(data={"tracks": npy(obs_matrix),         # (2F, P), NaN gaps ok
                  "depths": npy(depth_maps),         # (F, H, W)
                  "intrinsics": npy(K)},             # (3, 3)
            config={"sfm": {"command": "reconstruct"}})
```

## Test

```bash
# the core alone (no box): complete, ~35% banded missing, missing + noise
python test/test_projective_reconstruction.py

# the linear core: exact recovery without the offset prior, + noise
python test/test_linear_reconstruction.py

# a running box: complete / missing (mode B), tracks with NaN + out-of-image
# entries (mode A) for each of the three solvers, reset, in-band errors
BOX_HOST=localhost:9071 python test/test_sfm.py
```

Expected: `completion` rotation ≈ 0.6° / direction ≈ 0.32°, `pairs` ≈ 1.5° /
1.1° (offset-ridge bias), `linear` ≤ 0.03°, then `PASS`.

## Notes

- **Port vs. original.** Same results as `~/trackers` on the synthetic tests
  (camera difference 0). Changes: `projective_factorization_fast` is the
  det-guarded one from the former `sfm_core.py` (no mirrored cameras under
  noise); RANSAC draws from a per-request seeded `torch.Generator` instead of
  the global seed; prints became logging / the `removed` status field.
- **RANSAC is strict on gappy columns.** At iteration 10 the missing entries
  are filled from a not-yet-converged completion and compared against
  1 % of the median column norm, so columns with gaps are the ones rejected
  (on real lightglue tracks ~40 % of the points, even with complete tracks).
  Tune with `removal_iters` (`[]` disables it).
- **Real data with many gaps.** On `cozinha.mp4` (lightglue + MoGe, 8 frames)
  complete tracks give a smooth 0 → 28° camera rotation; partial tracks with
  ~40 % missing produce inconsistent rotations (jumps to ~68° and back)
  with `completion`. Use `pairs` or `linear` for heavily partial tracks.
- Convergence is to a local minimum of a non-convex objective; the
  monocular depth must start within the basin of attraction.
