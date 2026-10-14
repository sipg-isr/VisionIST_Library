# colmap - ground-truth SfM box

**COLMAP is the reference solver.** This box is the *arbiter* of camera
poses/points in the VisionIST fleet: when a fast box (`sfm`, `d4rt`,
`tapnext`) is in doubt, you run this one on the same images to get the
answer you compare everything else against.

What makes it different from the other geometry boxes:

| | `sfm` | `d4rt` / `tapnext` | **`colmap`** |
|---|---|---|---|
| input | depth + tracks + intrinsics | images + depth (streaming) | **plain images only** |
| intrinsics | given, or estimated from tracks | given | **estimated (self-calibrating)** or fixed |
| model | partial, depth-cued | rolling, windowed | **one self-consistent global batch** |
| runtime | seconds | ~200 ms/frame | **tens of seconds to minutes** |
| role | fast reconstruction | live reconstruction | **ground truth / verification** |

It runs the real COLMAP - the incremental mapper (strictest ground truth)
and the GLOMAP global mapper (merged into COLMAP 4.x, 1-2 orders of
magnitude faster, `mapper=global`).

## Command surface

One RPC, one envelope, commands selected via `config_json`:

| command | meaning |
|---|---|
| `reconstruct` (default) | run the full pipeline and return poses + points + intrinsics |
| `log` | live tail of COLMAP's output + job state (see "watching a long job") |
| `reset` | **cancel the running reconstruction** (SIGTERM, then SIGKILL) |

Example (gRPC request `config_json`, `data` omitted):

```json
{ "command": "reconstruct",
  "parameters": { "mapper": "incremental", "matcher": "exhaustive" } }
```

```json
{ "command": "log", "parameters": { "tail": 50 } }
```

- **mapper** `incremental | global | hierarchical` - `incremental` is the
  strictest ground truth; `global` is GLOMAP (fast, excellent quality);
  `hierarchical` targets very large sets.
- **matcher** `auto | exhaustive | sequential` - `auto` is exhaustive, or
  sequential when you send `image_names` (their order = video order).
- **max_image_size / max_num_features** - the quality/speed levers
  (3200 px / 8192 features by default).
- **use_gpu** `auto | true | false` - GPU SIFT extraction and matching when a
  device is present; the box always runs the same on CPU.
- **intrinsics** (optional) - fixed camera as `"PINHOLE 640 480 500 500 320 240"`;
  empty = COLMAP self-calibrates (the usual ground-truth setup).
- **max_runtime_s** - a budget; on expiry the box cancels COLMAP (which
  flushes partial results) and reports `partial: true`.

## The data envelope

- `data.images` **required** - JPEG/PNG bytes of the view set
  (`bb`), or a single blob.
- `data.image_names` optional - ordered names; order defines video order for
  the sequential matcher.
- Output, all `numpy`:
  - `cameras` (F,3,4) - `[R|t]` world→camera per registered image
  - `points` (P,3), `point_colors` (P,3 uint8)
  - `intrinsics` (F,3,3) - pinhole view of each COLMAP camera model
  - `frame_ids` (F,) - positions in `images` for each row
  - `image_names` (F,) - registered image names
- The answer object also carries `mapper`, `gpu`, `num_registered`,
  `num_points`, `reproj_error` (mean px, from `model_analyzer`),
  `stage_times_s` and `partial` flags.

## Watching a long job (the reason this box ships these limits)

A ground-truth reconstruction is minutes on real sets, and the gRPC reply
arrives only at the end. This is a *design property of a ground-truth
box*, not a defect - but you shouldn't have to sit in the dark:

1. **`reconstruct` is the single long call.** The gRPC reply comes at the
   end; use a client deadline of `max_runtime_s` + margin (or leave
   `max_runtime_s = 0` and rely on your deadline).
2. **Poll `command=log` on a second connection.** It returns the job state
   (`running`, `stage`, `elapsed_s`, `lines`) plus the last `tail` lines of
   COLMAP's own output (feature extraction, matching, incremental steps,
   reproj error). The box keeps the whole log in a bounded deque.
3. **`docker logs`** works too: every line COLMAP emits is echoed to the
   service stdout.
4. **`command=reset`** kills the job; unlike the no-op resets in most
   VisionIST boxes, this one has teeth.

Because `log`/`reset` must answer while the reconstruction occupies a
worker, `limits.max_workers` is set to **2** (one for the job, one to serve
the lightweight calls) - see `box.yaml`.

One at a time: a second concurrent `reconstruct` returns an in-band error,
not a queue.

## Limits (from box.yaml)

* `max_message_mb: 512` - image payloads are the bottleneck; raise only with `limits.max_message_mb` (or split the set).
* `max_workers: 2` - job worker + log/reset worker.

## Security

`reconstruct` runs **untrusted image bytes through a C++ binary** (COLMAP)
with no user code execution. Image contents are untrusted *data*; treat
outputs as data as well. `log`/`reset` are safe.

## Build

```
docker build -t sipgisr/visionist-colmap:0.1.0 -f docker/Dockerfile .
```

The base image is the official pre-compiled, CUDA-enabled **COLMAP**
(digest-pinned); we only add the Python service on top.

## Run

```
docker run --rm -p 8061:8061 --gpus all \
  -e "PORT=8061" -e "MAX_MESSAGE_MB=512" -e "MAX_WORKERS=2" \
  sipgisr/visionist-colmap:0.1.0
```

No `--restart unless-stopped` here: the box is meant to be driven on
demand, and a long job is normal - it just takes the time it takes.

## Test

Committed fixtures: 7 DSLR frames of the ETH3D Large-Scale 2 *door* scene
(see `test/data/provenance.json`), each ~200 KB - under the 256 KB cap. The
dataset ships its own ground-truth reconstruction (one shared PINHOLE camera,
all 7 images registered), so this test checks the box **against ground
truth**, not just self-consistency:

```
python test/test_colmap.py
```

It rebuilds the scene two ways (incremental and GLOMAP/global) and, beyond
the envelope contract (reset / empty / unknown) and the live `log` command,
verifies against GT quantities that are world-frame-free and scale-free:
7/7 frames registered, > 500 points, focal within 10 % of GT, and the
box-reported mean reprojection error < 1.5 px.

One documented caveat: the door cameras sit on a near-planar arc, where
scale is the weakly-constrained degree of freedom of any SfM run. The
incremental mapper nails the baselines; GLOMAP (a different objective)
lands ~2x off there while still matching rotations to 0.1 degree, focal to
a single pixel, and sub-pixel reprojection - the test asserts that,
and notes the baseline gap rather than failing on it. On non-degenerate
3D scenes GLOMAP's baselines are just as good.

## Registry

Run the tooling after any manifest change:

```
python3 tools/validate_boxes.py colmap
python3 tools/sync_contract.py
python3 tools/build_index.py
```

## References

- COLMAP: Schönberger et al., *Structure-from-Motion Revisited*, CVPR 2016.
- GLOMAP (now the `global` mapper): Pan, Barath, Pollefeys, Schönberger,
  *Global Structure-from-Motion Revisited*, ECCV 2024.
- License: BSD-3-Clause upstream.
