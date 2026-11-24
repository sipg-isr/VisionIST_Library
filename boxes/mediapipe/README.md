# mediapipe

Google [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) landmark
tasks behind the shared envelope: **21-point hands**, **33-point body pose** and the
**478-point face mesh**. CPU only, no network at request time, ~0.8 GB image.

This file is the authoritative request shape for the box. Port 9075 is pycv; this
box is **9076** in a generated fleet, 8061 inside its own container.

## Commands

| command | what comes back | model bundle |
|---|---|---|
| `hands` *(default)* | `landmarks` (21 per hand), `world_landmarks`, `handedness`, `scores`, `counts` | `hand_landmarker.task` |
| `pose` | `landmarks` (33 per person), `world_landmarks`, `visibility`, `counts` | `pose_landmarker_{lite,full,heavy}.task` |
| `face` | `landmarks` (478 per face), `counts`, optionally `blendshapes` + `blendshape_names` + `transformation_matrixes` | `face_landmarker.task` |
| `reset` | closes the cached landmarkers | — |

## Request

```json
{"mediapipe": {"command": "hands", "parameters": {"max_results": 2}}}
```

with `data.images` a `bb` list of JPEG/PNG bytes. One image or many; every image
in a call is run through the same landmarker.

### Parameters

| key | default | commands | meaning |
|---|---|---|---|
| `max_results` | 2 hands / 1 pose / 1 face | all | upstream `num_hands` / `num_poses` / `num_faces` |
| `model` | `lite` | `pose` | `lite`, `full` or `heavy`; heavy is ~3× slower and the most accurate |
| `coordinates` | `normalized` | all | `pixels` converts to image units (see below) |
| `min_detection_confidence` | 0.5 | all | |
| `min_presence_confidence` | 0.5 | all | |
| `min_tracking_confidence` | 0.5 | all | only read in VIDEO mode, which this box does not expose yet |
| `blendshapes` | false | `face` | also return the 52 expression scores |
| `transformation_matrix` | false | `face` | also return the 4×4 canonical-face matrix |

Changing any of these rebuilds the landmarker for that command — the limits and
thresholds are baked into the MediaPipe graph, not passed per call. Rebuilding
costs tens of milliseconds, so a client that alternates between two settings pays
for it on every request.

## Response

`{"mediapipe": {"status": "done", "command": …, "num_detections": …, "encoding": {…}}}`
plus the fields above. Every array is an `np.save` blob declared `numpy`;
`blendshape_names` is `json`.

### The two things callers get wrong

**1. `landmarks` is padded — read `counts` first.**

```
landmarks  (n_images, M, L, 3)    M = the largest detection count in the BATCH
counts     (n_images,)   int32    how many of those M rows are real, per image
```

An image where nothing was found contributes a row of zeros, and a batch where
nothing was found anywhere gives `M = 0`, i.e. shape `(n, 0, L, 3)`. Zeros are a
valid landmark position, so there is no sentinel to test for: `counts[i]` is the
only answer.

**2. Coordinates are normalized `(x, y, z)`, not pixels, and not `(y, x)`.**

MediaPipe puts `x` and `y` in `[0, 1]` across the width and height, and `z` on
roughly the same scale as `x` — depth relative to the subject (wrist for hands,
hips for pose, head centre for face), negative towards the camera. This is the
opposite order from `tapnext`, which reports `(y, x)`, and a different unit from
`opencv` / `features`, which report pixels. Send `coordinates: "pixels"` to get
`x·width, y·height, z·width` and line up with those boxes.

`world_landmarks` (hands and pose only) is the metric one: metres, origin at the
hand centre or the hip midpoint, independent of where the subject is in frame.
The face bundle has no world landmarks — use `transformation_matrixes` for pose
in space.

### Handedness

`handedness` is `int8`: `0` = left, `1` = right, `-1` = padding, with the model's
confidence in `scores`. MediaPipe labels the hand **as seen in the image**, so a
webcam frame that was not un-mirrored reports the opposite hand. Flip the frame
before sending if your capture path mirrors.

## Models

The bundles are not in the pip wheel. They are fetched once at **image build
time** by `docker/fetch_models.py` and copied into `/workspace/models`
(`MEDIAPIPE_MODEL_DIR` overrides the path) — the running box never reaches the
network. The same URLs are declared under `assets:` in `box.yaml`, so a checkout
gets them with:

```bash
python3 tools/fetch_assets.py mediapipe      # -> boxes/mediapipe/models/
```

The URLs end in `/float16/latest/`, which is what the MediaPipe documentation
publishes and is a **moving alias**. For a build whose weights are reproducible
byte for byte, pin the upstream version instead:

```bash
docker build -f docker/Dockerfile --build-arg MODEL_VERSION=1 -t sipgisr/visionist-mediapipe:latest .
```

`sha256` is deliberately absent from the `assets:` entries while the URL is
`latest`: a checksum against a moving target would fail the first time Google
republishes. Pin `MODEL_VERSION` and add the hashes `docker/fetch_models.py`
prints if your deployment needs them.

## Build notes

### System libraries

Four of them, all installed by the Dockerfile:

| needed at load time | Debian package | why |
|---|---|---|
| `libGL.so.1`, `libglib-2.0.so.0` | `libgl1`, `libglib2.0-0` | mediapipe depends on `opencv-contrib-python`, not the headless build |
| `libEGL.so.1`, `libGLESv2.so.2` | `libegl1`, `libgles2` | `mediapipe/tasks/c/libmediapipe.so` has DT_NEEDED entries for both |

**The EGL pair is not optional on a CPU-only box.** They are linked into
MediaPipe's native library, so the dynamic loader wants them before any task
is created, whatever the delegate. The 0.1.0 image shipped without them and
every command failed at `create_from_options` with `libEGL.so.1: cannot open
shared object file` — a message MediaPipe gives with no mention of a library,
which is why `load_error()` in the service now tells that case apart from a
bad bundle. Fixed in 0.1.1.

To confirm it inside a running 0.1.0 container, before rebuilding:

```bash
docker exec -u root visionist-mediapipe \
  sh -c 'apt-get update && apt-get install -y libegl1 libgles2'
docker restart visionist-mediapipe
```

`MPLCONFIGDIR=/workspace/.mplcache` is set for the same family of reason: the
`runner` user has no home directory, so matplotlib (another thing
opencv-contrib drags in) warns at every boot and rebuilds its font cache in a
fresh temp directory.

### arm64

mediapipe 1.1.0 publishes `manylinux_2_28` wheels for **x86_64 and aarch64**,
so the box builds natively on Apple Silicon — `docker/Dockerfile.mac` is the
same recipe with the arm64 base digest. `box.yaml` still declares
`platforms: [linux/amd64]`: that is what CI publishes, and a multi-arch push
would be a separate decision.

## Tests

```bash
python3 test/test_mediapipe.py            # starts its own server, no docker
python3 test/test_mediapipe.py --host 10.0.0.5:9076
```

47 checks, no committed fixtures (frames are generated with numpy). They split in
three: the packing functions are driven with stand-in landmarks **and** with real
MediaPipe result objects — neither needs a model bundle, and the second catches an
upstream field rename; the envelope checks cover `reset`, `empty_request` and
every error path; the inference checks run only when the bundles are present and
are reported as skipped when they are not.

## A worked notebook

`VisionIST/notebooks/mediapipe_walkthrough.ipynb` runs every command against a
live box with `visionist_client`: the three landmark sets drawn over photos, the
metric world landmarks in 3D, per-joint visibility, the blendshape scores, the
padding/`counts` contract and the error paths. It inlines MediaPipe's own
skeleton edge lists, so it needs no `mediapipe` install on the client side.

## Not here yet

* **VIDEO / LIVE_STREAM running mode.** Both need monotonic timestamps and keep
  tracking state between frames, which is the fleet's `session` convention
  (`cumulative: true`, as in tapnext). The box is IMAGE-mode and stateless today.
* **The other MediaPipe tasks** — holistic, segmentation, gesture recognition,
  object detection, embeddings. Holistic is the obvious next one; gesture
  recognition is a thin layer over the hand bundle. Detection and embedding
  overlap with `yolo` and `clip`, which are better at both.
* **GPU.** MediaPipe's GPU delegate targets Android, iOS and the web. On a Linux
  server this box is CPU, which is why `runtime: cpu` and why there is no
  `device` parameter.
