# YOLO Box (detection, pose, segmentation, oriented boxes, classification — with tracking)

A gRPC box that runs ultralytics YOLO over **images and videos** and returns
per-frame records plus optionally annotated frames.

**Five tasks, chosen with `command`:**

| command | what you get back | tracked |
|---|---|---|
| `detect` (default) | boxes, labels, scores | yes |
| `pose` | the above **plus 17 COCO keypoints per person** | yes |
| `segment` | the above **plus one polygon per instance** | yes |
| `obb` | oriented boxes: centre, size, angle, four corners | yes |
| `classify` | one top-5 verdict for the whole frame | no — ultralytics refuses to track it |

The command **is** the task, and a checkpoint knows its own: `pose` needs a
`-pose` checkpoint, `segment` a `-seg` one. Ask for a task the active
checkpoint cannot do and the box says which checkpoint to use, rather than
running happily and returning none of the fields you wanted. Each task has
its own default checkpoint (`yolo26n-<task>.pt`), so `command: "pose"` works
with no `weights` at all; `detect` keeps its historical default so nothing
changes underneath existing callers.

For `detect`, `pose`, `segment` and `obb` the box is **always in tracking
mode, so every detection also carries a stable per-session track id**.

The box speaks the shared **envelope** interface, so it is addressable
through `visionist_client` exactly like clip:

```python
service PipelineService {
  rpc Process( Envelope ) returns ( Envelope );
}
```

It follows the **tapnext multi-session contract**: the tracker state (object
identities + id counter) lives per `session_id`, so one call runs detection
+ tracking over the frames you send, and *the same session* keeps stable
track ids across calls while *different sessions* get independent id
sequences. Every **fresh tracker** (new session id — regenerate, page
reload, or right after `reset`) starts its numbering from the base, so
"start over" always means `track_id` 1, 2, … — track ids are per-session
namespaces and new sessions never collide with old ones. `reset` clears
**this session's** tracker (other sessions are untouched), `list` gives the
operator the active sessions, and idle sessions are reaped after
`YOLO_SESSION_TTL` seconds (default 1800; `0` keeps them forever).
Sessions live on the **box**, not in the image bytes — a box
restart starts all of them fresh.

## Directory structure

```
yolo/
├── docker/
│   └── Dockerfile
├── protos/
│   ├── pipeline.proto         # shared proto (same as every other box)
│   ├── pipeline_pb2.py        # generated
│   ├── pipeline_pb2_grpc.py   # generated
│   └── aux.py                 # wrap_value / unwrap_value helpers
├── src/
│   └── yolo_service.py        # PipelineService.Process(Envelope)
├── test/
│   ├── test_yolo.py           # in-process / remote smoke test (dummy images)
│   └── car.jpg                # test fixture
├── requirements.txt
└── README.md
```

## Build

```bash
cd boxes/yolo
docker build --tag sipgisr/visionist-yolo --build-arg SERVICE_NAME=yolo -f docker/Dockerfile .
```

**No checkpoint is baked into the image.** The default weights download at
**box startup** (into the container workspace, ~6 MB for `yolov8n.pt`), the
same pattern as clip's startup ViT download. Pick a different default with
`-e YOLO_WEIGHTS=…` at run time, or any checkpoint per call with
`parameters.weights` (a fresh one downloads into the workspace on first use).

## Run

```bash
docker run --rm --gpus all -p 8061:8061 -e PORT=8061 sipgisr/visionist-yolo
```

## Service usage

### Request

Up to two fields in `data` (send **either**, not both), and a small `config`:

- `data["images"]` — list of image bytes (JPEG/PNG/…), one detection record
  per image.
- `data["video"]` — a single video file (mp4/avi/webm/mov). The box decodes
  it server-side (OpenCV/ffmpeg), samples every `frame_step`-th frame, and
  stops at `max_frames` sampled frames.
- `config["yolo"]["command"]` — the task: `"detect"` (default), `"pose"`,
  `"segment"`, `"obb"` or `"classify"`; plus `"reset"` (clear this session's
  tracker state) and `"list"` (active sessions — no `data` needed).
  Tracking is always on for the four detection-shaped tasks.
- `config["yolo"]["session_id"]` — opaque session key (section top level,
  like tapnext). Omitted → the shared `"default"` session. Same session =
  stable track ids across calls; different sessions = independent sequences.
- `config["yolo"]["parameters"]` — the knobs:

| key               | default | meaning                                                        |
|-------------------|---------|----------------------------------------------------------------|
| `weights`         | env `YOLO_WEIGHTS` (`yolov8n.pt`) | any ultralytics checkpoint (name, local path, or URL); fresh ones download into the workspace on first use; echoed back as `weights` in the response status |
| `conf`            | `0.25`  | confidence threshold                                            |
| `iou`             | `0.70`  | NMS IoU threshold                                               |
| `imgsz`           | `640`   | inference image size                                            |
| `max_det`         | `300`   | max detections per frame                                        |
| `classes`         | all     | list of class ids to keep (e.g. `[16]` = dogs only)             |
| `device`          | auto    | `"cpu"` / `"cuda"` / `"cuda:N"` — wins over the auto lifecycle |
| `save_annotated`  | `true`  | also return the annotated frames as JPEGs                       |
| `frame_step`      | `1`     | video only: sample every Nth frame                              |
| `max_frames`      | `1024`  | safety cap on sampled frames per call                           |
| `return_masks`    | `false` | `segment` only: also return full binary masks (see below)       |
| `nms`             | auto    | `false` selects YOLO26's NMS-free one-to-one head (end-to-end)  |

### Response

`config_json` carries the status:

```json
{
  "yolo": {
    "status": "done",
    "session": "default",          // or the session_id you sent
    "weights": "yolov8n.pt",
    "runtime": 4.21,
    "source": "images",            // or "video"
    "num_frames": 2,
    "frames_sampled": 2,
    "num_detections": 3,
    "tracked": true,               // every call tracks; ids are per-session
    "num_tracks": 1,               // unique track ids in this call
    "frames_in_video": 211,        // video only: total (container metadata; fallback: decoded)
    "frame_step": 30,              // video only
    "encoding": { "detections": "json", "annotated": "identity" }
  }
}
```

`reset` / `list` respond in the same shape as tapnext:

```json
{ "yolo": { "status": "done", "action": "reset", "session": "ses_abc" } }
{ "yolo": { "status": "done", "action": "list",
            "sessions": [ { "session": "default", "frames_processed": 2,
                            "has_tracker": true, "idle_seconds": 42.0 } ] } }
```

and `data` carries:

| field        | kind | description |
|--------------|------|-------------|
| `detections` | `b` (declared `json`) | list with **one record per frame**: see below |
| `annotated`  | `bb` (declared `identity`) | JPEG-annotated frame per input frame (only when `save_annotated`) |
| `annotated_video` | `b` (declared `identity`) | **video input only**: the same annotated frames re-encoded as one MP4 (source fps) — a video in, a video out. **H.264** (`libx264`/yuv420p via PyAV) so the browser's `<video>` plays it; falls back to `mp4v` (VLC-playable, not HTML5) when PyAV is missing. The status echoes `annotated_video_codec: h264 \| mp4v`; absent when no writer is available (the JPEG list still carries the frames) |

A detection record (plain JSON):

```json
{
  "frame_index": 30,
  "width": 1920,
  "height": 1080,
  "boxes":       [[x1, y1, x2, y2], ...],   // xyxy in input-frame pixels
  "class_ids":   [49, ...],
  "labels":      ["orange", ...],
  "scores":      [0.65, ...],
  "track_id":    [3, ...]                   // per-box tracker id in THIS session (null when the frame has no boxes)
}
```

`frame_index` is the frame's original index in the source (its position in
`images`, or the video frame number when sampling a video). `boxes` are
axis-aligned `xyxy` rectangles in the **original frame's** pixel
coordinates (rounded to 2 decimals). `track_id` is aligned with `boxes`
(same order/length): the same object keeps the same id within a session
across frames *and* across calls — that's what makes a video's detections
joinable into trajectories. A fresh session (or after `reset`) starts
numbering from the base.

## What each task adds to a record

Every record has the same shape; a task leaves the keys it has nothing to say
about at `null`. `detect` produces exactly what it always did.

**`pose`** → `keypoints`, `(num_people, 17, 3)` as `[x, y, confidence]` in
frame pixels, and `keypoint_names` — the COCO-17 order, reported only when
the model really has 17 joints, so a custom skeleton is never mislabelled.
The track id of person *i* is `track_id[i]`, so skeletons follow people
across frames for free.

**`segment`** → `polygons`, one contour per instance, in **frame pixels**.

This is deliberately not ultralytics' raw `masks.data`. That array comes back
at the *model's* input resolution — 640×480 for an 810×1080 frame — so
sending it would hand you coordinates that do not line up with the boxes
beside it. It is also about **900× larger**: for five people in one frame,
1,536,000 mask values against 1,653 polygon points. Measured on a real frame,
a whole segmentation reply is ~57 KB.

If you do want dense masks, `return_masks: true` adds a `masks` field: one
`(num_instances, H, W)` `uint8` stack per frame, **rasterised from the
polygons at frame resolution**, so it lines up with everything else. Declared
with the `numpy` codec. Mind your `MAX_MESSAGE_MB` — this is the one option
here that can approach it.

**`obb`** → `obb` with `xywhr` (centre, size, rotation in radians) and
`corners`, the four `(x, y)` points. Note the shipped checkpoint is trained
on DOTA aerial imagery and finds nothing in ordinary ground-level photos.

**`classify`** → `classification` with `top1` and `top5`, each
`{class_id, label, score}`. No per-object entries and no `track_id`:
ultralytics raises outright for `mode=track` on a classifier.

```python
# tracked skeletons
res = b.run(data={"video": pathlib.Path("clip.mp4")},
            config={"yolo": {"command": "pose", "session_id": "alice",
                             "parameters": {"frame_step": 5}}})
rec = res.detections[0]
kp  = rec["keypoints"]          # [person][joint][x, y, conf]
who = rec["track_id"]           # same order as keypoints
nose_of_first = kp[0][rec["keypoint_names"].index("nose")]
```

## Call with visionist_client

```python
from visionist_client import Visionist
import pathlib

b = Visionist("localhost:8061")

# --- images ---------------------------------------------------------
res = b.run(
    data   = {"images": [pathlib.Path("car.jpg")]},
    config = {"yolo": {"command": "detect",
                       "parameters": {"conf": 0.25}}},
)
print(res.encoding)     # {'detections': 'json', 'annotated': 'identity'}
for d in res.detections:                       # decoded: a Python list
    for box, label, score in zip(d["boxes"], d["labels"], d["scores"]):
        print(label, score, box)
open("frame0.jpg", "wb").write(res.annotated[0])   # optional annotated JPEGs

# --- video ----------------------------------------------------------
res = b.run(
    data   = {"video": pathlib.Path("pan.mp4")},
    config = {"yolo": {"command": "detect",
                       "parameters": {"frame_step": 30, "max_frames": 16,
                                      "save_annotated": False}}},
)
print(res.config["yolo"])   # frames_sampled / frames_in_video / num_detections

# --- switch the checkpoint for one call (downloads on first use) ----
res = b.run(
    data   = {"images": [pathlib.Path("car.jpg")]},
    config = {"yolo": {"command": "detect",
                      "parameters": {"weights": "yolov11n.pt"}}},
)
print(res.config["yolo"]["weights"])   # 'yolov11n.pt'
```

Or use the vendored protos directly:

```python
import grpc, json, pipeline_pb2, pipeline_pb2_grpc, aux

channel = grpc.insecure_channel("localhost:8061",
                                options=[("grpc.max_send_message_length", -1),
                                         ("grpc.max_receive_message_length", -1)])
stub = pipeline_pb2_grpc.PipelineServiceStub(channel)

req = pipeline_pb2.Envelope(
    config_json=json.dumps({"yolo": {"command": "detect"}}),
    data={"images": aux.wrap_value([open("car.jpg", "rb").read()])},
)
resp = stub.Process(req)
dets = json.loads(aux.unwrap_value(resp.data["detections"]))  # bytes -> JSON
print(json.loads(resp.config_json), dets)
```

## GPU behaviour

The fleet convention: the model loads on **CPU at startup** (fast start, no
VRAM), moves to CUDA **in place** on the first request (unless
`parameters.device` says otherwise), and a watchdog thread moves it back to
CPU after ~60 s idle. `parameters.device` always wins.

Inference is **chunked**: N frames run in `⌈N / batch⌉` forward passes of
≤ `batch` frames (default 16 — ultralytics' own video default). Without that,
ultralytics' list path would stack every frame into one batch, and the VRAM
peak would scale with the full frame count (≈1 GB for 128 frames @640, ≈10 GB
for 1024 @640, plus host RAM holding all decoded frames).

Raising `batch` uses more VRAM per pass (and fewer passes); lowering it
further bounds the spike at the cost of a little latency.

`annotated_video` (a video in, one annotated MP4 out) is encoded with H.264
via x264 `preset=veryfast, tune=zerolatency, crf=23` — fast enough that a
1080p/24-frame preview encodes in well under ~0.5 s on a modern CPU, and
slightly smaller than x264's default `medium` preset.  For long videos
(>~200 frames) consider `save_annotated: false` in the call to skip the
per-frame JPEG encode and the MP4 re-encode entirely — the `detections`
JSON is still returned.

## Test

A clip-style script (no notebook) that builds Envelopes with the bundled
test images, calls `Process`, and checks the decoded detection records, the
annotated JPEGs, the `reset` no-op, and — when a video is available — the
video input path:

```bash
# server already running at :8061
python test/test_yolo.py
BOX_HOST=10.0.0.5:8061 python test/test_yolo.py
YOLO_TEST_VIDEO=my_clip.mp4 python test/test_yolo.py
```

The video case is **skipped** (not failed) when no video file is available:
it checks `YOLO_TEST_VIDEO`, then `boxes/tapnext/test/pan.mp4` (a short synthetic clip from `tapnext/test/make_test_video.py`).
Likewise, a `parameters.weights` switch is exercised only when
`YOLO_TEST_WEIGHTS` names a checkpoint the test may fetch (e.g.
`YOLO_TEST_WEIGHTS=yolov8s.pt`).
