"""Ad-hoc in-process smoke test for the modernized opencv service (no box build
needed): instantiate the servicer and drive Process() with envelopes."""
import io
import json
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "protos")

import numpy as np
import cv2
import opencv_service as svc
import pipeline_pb2
from aux import wrap_value, unwrap_value


def make_image(seed=0, size=(320, 240)):
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 220, size=(*size[::-1], 3), dtype=np.uint8)
    # some structure: blocks + lines so SIFT has corners
    for _ in range(12):
        x, y = rng.integers(0, 160, 2)
        w, h = rng.integers(10, 60, 2)
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)),
                      tuple(int(v) for v in rng.integers(0, 255, 3)), 2)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def call(srv, config=None, images=None, method="Process"):
    req = pipeline_pb2.Envelope(
        config_json=json.dumps(config) if config else "",
        data={"images": wrap_value(images)} if images is not None else {})
    return getattr(srv, method)(req, None)


A = make_image(1)
B = make_image(2)

srv = svc.PipelineService()
failures = []

def section(resp):
    return (json.loads(resp.config_json or "{}")).get("opencv", {})

def check(name, cond, detail=""):
    print(f"  {'ok ' if cond else 'FAIL'} {name} {detail}")
    if not cond:
        failures.append(name)

print("== 1. match SIFT, 2 images ==")
r = call(srv, {"opencv": {"command": "match", "parameters": {}}}, [A, B])
s = section(r)
print("   ", s)
check("status done", s.get("status") == "done")
for f in ("keypoints", "descriptors", "matches_inliers_a", "matches_inliers_b",
          "fundamental_matrix"):
    check(f"field {f}", f in r.data)
    if f in r.data:
        arr = np.load(io.BytesIO(bytes(unwrap_value(r.data[f]))))
        print(f"      {f}: {arr.shape} {arr.dtype}")
        check(f"{f} is array", arr.ndim > 0)
check("num_inliers consistent", s.get("num_inliers") == 0, str(s.get("num_inliers")))

check("encoding declared", s.get("encoding", {}).get("keypoints") == "numpy")
check("matcher", s.get("matcher") == "FLANN", s.get("matcher"))
check("num_images", s.get("num_images") == 2)

print("== 2. match SIFT, 1 image (extraction only) ==")
r = call(srv, {"opencv": {"command": "match", "parameters": {"max_keypoints": 200}}}, [A])
s = section(r)
check("status done", s.get("status") == "done", str(s.get("status")))
check("no matching fields", "matches_inliers_a" not in r.data and "fundamental_matrix" not in r.data)
check("keypoints in", "keypoints" in r.data)
k = np.load(io.BytesIO(bytes(unwrap_value(r.data["keypoints"]))))
print("      keypoints:", k.shape)

print("== 2b. legacy call without command ==")
r = call(srv, {"opencv": {"parameters": {"feature_extractor": "ORB"}}}, [A, B])
check("legacy no-command ok", section(r).get("status") == "done")

print("== 3. missing images -> empty_request ==")
r = call(srv, {"opencv": {"command": "match"}}, None)
check("empty_request", section(r).get("status") == "empty_request", str(section(r)))

print("== 4. bad config -> error ==")
r = call(srv, {"clip": {"command": "encode"}}, [A])
s = section(r)
check("error status", s.get("status") == "error")
check("error message", bool(s.get("error")), s.get("error"))
r = call(srv, None, [A])
check("no config error", section(r).get("status") == "error")
r = call(srv, {"opencv": {"command": "explode"}}, [A])
check("unknown command error", section(r).get("status") == "error"
      and "explode" in section(r).get("error", ""))
r = call(srv, {"opencv": {"command": "match", "parameters": {"max_keypoints": "lots"}}}, [A])
check("bad param error", section(r).get("status") == "error",
      str(section(r).get("error")))

print("== 5. lightglue path (not installed here) -> graceful error ==")
r = call(srv, {"opencv": {"command": "match",
                          "parameters": {"feature_extractor": "SUPERPOINT",
                                        "max_keypoints": 64}}}, [A, B])
s = section(r)
print("   ", s)
check("graceful error", s.get("status") == "error" and "LightGlue" in s.get("error", ""))

print("== 6. reset ==")
r = call(srv, {"opencv": {"command": "reset"}})
s = section(r)
check("reset done", s.get("status") == "done" and s.get("action") == "reset", str(s))

print("== 7. bad frame -> error ==")
r = call(srv, {"opencv": {"command": "match"}}, [b"not an image"])
check("error on undecodable", section(r).get("status") == "error",
      str(section(r).get("error")))

print("== 8. image with no keypoints (blank wall, lens cap) is not an error ==")
ok, buf = cv2.imencode(".jpg", np.full((240, 320, 3), 128, np.uint8))
BLANK = buf.tobytes()
for label, imgs in (("two images, one blank", [A, BLANK]), ("two blank images", [BLANK, BLANK]),
                    ("one blank image", [BLANK])):
    r = call(srv, {"opencv": {"command": "match", "parameters": {}}}, imgs)
    s = section(r)
    check(f"{label}: status done", s.get("status") == "done", str(s.get("error", s.get("status"))))
    if len(imgs) == 2 and s.get("status") == "done":
        check(f"{label}: no inliers, empty F", s.get("num_inliers") == 0
              and np.load(io.BytesIO(bytes(unwrap_value(r.data["fundamental_matrix"])))).size == 0)

print()
print("FAIL -- " + "; ".join(failures) if failures else "PASS -- all in-process cases ok")
sys.exit(1 if failures else 0)
