#!/usr/bin/env python3
"""Smoke test for the mediapipe box: start it, drive it, check the contract.

    python3 test/test_mediapipe.py                   # starts its own server
    python3 test/test_mediapipe.py --host host:port  # against a running box

Two layers, on purpose:

* the **packing** checks need neither mediapipe nor a model bundle - they feed
  stand-in landmark objects to the pure functions that turn MediaPipe results
  into padded arrays, which is where the off-by-one bugs live (padding, counts,
  pixel scaling, the handedness code);
* the **inference** checks need the .task bundles. They run when the bundles
  are present (`python3 tools/fetch_assets.py mediapipe`, or set
  MEDIAPIPE_MODEL_DIR) and are reported as skipped when they are not, so this
  file is still worth running in a checkout that has no weights.

No image fixtures: the frames are generated with numpy, so nothing large is
committed and the test runs anywhere.
"""

import argparse
import io
import json
import pathlib
import sys
import time
import types

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "protos"))
sys.path.insert(0, str(ROOT / "src"))

import grpc                                                     # noqa: E402
import pipeline_pb2                                             # noqa: E402
import pipeline_pb2_grpc                                        # noqa: E402
from aux import wrap_value, unwrap_value                        # noqa: E402

import cv2                                                      # noqa: E402

import mediapipe_service as svc                                 # noqa: E402

PASS, FAIL, SKIP = [], [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"{'ok  ' if condition else 'FAIL'} {name:46s} {detail}")


def skip(name, why):
    SKIP.append(name)
    print(f"skip {name:46s} {why}")


# ------------------------------------------------------------------ helpers
def serve():
    """A server in this process, so the test needs no docker."""
    import concurrent.futures as futures
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4),
                         options=[("grpc.max_send_message_length", -1),
                                  ("grpc.max_receive_message_length", -1)])
    pipeline_pb2_grpc.add_PipelineServiceServicer_to_server(
        svc.PipelineService(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, f"127.0.0.1:{port}"


def call(stub, section, data=None):
    env = pipeline_pb2.Envelope(config_json=json.dumps({"mediapipe": section}))
    for k, v in (data or {}).items():
        env.data[k].CopyFrom(wrap_value(v))
    reply = stub.Process(env, timeout=180)
    cfg = json.loads(reply.config_json)["mediapipe"]
    fields = {k: unwrap_value(v) for k, v in reply.data.items()}
    return cfg, fields


def decode(fields, cfg, name):
    """Apply the codec the box declared, as a client would."""
    codec = (cfg.get("encoding") or {}).get(name)
    raw = fields[name]
    if codec == "numpy":
        return np.load(io.BytesIO(raw), allow_pickle=False)
    if codec == "json":
        return json.loads(raw.decode())
    return raw


def jpeg(width=320, height=240, seed=0):
    """A synthetic frame. Nothing in it is a hand, a body or a face - these
    tests check the envelope and the array shapes, not MediaPipe's accuracy."""
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), 180, dtype=np.uint8)
    cv2.circle(img, (width // 2, height // 3), height // 6, (120, 140, 200), -1)
    cv2.rectangle(img, (width // 3, height // 2), (2 * width // 3, height - 10),
                  (90, 110, 160), -1)
    img = np.clip(img.astype(np.int16)
                  + rng.integers(-8, 8, img.shape, dtype=np.int16), 0, 255).astype(np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def landmark(x, y, z, visibility=0.0):
    return types.SimpleNamespace(x=x, y=y, z=z, visibility=visibility)


def category(name, score):
    return types.SimpleNamespace(category_name=name, score=score, index=0,
                                 display_name=None)


def models_available():
    missing = [c for c in ("hands", "pose", "face")
               if not svc.model_path(c).is_file()]
    return not missing, missing


# ------------------------------------------------------------------ packing
def test_packing():
    print("\n== packing (no model bundles needed) ==")

    # Two images: the first has two detections, the second has none. The
    # padded array must be (2, 2, 3, 3) and counts must say [2, 0].
    a = [landmark(0.1, 0.2, 0.3), landmark(0.4, 0.5, 0.6), landmark(0.7, 0.8, 0.9)]
    b = [landmark(0.0, 0.0, 0.0)] * 3
    arr, counts = svc.pack_landmarks([[a, b], []], 3, [(1.0, 1.0, 1.0)] * 2)
    check("pack: padded shape", arr.shape == (2, 2, 3, 3), f"{arr.shape}")
    check("pack: counts", counts.tolist() == [2, 0], f"{counts.tolist()}")
    check("pack: dtype float32", arr.dtype == np.float32, f"{arr.dtype}")
    check("pack: first landmark kept", np.allclose(arr[0, 0, 0], [0.1, 0.2, 0.3]))
    check("pack: padding is zero", np.all(arr[1] == 0))

    # Nothing detected anywhere: M collapses to 0 rather than inventing a row.
    empty, counts0 = svc.pack_landmarks([[], []], 21, [(1.0, 1.0, 1.0)] * 2)
    check("pack: empty batch -> M=0", empty.shape == (2, 0, 21, 3), f"{empty.shape}")
    check("pack: empty counts", counts0.tolist() == [0, 0])

    # pixels: x and z scale with the width, y with the height.
    px, _ = svc.pack_landmarks([[a]], 3, [(320.0, 240.0, 320.0)])
    check("pack: pixel scaling",
          np.allclose(px[0, 0, 0], [0.1 * 320, 0.2 * 240, 0.3 * 320]),
          f"{px[0, 0, 0]}")

    # More landmarks than the model declares: the extras are dropped, never
    # written past the end of the row.
    long_row = [landmark(0.5, 0.5, 0.5)] * 40
    clipped, _ = svc.pack_landmarks([[long_row]], 33, [(1.0, 1.0, 1.0)])
    check("pack: clipped to L", clipped.shape == (1, 1, 33, 3), f"{clipped.shape}")

    codes, scores = svc._handedness_codes(
        [[[category("Left", 0.98)], [category("Right", 0.81)]], []], 2)
    check("pack: handedness codes", codes.tolist() == [[0, 1], [-1, -1]],
          f"{codes.tolist()}")
    check("pack: handedness scores", np.allclose(scores[0], [0.98, 0.81]))

    vis = svc._visibility([[[landmark(0, 0, 0, 0.9), landmark(0, 0, 0, 0.1)]], []], 1, 2)
    check("pack: visibility", np.allclose(vis[0, 0], [0.9, 0.1]) and np.all(vis[1] == 0),
          f"{vis[0, 0]}")

    blend = svc.pack_vectors([[[0.1] * 52], []], 1, 52)
    check("pack: blendshapes", blend.shape == (2, 1, 52) and np.allclose(blend[0, 0, 0], 0.1),
          f"{blend.shape}")

    mats = svc.pack_matrices([[np.eye(4)], []], 1, 4, 4)
    check("pack: matrices", mats.shape == (2, 1, 4, 4) and np.allclose(mats[0, 0], np.eye(4)),
          f"{mats.shape}")


def test_pack_real_results():
    """The same packing, fed MediaPipe's own result objects rather than
    stand-ins. Needs mediapipe installed but no model bundle, and is the check
    that catches an upstream field rename (hand_world_landmarks,
    facial_transformation_matrixes and friends) without a 300 MB download."""
    print("\n== packing MediaPipe result objects (no bundles needed) ==")
    try:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.components.containers import (
            category as mp_category, landmark as mp_landmark)
    except ImportError as e:
        skip("packing MediaPipe result objects", f"mediapipe not installed ({e})")
        return

    def normalized(n):
        return [mp_landmark.NormalizedLandmark(x=0.001 * i, y=0.002 * i, z=0.003 * i,
                                               visibility=0.5) for i in range(n)]

    def world(n):
        return [mp_landmark.Landmark(x=0.01 * i, y=0.02 * i, z=0.03 * i) for i in range(n)]

    cases = (
        ("hands",
         vision.HandLandmarkerResult(
             handedness=[[mp_category.Category(index=0, score=0.97,
                                               category_name="Left", display_name="Left")]],
             hand_landmarks=[normalized(21)], hand_world_landmarks=[world(21)]),
         {"landmarks", "counts", "world_landmarks", "handedness", "scores"}, 21),
        ("pose",
         vision.PoseLandmarkerResult(pose_landmarks=[normalized(33)],
                                     pose_world_landmarks=[world(33)],
                                     segmentation_masks=None),
         {"landmarks", "counts", "world_landmarks", "visibility"}, 33),
        ("face",
         vision.FaceLandmarkerResult(
             face_landmarks=[normalized(478)],
             face_blendshapes=[[mp_category.Category(index=i, score=0.01 * i,
                                                     category_name=f"bs{i}",
                                                     display_name=None)
                                for i in range(52)]],
             facial_transformation_matrixes=[np.eye(4)]),
         {"landmarks", "counts", "blendshapes", "blendshape_names",
          "transformation_matrixes"}, 478),
    )
    for command, result, expected, n_points in cases:
        spec = {"blendshapes": command == "face", "transformation_matrix": command == "face"}
        out, enc = svc.PipelineService._pack(command, spec, [result], [(1.0, 1.0, 1.0)])
        check(f"{command}: result object -> declared fields", set(out) == expected,
              f"{sorted(out)}")
        check(f"{command}: landmark count", out["landmarks"].shape == (1, 1, n_points, 3),
              f"{out['landmarks'].shape}")
        check(f"{command}: every field has a codec", set(enc) == set(out), f"{enc}")
    check("face: blendshape names come back in order",
          svc.PipelineService._pack("face", {"blendshapes": True,
                                             "transformation_matrix": False},
                                    [cases[2][1]], [(1.0, 1.0, 1.0)])[0]
          ["blendshape_names"][:2] == ["bs0", "bs1"])


def test_load_error():
    """Which of the two load failures the message blames. The libEGL one is
    not hypothetical: the 0.1.0 image was built without libegl1/libgles2 and
    every command failed this way, under a message that blamed the bundle."""
    print("\n== load failures are told apart ==")
    path = pathlib.Path("/workspace/models/pose_landmarker_heavy.task")

    lib = str(svc.load_error(path, OSError(
        "libEGL.so.1: cannot open shared object file: No such file or directory")))
    check("missing library: names the package", "libegl1" in lib, lib[:80])
    check("missing library: does not blame the bundle",
          "truncated" not in lib and "re-fetch" not in lib, lib[:80])

    bundle = str(svc.load_error(path, RuntimeError("Unable to open zip archive.")))
    check("bad bundle: names the file",
          "pose_landmarker_heavy.task" in bundle and "truncated" in bundle, bundle[:80])
    check("bad bundle: says how to re-fetch", "fetch_assets" in bundle, bundle[:80])


def test_parameters():
    print("\n== parameter parsing ==")
    spec, coords = svc.parse_parameters("hands", {})
    check("params: hand default is 2 hands", spec["max_results"] == 2)
    check("params: default normalized", coords == "normalized")
    spec, _ = svc.parse_parameters("pose", {})
    check("params: pose default is 1 pose, lite",
          spec["max_results"] == 1 and spec["model"] == "lite")
    spec, coords = svc.parse_parameters("face", {"blendshapes": "true",
                                                 "coordinates": "pixels"})
    check("params: flags coerce from strings",
          spec["blendshapes"] is True and coords == "pixels")
    for bad, why in (({"model": "enormous"}, "unknown pose model"),
                     ({"max_results": 0}, "max_results below 1"),
                     ({"min_detection_confidence": 2}, "confidence out of range"),
                     ({"coordinates": "metres"}, "unknown coordinates")):
        try:
            svc.parse_parameters("pose", bad)
            check(f"params: rejects {why}", False, "accepted")
        except ValueError:
            check(f"params: rejects {why}", True)


# ------------------------------------------------------------------ envelope
def test_envelope(stub):
    print("\n== envelope contract ==")

    cfg, _ = call(stub, {"command": "reset"})
    check("reset -> done", cfg.get("status") == "done" and cfg.get("action") == "reset",
          f"{cfg}")

    cfg, _ = call(stub, {"command": "hands"})
    check("no images -> empty_request", cfg.get("status") == "empty_request", f"{cfg}")

    cfg, _ = call(stub, {"command": "levitate"}, {"images": [jpeg()]})
    check("unknown command -> error", cfg.get("status") == "error", f"{cfg}")
    check("unknown command names the options",
          "hands" in (cfg.get("error") or ""), f"{cfg.get('error')}")

    cfg, _ = call(stub, {"command": "pose", "parameters": {"model": "enormous"}},
                  {"images": [jpeg()]})
    check("bad parameter -> error", cfg.get("status") == "error", f"{cfg.get('error')}")

    cfg, _ = call(stub, {"command": "hands", "parameters": []}, {"images": [jpeg()]})
    check("parameters not an object -> error", cfg.get("status") == "error",
          f"{cfg.get('error')}")

    cfg, _ = call(stub, {"command": "hands"}, {"images": [b"not an image"]})
    check("undecodable image -> error", cfg.get("status") == "error",
          f"{cfg.get('error')}")

    # Section missing entirely.
    reply = stub.Process(pipeline_pb2.Envelope(config_json=json.dumps({"yolo": {}})),
                         timeout=30)
    cfg = json.loads(reply.config_json)["mediapipe"]
    check("foreign config section -> error", cfg.get("status") == "error",
          f"{cfg.get('error')}")


def test_missing_models(stub):
    """With no bundles present the box must say which file it wants and how to
    get it - not raise something opaque from inside MediaPipe."""
    print("\n== missing model bundles ==")
    cfg, _ = call(stub, {"command": "hands"}, {"images": [jpeg()]})
    msg = cfg.get("error") or ""
    check("missing bundle -> error", cfg.get("status") == "error", f"{msg[:70]}")
    check("missing bundle names the file", "hand_landmarker.task" in msg, msg[:70])
    check("missing bundle says how to fix it", "fetch_assets" in msg, msg[:70])


# ----------------------------------------------------------------- inference
def test_inference(stub):
    print("\n== inference (model bundles present) ==")
    images = [jpeg(seed=1), jpeg(width=200, height=400, seed=2)]

    # ---- hands
    cfg, fields = call(stub, {"command": "hands"}, {"images": images})
    check("hands -> done", cfg.get("status") == "done", f"{cfg.get('error', '')}")
    if cfg.get("status") == "done":
        enc = cfg.get("encoding") or {}
        counts = decode(fields, cfg, "counts")
        lms = decode(fields, cfg, "landmarks")
        world = decode(fields, cfg, "world_landmarks")
        hand = decode(fields, cfg, "handedness")
        check("hands: fields present",
              {"landmarks", "counts", "world_landmarks", "handedness",
               "scores"} <= set(fields), f"{sorted(fields)}")
        check("hands: every field declares a codec",
              all(enc.get(f) in ("numpy", "json") for f in fields), f"{enc}")
        check("hands: 21 landmarks", lms.shape[2] == 21 if lms.shape[1] else True,
              f"{lms.shape}")
        check("hands: one row of counts per image", counts.shape == (2,), f"{counts.shape}")
        check("hands: counts match the padded axis",
              lms.shape[1] == int(counts.max()), f"{lms.shape} {counts.tolist()}")
        check("hands: world landmarks align", world.shape == lms.shape,
              f"{world.shape} vs {lms.shape}")
        check("hands: handedness per detection", hand.shape == lms.shape[:2],
              f"{hand.shape}")
        check("hands: section reports the geometry",
              cfg.get("landmarks_per_detection") == 21
              and cfg.get("num_images") == 2
              and cfg.get("coordinates") == "normalized", f"{cfg}")

    # ---- pose, in pixels
    cfg, fields = call(stub, {"command": "pose",
                              "parameters": {"model": "lite", "coordinates": "pixels"}},
                       {"images": images})
    check("pose -> done", cfg.get("status") == "done", f"{cfg.get('error', '')}")
    if cfg.get("status") == "done":
        lms = decode(fields, cfg, "landmarks")
        vis = decode(fields, cfg, "visibility")
        check("pose: 33 landmarks", lms.shape[2] == 33 if lms.shape[1] else True,
              f"{lms.shape}")
        check("pose: visibility aligns", vis.shape == lms.shape[:3],
              f"{vis.shape} vs {lms.shape}")
        check("pose: reports pixels", cfg.get("coordinates") == "pixels", f"{cfg}")

    # ---- face, with the extras
    cfg, fields = call(stub, {"command": "face",
                              "parameters": {"blendshapes": True,
                                             "transformation_matrix": True}},
                       {"images": images[:1]})
    check("face -> done", cfg.get("status") == "done", f"{cfg.get('error', '')}")
    if cfg.get("status") == "done":
        lms = decode(fields, cfg, "landmarks")
        check("face: 478 landmarks", lms.shape[2] == 478 if lms.shape[1] else True,
              f"{lms.shape}")
        check("face: extras requested, extras present",
              {"blendshapes", "blendshape_names", "transformation_matrixes"} <= set(fields),
              f"{sorted(fields)}")
        check("face: blendshape names are json",
              (cfg.get("encoding") or {}).get("blendshape_names") == "json")
        mats = decode(fields, cfg, "transformation_matrixes")
        check("face: 4x4 matrices", mats.shape[-2:] == (4, 4), f"{mats.shape}")

    # ---- face without the extras: the fields must be absent, not empty
    cfg, fields = call(stub, {"command": "face"}, {"images": images[:1]})
    check("face: extras off -> fields absent",
          "blendshapes" not in fields and "transformation_matrixes" not in fields,
          f"{sorted(fields)}")

    # ---- a second call with different options rebuilds the landmarker
    t0 = time.time()
    cfg, _ = call(stub, {"command": "hands", "parameters": {"max_results": 4}},
                  {"images": images[:1]})
    check("options change -> still done", cfg.get("status") == "done",
          f"{time.time() - t0:.2f}s")
    cfg, _ = call(stub, {"command": "reset"})
    check("reset after use", cfg.get("status") == "done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None)
    args = ap.parse_args()

    server = None
    if args.host:
        addr = args.host
    else:
        server, addr = serve()
        time.sleep(0.3)
    print(f"mediapipe at {addr}")
    have_models, missing = models_available()
    print(f"model bundles: {'present in ' + str(svc.model_dir()) if have_models else 'MISSING ' + str(missing)}\n")

    channel = grpc.insecure_channel(addr, options=[
        ("grpc.max_send_message_length", -1),
        ("grpc.max_receive_message_length", -1)])
    stub = pipeline_pb2_grpc.PipelineServiceStub(channel)

    test_packing()
    test_pack_real_results()
    test_load_error()
    test_parameters()
    test_envelope(stub)
    if have_models:
        test_inference(stub)
    else:
        test_missing_models(stub)
        skip("inference (hands / pose / face)",
             "no .task bundles — python3 tools/fetch_assets.py mediapipe")

    if server:
        server.stop(0)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
    if FAIL:
        print("failed: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
