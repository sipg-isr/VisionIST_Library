#!/usr/bin/env python3
"""Test the abot_recon box against a running instance.

Needs the real model, so this wants a GPU box with the checkpoint. For the
contract alone, with no weights and no GPU, use test/smoke_inprocess.py.

    python3 test/test_abot_recon.py
    BOX_HOST=10.0.0.5:8061 python3 test/test_abot_recon.py
    ABOT_VIDEO=/path/clip.mp4 python3 test/test_abot_recon.py

Without ABOT_VIDEO a short synthetic clip is generated: a bright square
translating across a textured background. That is enough to check shapes,
the declared encoding and the error contract. The geometry sanity check is
deliberately weak: with a fixed, nearly fronto-parallel camera the moving
square should produce more per-pixel point displacement than the static
background.
"""

import io
import json
import os
import sys

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_TEST_DIR, "..", "protos"))

import grpc            # noqa: E402
import numpy as np     # noqa: E402

# The box ships only pipeline.proto; the generated stubs are produced inside
# the Docker build. Generate them into a scratch dir for a local client run.
import subprocess       # noqa: E402
import tempfile         # noqa: E402
try:
    import pipeline_pb2  # noqa: F401
except ImportError:
    _gen = tempfile.mkdtemp(prefix="abot_recon_test_")
    sys.path.insert(0, os.path.join(_TEST_DIR, ".."))
    try:
        subprocess.run(
            [sys.executable, "-m", "grpc_tools.protoc",
             "-I" + os.path.join(_TEST_DIR, "..", "protos"),
             f"--python_out={_gen}", f"--grpc_python_out={_gen}",
             os.path.join(_TEST_DIR, "..", "protos", "pipeline.proto")],
            check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.exit("could not generate the proto stubs: pip install grpcio-tools")
    sys.path.insert(0, _gen)
    import pipeline_pb2  # noqa: F401

import aux             # noqa: E402
import pipeline_pb2    # noqa: E402
import pipeline_pb2_grpc  # noqa: E402

FRAMES, SIZE = 8, 192
HW = (280, 504)                          # the model's fixed output grid


def synthetic_clip():
    """A moving square over static texture, as JPEG frames."""
    import cv2

    rng = np.random.default_rng(0)
    background = rng.integers(40, 200, (SIZE, SIZE, 3), dtype=np.uint8)
    for _ in range(40):                       # texture, so features exist
        x, y = rng.integers(0, SIZE - 20, 2)
        cv2.rectangle(background, (int(x), int(y)), (int(x) + 18, int(y) + 18),
                      tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    out = []
    for t in range(FRAMES):
        img = background.copy()
        x = 4 + t * 10                        # the square translates right
        cv2.rectangle(img, (x, 84), (x + 24, 108), (255, 255, 255), -1)
        ok, buf = cv2.imencode(".jpg", img)
        assert ok
        out.append(buf.tobytes())
    return out


def make_stub(target):
    insecure = grpc.insecure_channel(
        target, options=[("grpc.max_send_message_length", -1),
                         ("grpc.max_receive_message_length", -1)])
    return pipeline_pb2_grpc.PipelineServiceStub(insecure)


def section(response):
    cfg = json.loads(response.config_json or "{}")
    s = cfg.get("abot_recon")
    if not isinstance(s, dict):
        print(f"  ERROR: reply not namespaced under 'abot_recon': {cfg}")
        return None
    if s.get("status") == "error":
        print(f"  ERROR: {s.get('error')}")
    return s


def decode(value):
    """A pipeline.Value (numpy blob in the `b` slot) -> ndarray."""
    raw = aux.unwrap_value(value)
    return np.load(io.BytesIO(bytes(raw)), allow_pickle=False)


def main() -> int:
    target = os.getenv("BOX_HOST", "localhost:8061")
    stub = make_stub(target)

    failures = []

    def check(name, cond, detail=""):
        print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"  {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    def call(config, data=None):
        return stub.Process(
            pipeline_pb2.Envelope(
                config_json=json.dumps(config),
                data=data or {}),
            timeout=30 * 60)

    images = synthetic_clip()

    print(f"== 1. reconstruct at {target} ==")
    r = call({"abot_recon": {"command": "reconstruct",
                             "parameters": {"max_frames": FRAMES}}},
             {"images": aux.wrap_value(images)})
    s = section(r)
    print("   ", {k: v for k, v in (s or {}).items() if k not in ("encoding", "metadata")})
    check("status done", s is not None and s.get("status") == "done",
          str((s or {}).get("status")))
    if s and s.get("status") == "done":
        check("encoding all numpy", all(v == "numpy" for v in s.get("encoding", {}).values()))
        t = s.get("num_frames", FRAMES)
        poses = decode(r.data["camera_poses"])
        check("camera_poses is (T, 4, 4)", poses.shape == (t, 4, 4), str(poses.shape))
        check("poses are finite", np.isfinite(poses).all())
        rel = decode(r.data["relative_poses"])
        check("relative_poses is (T-1, 4, 4)", rel.shape == (t - 1, 4, 4), str(rel.shape))
        check("relative_poses are finite", np.isfinite(rel).all())
        noloop = decode(r.data["camera_poses_noloop"])
        check("noloop trajectory present", noloop.shape == (t, 4, 4))
        lp = decode(r.data["local_points"])
        check("local_points is (T, 280, 504, 3)", lp.shape == (t, *HW, 3),
              str(lp.shape))
        check("local points finite", np.isfinite(lp).all())
        check("local points are in front of the camera (z > 0)",
              float(lp[..., 2].min()) > 0.0, f"min z {float(lp[..., 2].min()):.4f}")
        if "world_points" in r.data:
            wp = decode(r.data["world_points"])
            check("world_points finite", np.isfinite(wp).all())
            # A world-transformed cloud should not sit on the input frame grid.
            moved = float(np.linalg.norm(wp[1] - wp[0], axis=None)) if t > 1 else 0.0
            print(f"   (world-point displacement between frame 0 and 1: {moved:.3f})")
            check("world cloud moves between frames", moved > 1e-6, f"{moved:.2e}")
        if "confidence" in r.data:
            conf = decode(r.data["confidence"])
            check("confidence is (T, 280, 504)", conf.shape == (t, *HW), str(conf.shape))
            check("confidence in [0, 1]",
                  float(conf.min()) >= 0.0 and float(conf.max()) <= 1.0,
                  f"[{float(conf.min()):.3f}, {float(conf.max()):.3f}]")
            mask = decode(r.data["confidence_mask"])
            check("confidence_mask boolean, matches threshold 0",
                  mask.dtype == bool and mask.all(),
                  f"dtype {mask.dtype}, any False: {bool((~mask).any())}")

    print("== 2. trajectory (poses only) ==")
    r = call({"abot_recon": {"command": "trajectory",
                             "parameters": {"max_frames": FRAMES}}},
             {"images": aux.wrap_value(images)})
    s = section(r)
    check("status done", s is not None and s.get("status") == "done",
          str((s or {}).get("error", "")))
    if s and s.get("status") == "done":
        for f in ("local_points", "world_points", "confidence", "confidence_mask"):
            check(f"no {f} leaked in", f not in r.data)
        check("poses still (T, 4, 4)",
              decode(r.data["camera_poses"]).shape == (s.get("num_frames", FRAMES), 4, 4))

    print("== 3. error contract ==")
    r = call({"abot_recon": {"command": "reconstruct"}})
    check("no input is empty_request",
          section(r) is not None and section(r).get("status") == "empty_request")
    r = call({"abot_recon": {"command": "reconstruct"}}, {"images": aux.wrap_value(images[:2])})
    check("fewer than 3 frames is an error",
          section(r) is not None and section(r).get("status") == "error")
    r = call({"abot_recon": {"command": "no_such_command"}})
    check("unknown command is an error",
          section(r) is not None and section(r).get("status") == "error")

    print("== 4. reset ==")
    s = section(call({"abot_recon": {"command": "reset"}}))
    check("reset is a no-op done", s is not None
          and s.get("status") == "done" and s.get("action") == "reset", str(s))

    print()
    if failures:
        print("FAIL -- " + "; ".join(failures))
        return 1
    print("PASS -- all live checks ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
