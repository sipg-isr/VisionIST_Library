"""In-process contract test for the abot_recon box, with the model stubbed out.

The real model is ~1.0 B parameters behind a 4.0 GB checkpoint, so this does
not test the geometry - it tests everything around it, which is where the
envelope bugs live: command dispatch, input decoding, the images/video
exclusion, the frame bookkeeping (the lexicographic file naming upstream
depends on), the output-flag logic, the declared encoding, the namespaced
status vocabulary, and that every output is an np.save blob of the
documented shape.

    cd boxes/abot_recon && python3 test/smoke_inprocess.py

Needs numpy and opencv-python(-headless). abot_recon is faked, so it runs on
a laptop with no GPU and no weights. For the real thing, build the box and
use test/test_abot_recon.py.
"""
import io
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, "src")
sys.path.insert(0, "protos")

import numpy as np
import cv2

# The box ships only pipeline.proto; the generated stubs are produced inside
# the Docker build, so generate them into a temp dir for a local run.
try:
    import pipeline_pb2  # noqa: F401
except ImportError:
    import subprocess
    _gen = tempfile.mkdtemp(prefix="abot_recon_protos_")
    try:
        subprocess.run([sys.executable, "-m", "grpc_tools.protoc", "-Iprotos",
                        f"--python_out={_gen}", f"--grpc_python_out={_gen}",
                        "protos/pipeline.proto"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.exit("could not generate the proto stubs: pip install grpcio-tools")
    sys.path.insert(0, _gen)

# ---------------------------------------------------------------- fake torch
# torch_device_default() probes cuda availability; the abot_recon import is
# faked anyway so neither needs the real thing.
_torch = types.ModuleType("torch")
_torch.cuda = types.SimpleNamespace(is_available=lambda: False,
                                    empty_cache=lambda: None)
sys.modules.setdefault("torch", _torch)

# ---------------------------------------------------------------- fake abot
CALLS = {"infer": []}
_HW = (280, 504)                                  # the model's fixed grid


class _T:
    """Duck-types a torch.Tensor just enough for _to_np."""

    def __init__(self, arr):
        self.a = np.asanyarray(arr)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Result:
    def __init__(self, t, loop, local_on, world_on, conf_on, dense, threshold):
        self.camera_poses = _T(np.tile(np.eye(4, dtype=np.float32), (t, 1, 1)) + 1.0)
        self.relative_poses = _T(np.tile(np.eye(4, dtype=np.float32), (t - 1, 1, 1)))
        self.camera_poses_noloop = _T(np.tile(np.eye(4, dtype=np.float32), (t, 1, 1)))
        self.relative_poses_noloop = _T(np.tile(np.eye(4, dtype=np.float32), (t - 1, 1, 1)))
        self.camera_poses_loop = self.camera_poses if loop else None
        self.relative_poses_loop = self.relative_poses if loop else None
        td = t if dense is None else len(dense)
        self.local_points = _T(np.zeros((td, *_HW, 3), np.float32)) if local_on else None
        self.world_points = _T(np.zeros((td, *_HW, 3), np.float32)) if world_on else None
        self.confidence = _T(np.full((td, *_HW), 0.9, np.float32)) if conf_on else None
        self.confidence_mask = (_T(np.full((td, *_HW), conf_on and threshold <= 0.9,
                                          bool)) if conf_on else None)
        self.metadata = {"num_frames": t, "loop_closure": bool(loop),
                         "attention_backend": "sdpa",
                         "dense_output_indices": dense or list(range(t)),
                         "something": PathStub("weird")}


class PathStub:
    """A non-JSON object in metadata: the service must stringify, not crash."""

    def __init__(self, v):
        self.v = v

    def __str__(self):
        return f"<{self.v}>"


class _FakeABot:
    def __init__(self, name, **overrides):
        self.name, self.overrides = name, overrides
        self.model = types.SimpleNamespace(eval=lambda: None,
                                           to=lambda d: None,
                                           parameters=lambda: [np.zeros(8)])

    def infer(self, image_paths, **kw):
        CALLS["infer"].append({"paths": list(image_paths), **kw})
        t = len(image_paths)
        dense = kw.get("dense_output_indices")
        return _Result(t, kw["loop_closure"], kw["output_local_points"],
                       kw["output_world_points"], kw["output_confidence"],
                       dense, kw["confidence_threshold"])


_abot = types.ModuleType("abot_recon")
_abot.ABotRecon = _FakeABot
sys.modules["abot_recon"] = _abot

import abot_recon_service as svc                                    # noqa: E402
import pipeline_pb2                                                 # noqa: E402
from aux import wrap_value, unwrap_value                             # noqa: E402

T_FRAMES = 6
svc._MODEL.get = lambda device: _FakeABot("/weights/abot_recon.safetensors")
# Loop assets: pretend they are present so loop_closure=true can be exercised.
svc._loop_assets = lambda: (os.path.join(svc._LOOP_DIR, "dino_salad.ckpt"),
                            os.path.join(svc._LOOP_DIR, "dinov2_vitb14_pretrain.pth"))

# ------------------------------------------------------------------ fixtures
def frame(seed=0, h=120, w=160):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


IMAGES = [frame(i) for i in range(T_FRAMES)]

srv = svc.PipelineService()
failures = []


def call(config=None, data=None):
    req = pipeline_pb2.Envelope(
        config_json=json.dumps(config) if config else "",
        data=data or {})
    return srv.Process(req, None)


def section(resp):
    return json.loads(resp.config_json or "{}").get("abot_recon", {})


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def arr(resp, field):
    return np.load(io.BytesIO(bytes(unwrap_value(resp.data[field]))), allow_pickle=False)


print("== 1. reconstruct, the default ==")
r = call({"abot_recon": {"command": "reconstruct"}},
         {"images": wrap_value(IMAGES)})
s = section(r)
print("   ", {k: v for k, v in s.items() if k not in ("encoding", "metadata")})
check("status done", s.get("status") == "done", str(s.get("error", "")))
check("encoding all numpy",
      all(v == "numpy" for v in s.get("encoding", {}).values()),
      str(s.get("encoding"))[:80])
check("camera_poses is (T, 4, 4)", arr(r, "camera_poses").shape == (T_FRAMES, 4, 4),
      str(arr(r, "camera_poses").shape))
check("relative_poses is (T-1, 4, 4)",
      arr(r, "relative_poses").shape == (T_FRAMES - 1, 4, 4),
      str(arr(r, "relative_poses").shape))
check("local_points is (Td, 280, 504, 3)",
      arr(r, "local_points").shape == (T_FRAMES, 280, 504, 3),
      str(arr(r, "local_points").shape)[:30])
check("world_points present", "world_points" in r.data)
check("confidence is (Td, 280, 504)",
      arr(r, "confidence").shape == (T_FRAMES, 280, 504), str(arr(r, "confidence").shape))
check("confidence_mask is boolean", arr(r, "confidence_mask").dtype == bool)
check("no loop variants by default", "camera_poses_loop" not in r.data)
check("num_frames reported", s.get("num_frames") == T_FRAMES)
check("metadata carried and JSON-safe",
      isinstance(s.get("metadata"), dict) and s["metadata"].get("num_frames") == T_FRAMES,
      str(s.get("metadata"))[:80])
last = CALLS["infer"][-1]
check("infer got 6 paths in order", len(last["paths"]) == T_FRAMES
      and [os.path.basename(p) for p in last["paths"]]
      == [f"frame_{i:06d}.jpg" for i in range(T_FRAMES)],
      str([os.path.basename(p) for p in last["paths"]])[:80])
check("loop off at the model", last["loop_closure"] is False)
check("no dense restriction", last.get("dense_output_indices") is None)

print("== 2. trajectory: poses only ==")
r = call({"abot_recon": {"command": "trajectory"}}, {"images": wrap_value(IMAGES)})
s = section(r)
check("status done", s.get("status") == "done", str(s.get("error", "")))
check("poses still (T, 4, 4)", arr(r, "camera_poses").shape == (T_FRAMES, 4, 4))
check("no point maps leaked in",
      all(f not in r.data for f in ("local_points", "world_points",
                                    "confidence", "confidence_mask")),
      f"fields: {sorted(r.data)}")
check("the model asked for no points",
      CALLS["infer"][-1]["output_local_points"] is False
      and CALLS["infer"][-1]["output_world_points"] is False
      and CALLS["infer"][-1]["output_confidence"] is False)

print("== 3. output flags and thresholds reach the model ==")
r = call({"abot_recon": {"command": "reconstruct",
                         "parameters": {"output_world_points": False,
                                        "confidence_threshold": 0.5}}},
         {"images": wrap_value(IMAGES)})
s = section(r)
check("status done", s.get("status") == "done", str(s.get("error", "")))
check("world_points suppressed", "world_points" not in r.data,
      f"fields: {sorted(r.data)}")
check("threshold reached the model",
      CALLS["infer"][-1]["confidence_threshold"] == 0.5,
      str(CALLS["infer"][-1]["confidence_threshold"]))
check("threshold echoed", s.get("confidence_threshold") == 0.5)
check("mask reflects the threshold", bool(arr(r, "confidence_mask").all()),
      f"any: {bool(arr(r, 'confidence_mask').any())}")
r = call({"abot_recon": {"command": "reconstruct",
                         "parameters": {"confidence_threshold": 1.5}}},
         {"images": wrap_value(IMAGES)})
check("threshold out of [0, 1] is an error",
      section(r).get("status") == "error", str(section(r).get("error", ""))[:60])

print("== 4. dense_stride thins the maps, not the trajectory ==")
r = call({"abot_recon": {"command": "reconstruct", "parameters": {"dense_stride": 2}}},
         {"images": wrap_value(IMAGES)})
s = section(r)
check("status done", s.get("status") == "done", str(s.get("error", "")))
check("poses stay per-frame", arr(r, "camera_poses").shape == (T_FRAMES, 4, 4),
      str(arr(r, "camera_poses").shape))
check("local_points thinned to Td", arr(r, "local_points").shape[0] == 3,
      str(arr(r, "local_points").shape)[:30])
check("num_dense_frames reported", s.get("num_dense_frames") == 3,
      str(s.get("num_dense_frames")))
r = call({"abot_recon": {"command": "reconstruct", "parameters": {"dense_stride": 0}}},
         {"images": wrap_value(IMAGES)})
check("dense_stride 0 is an error", section(r).get("status") == "error",
      str(section(r).get("error", ""))[:50])

print("== 5. loop closure (assets present in this smoke) ==")
r = call({"abot_recon": {"command": "reconstruct", "parameters": {"loop_closure": True}}},
         {"images": wrap_value(IMAGES)})
s = section(r)
check("status done", s.get("status") == "done", str(s.get("error", "")))
check("loop variants present",
      "camera_poses_loop" in r.data and "relative_poses_loop" in r.data,
      f"fields: {sorted(r.data)}")
check("loop reached the model", CALLS["infer"][-1]["loop_closure"] is True)
check("loop_closure_used echoed", s.get("loop_closure_used") is True)

print("== 6. video input ==")
with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as fh:
    path = fh.name
vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
for i in range(12):
    vw.write(np.random.default_rng(i).integers(0, 255, (120, 160, 3), dtype=np.uint8))
vw.release()
blob = open(path, "rb").read()
os.unlink(path)
r = call({"abot_recon": {"command": "trajectory",
                         "parameters": {"max_frames": 5}}},
         {"video": wrap_value(blob)})
s = section(r)
check("video decodes", s.get("status") == "done", str(s.get("error", ""))[:70])
check("max_frames respected", s.get("num_frames") == 5, str(s.get("num_frames")))

print("== 7. input contract ==")
r = call({"abot_recon": {"command": "reconstruct"}},
         {"images": wrap_value(IMAGES), "video": wrap_value(blob)})
check("images + video is an error", section(r).get("status") == "error",
      str(section(r).get("error", ""))[:50])
r = call({"abot_recon": {"command": "reconstruct"}})
check("no input is empty_request", section(r).get("status") == "empty_request",
      str(section(r)))
r = call({"abot_recon": {"command": "reconstruct"}}, {"images": wrap_value(IMAGES[:2])})
check("fewer than 3 frames is an error", section(r).get("status") == "error",
      str(section(r).get("error", ""))[:50])
mixed = [IMAGES[0], frame(9, h=64, w=64)]
r = call({"abot_recon": {"command": "reconstruct"}}, {"images": wrap_value(mixed)})
check("mismatched frame sizes are an error", section(r).get("status") == "error",
      str(section(r).get("error", ""))[:50])
r = call({"abot_recon": {"command": "reconstruct"}},
         {"images": wrap_value([b"not an image"] * 3)})
check("undecodable frame is an error", section(r).get("status") == "error",
      str(section(r).get("error", ""))[:50])
r = call({"abot_recon": {"command": "trajectory",
                         "parameters": {"max_frames": 2}}},
         {"images": wrap_value(IMAGES)})
check("max_frames below the minimum is an error",
      section(r).get("status") == "error", str(section(r).get("error", ""))[:50])
r = call({"abot_recon": {"command": "trajectory",
                         "parameters": {"max_frames": 10 ** 6}}},
         {"images": wrap_value(IMAGES)})
check("max_frames above the model cap is an error",
      section(r).get("status") == "error", str(section(r).get("error", ""))[:60])
r = call({"abot_recon": {"command": "trajectory",
                         "parameters": {"frame_step": 0}}},
         {"images": wrap_value(IMAGES)})
check("frame_step 0 is an error", section(r).get("status") == "error",
      str(section(r).get("error", ""))[:50])

print("== 8. config contract ==")
check("wrong namespace", section(call({"d4rt": {"command": "infer"}},
                                      {"images": wrap_value(IMAGES)})).get("status") == "error")
check("no config", section(call(None, {"images": wrap_value(IMAGES)})).get("status") == "error")
r = call({"abot_recon": {"command": "explode"}}, {"images": wrap_value(IMAGES)})
check("unknown command", section(r).get("status") == "error"
      and "explode" in section(r).get("error", ""))
r = call({"abot_recon": {"command": "reconstruct",
                         "parameters": {"output_world_points": "maybe"}}},
         {"images": wrap_value(IMAGES)})
check("bad boolean parameter", section(r).get("status") == "error",
      str(section(r).get("error", ""))[:50])
r = call({"abot_recon": {"command": "reconstruct",
                         "parameters": {"dense_stride": "lots"}}},
         {"images": wrap_value(IMAGES)})
check("bad numeric parameter", section(r).get("status") == "error",
      str(section(r).get("error", ""))[:50])
r = call({"abot_recon": {"command": "reconstruct", "parameters": {"device": "tpu"}}},
         {"images": wrap_value(IMAGES)})
check("bad device is an error", section(r).get("status") == "error",
      str(section(r).get("error", ""))[:50])

print("== 9. reset ==")
s = section(call({"abot_recon": {"command": "reset"}}))
check("reset is a no-op done", s.get("status") == "done" and s.get("action") == "reset",
      str(s))

print()
print("FAIL -- " + "; ".join(failures) if failures
      else "PASS -- all in-process contract cases ok")
sys.exit(1 if failures else 0)
