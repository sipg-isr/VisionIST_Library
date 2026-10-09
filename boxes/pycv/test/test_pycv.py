#!/usr/bin/env python3
"""Smoke test for the pycv box: start it, drive it, check every tier.

    python3 test/test_pycv.py                 # starts its own server
    python3 test/test_pycv.py --host host:port  # against a running box

No fixtures: the images are generated with numpy, so nothing large is
committed and the test runs anywhere.
"""

import argparse
import io
import json
import pathlib
import sys
import threading
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "protos"))
sys.path.insert(0, str(ROOT / "src"))

import grpc                                                     # noqa: E402
import pipeline_pb2                                             # noqa: E402
import pipeline_pb2_grpc                                        # noqa: E402
from aux import wrap_value, unwrap_value                        # noqa: E402

import cv2                                                      # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"{'ok  ' if condition else 'FAIL'} {name:44s} {detail}")


def serve():
    """A server in this process, so the test needs no docker."""
    import concurrent.futures as futures
    from pycv_service import PipelineService
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4),
                         options=[("grpc.max_send_message_length", -1),
                                  ("grpc.max_receive_message_length", -1)])
    pipeline_pb2_grpc.add_PipelineServiceServicer_to_server(PipelineService(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, f"127.0.0.1:{port}"


def call(stub, section, data=None):
    env = pipeline_pb2.Envelope(config_json=json.dumps({"pycv": section}))
    for k, v in (data or {}).items():
        env.data[k].CopyFrom(wrap_value(v))
    reply = stub.Process(env, timeout=120)
    cfg = json.loads(reply.config_json)["pycv"]
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
    print(f"pycv at {addr}\n")

    channel = grpc.insecure_channel(addr, options=[
        ("grpc.max_send_message_length", -1),
        ("grpc.max_receive_message_length", -1)])
    stub = pipeline_pb2_grpc.PipelineServiceStub(channel)

    img = np.zeros((60, 80, 3), np.uint8)
    cv2.rectangle(img, (10, 10), (40, 40), (255, 255, 255), -1)
    png = cv2.imencode(".png", img)[1].tobytes()

    # --- reset -----------------------------------------------------------
    cfg, _ = call(stub, {"command": "reset"})
    check("reset is accepted", cfg["status"] == "done", cfg.get("action", ""))

    # --- eval, one expression --------------------------------------------
    cfg, f = call(stub, {"command": "eval"},
                  {"code": "cv2.cvtColor(vs.image, cv2.COLOR_BGR2GRAY)",
                   "images": [png]})
    gray = decode(f, cfg, "result") if "result" in f else None
    check("eval returns an array", cfg["status"] == "done" and gray is not None
          and gray.shape == (60, 80) and gray.dtype == np.uint8,
          f"{cfg['status']} {None if gray is None else (gray.shape, gray.dtype)}")

    # --- eval, a tuple is unpacked ---------------------------------------
    cfg, f = call(stub, {"command": "eval"},
                  {"code": "cv2.threshold(cv2.cvtColor(vs.image, cv2.COLOR_BGR2GRAY), 128, 255, cv2.THRESH_BINARY)",
                   "images": [png]})
    check("eval unpacks a cv2 tuple",
          cfg["status"] == "done" and {"result_0", "result_1"} <= set(cfg["emitted"]),
          str(cfg.get("emitted")))

    # --- eval, statements rather than an expression ----------------------
    cfg, f = call(stub, {"command": "eval"},
                  {"code": "n = int(vs.image.mean())\nvs.emit('mean', n)",
                   "images": [png]})
    check("eval accepts statements", cfg["status"] == "done"
          and decode(f, cfg, "mean") == int(img.mean()),
          str(cfg.get("emitted")))

    # --- run, multi-file with a helper module ----------------------------
    main_py = b"""
import vs, helper
vs.emit("edges", helper.edges(vs.image, vs.args["lo"], vs.args["hi"]))
vs.emit("meta", {"n_images": len(vs.images), "texts": vs.texts,
                 "numbers": vs.numbers, "inputs": sorted(vs.inputs)})
print("helper said", helper.NAME)
"""
    helper_py = b"""
import cv2
NAME = "hello-from-helper"
def edges(img, lo, hi):
    return cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), lo, hi)
"""
    buf = io.BytesIO()
    np.savez(buf, K=np.eye(3, dtype=np.float64), pts=np.arange(6, dtype=np.float32).reshape(3, 2))
    cfg, f = call(stub,
                  {"command": "run", "args": {"lo": 50, "hi": 150}},
                  {"code": [main_py, helper_py],
                   "names": ["main.py", "helper.py"],
                   "images": [png], "texts": ["a", "b"], "numbers": [1.5, 2.5],
                   "inputs": buf.getvalue()})
    edges = decode(f, cfg, "edges") if "edges" in f else None
    meta = decode(f, cfg, "meta") if "meta" in f else None
    check("run: multi-file, helper imported",
          cfg["status"] == "done" and edges is not None and edges.shape == (60, 80),
          cfg.get("error", ""))
    check("run: args reached the script",
          meta is not None and meta["n_images"] == 1, str(meta))
    check("run: texts and numbers arrived",
          meta is not None and meta["texts"] == ["a", "b"]
          and meta["numbers"] == [1.5, 2.5], str(meta and meta.get("numbers")))
    check("run: npz bundle arrived",
          meta is not None and meta["inputs"] == ["K", "pts"],
          str(meta and meta.get("inputs")))
    check("run: stdout is returned",
          "hello-from-helper" in (f.get("stdout") or ""), repr(f.get("stdout"))[:60])
    check("run: codecs declared per field",
          (cfg.get("encoding") or {}).get("edges") == "numpy"
          and (cfg["encoding"]).get("meta") == "json", str(cfg.get("encoding")))

    # --- emitting raw bytes ----------------------------------------------
    cfg, f = call(stub, {"command": "eval"},
                  {"code": "vs.emit('png', bytes(cv2.imencode('.png', vs.image)[1]))",
                   "images": [png]})
    out_png = f.get("png")
    check("bytes come back as identity",
          (cfg.get("encoding") or {}).get("png") == "identity"
          and out_png[:4] == b"\x89PNG", str(cfg.get("encoding")))

    # --- failures ---------------------------------------------------------
    cfg, f = call(stub, {"command": "run"},
                  {"code": [b"print(1)"], "names": ["other.py"]})
    check("missing main.py is a clear error",
          cfg["status"] == "error" and "main.py" in cfg["error"], cfg.get("error", "")[:60])

    cfg, f = call(stub, {"command": "eval"}, {"code": "1/0"})
    check("a crashing script reports its traceback",
          cfg["status"] == "error" and "ZeroDivisionError" in (f.get("stderr") or ""),
          cfg.get("error", "")[:50])

    cfg, f = call(stub, {"command": "eval", "parameters": {"timeout": 2}},
                  {"code": "import time\nwhile True: time.sleep(0.1)"})
    check("an endless script is killed",
          cfg["status"] == "error" and cfg.get("timed_out") is True,
          cfg.get("error", "")[:50])

    cfg, f = call(stub, {"command": "eval"},
                  {"code": "vs.emit('x', object())"})
    check("an unsendable value is refused, not pickled",
          cfg["status"] == "error" and "TypeError" in (f.get("stderr") or ""),
          cfg.get("error", "")[:50])

    cfg, _ = call(stub, {"command": "frobnicate"}, {"code": "1"})
    check("unknown command rejected", cfg["status"] == "error",
          cfg.get("error", "")[:50])

    cfg, _ = call(stub, {"command": "run"})
    check("no code at all is empty_request", cfg["status"] == "empty_request")

    cfg, _ = call(stub, {"command": "eval", "parameters": {"timeout": 9999}},
                  {"code": "1"})
    check("an absurd timeout is capped", cfg["status"] == "error"
          and "capped" in cfg["error"], cfg.get("error", "")[:40])

    # --- a script that spawns children is killed with its group -----------
    spawn = ("import subprocess, sys, time\n"
             "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
             "print('spawned', flush=True)\n"
             "time.sleep(120)\n")
    t0 = time.time()
    cfg, f = call(stub, {"command": "eval", "parameters": {"timeout": 3}},
                  {"code": spawn})
    check("a spawning script is killed with its group",
          cfg.get("timed_out") is True and time.time() - t0 < 30,
          f"{round(time.time() - t0, 1)}s")

    # --- memory limit ------------------------------------------------------
    cfg, f = call(stub, {"command": "eval",
                         "parameters": {"max_memory_mb": 256, "timeout": 20}},
                  {"code": "import numpy as np\nx = np.zeros((20000, 20000), np.float64)"})
    check("the memory cap bites", cfg["status"] == "error",
          (cfg.get("error") or "")[:40])

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", ", ".join(FAIL))
    if server:
        server.stop(0)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
