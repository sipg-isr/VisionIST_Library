#!/usr/bin/env python3
"""Tests for the internvl box.

    python3 test/test_internvl.py                    # in-process, no model needed
    python3 test/test_internvl.py --host host:port   # against a running box
    BOX_HOST=host:port python3 test/test_internvl.py

Two layers, on purpose:

* **in-process** (the default) serves the box in this process with a stand-in
  engine in place of InternVL, so it needs neither torch nor the weights. It
  checks what can be wrong without the model: the tiling (the same algorithm
  OpenFACADES and the InternVL model cards use), the normalisation, how
  questions are paired with images, the parameter checks, the JSON parsing and
  the envelope - every status, every declared encoding, decoded the way
  visionist_client decodes it;
* **--host** drives a real box: `info`, `reset`, then real answers for a
  synthetic image, in both modes. Run it after building the image.

No image fixtures: frames are drawn with PIL, so nothing large is committed.
"""

import argparse
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _ensure_stubs():
    """The generated pipeline_pb2 modules are a build product (the Dockerfile
    compiles them). Compile them into a temp dir when a checkout has none."""
    try:
        sys.path.insert(0, str(ROOT / "protos"))
        import pipeline_pb2  # noqa: F401
        return
    except ImportError:
        sys.path.pop(0)
    out = tempfile.mkdtemp(prefix="internvl_pb2_")
    subprocess.run([sys.executable, "-m", "grpc_tools.protoc",
                    f"-I{ROOT / 'protos'}", f"--python_out={out}",
                    f"--grpc_python_out={out}", "pipeline.proto"], check=True)
    sys.path.insert(0, out)
    sys.path.insert(1, str(ROOT / "protos"))


_ensure_stubs()
sys.path.insert(0, str(ROOT / "src"))

import grpc                                                     # noqa: E402
import pipeline_pb2                                             # noqa: E402
import pipeline_pb2_grpc                                        # noqa: E402
from aux import wrap_value, unwrap_value                        # noqa: E402

import internvl_service as svc                                  # noqa: E402

PASS, FAIL, SKIP = [], [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"{'ok  ' if condition else 'FAIL'} {name:50s} {detail}")


def skip(name, why):
    SKIP.append(name)
    print(f"skip {name:50s} {why}")


# ------------------------------------------------------------------ helpers
def image_bytes(width=640, height=480, fmt="JPEG", seed=0):
    """A synthetic street-ish frame: sky, a house with windows, a road."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (width, height), (150, 190, 230))
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(height * .75), width, height], fill=(80, 80, 80))
    d.rectangle([int(width * .25), int(height * .3), int(width * .75),
                 int(height * .75)], fill=(200, 120, 90))
    for k in range(3):
        x = int(width * (.3 + .15 * k))
        d.rectangle([x, int(height * .4), x + width // 16, int(height * .5)],
                    fill=(60, 80, 120))
    arr = np.asarray(img).astype(np.int16) + rng.integers(-6, 6, (height, width, 3))
    buf = io.BytesIO()
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(buf, fmt)
    return buf.getvalue()


def call(stub, section, data=None):
    env = pipeline_pb2.Envelope(config_json=json.dumps({"internvl": section}))
    for k, v in (data or {}).items():
        env.data[k].CopyFrom(wrap_value(v))
    reply = stub.Process(env, timeout=600)
    cfg = json.loads(reply.config_json)["internvl"]
    fields = {k: unwrap_value(v) for k, v in reply.data.items()}
    return cfg, fields


def decode(fields, cfg, name):
    """Apply the codec the box declared, as visionist_client does."""
    codec = (cfg.get("encoding") or {}).get(name)
    raw = fields[name]
    if codec == "json":
        return json.loads(raw.decode("utf-8"))
    return raw


class FakeEngine:
    """Stands in for InternVLEngine: same interface, no torch. Answers echo
    what the model would have been given, so pairing and tiling are visible."""

    image_size = 448

    def __init__(self, answer=None):
        self.answer = answer
        self.calls = []

    def ensure_loaded(self):
        pass

    def describe(self):
        return {"model": "fake/InternVL3-test", "loaded": True, "device": "cpu",
                "dtype": "fp32", "flash_attn": False, "image_size": 448}

    def answer_each(self, arrays, questions, p):
        self.calls.append(("each", [a.shape for a in arrays], list(questions)))
        return [self.answer or f"{q} | tiles={a.shape[0]}"
                for a, q in zip(arrays, questions)]

    def answer_together(self, arrays, question, p):
        self.calls.append(("together", [a.shape for a in arrays], question))
        return [self.answer or f"{question} | images={len(arrays)}"]

    def is_oom(self, exc):
        return False

    def free_cache(self):
        pass


def serve(engine):
    import concurrent.futures as futures
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pipeline_pb2_grpc.add_PipelineServiceServicer_to_server(
        svc.PipelineService(engine=engine), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, f"127.0.0.1:{port}"


# ------------------------------------------------------------------ pure parts
def test_tiling():
    img = Image.open(io.BytesIO(image_bytes(640, 480))).convert("RGB")
    tiles = svc.dynamic_preprocess(img, max_num=12, image_size=448,
                                   use_thumbnail=True)
    # 4:3 -> the 4x3 grid (12 tiles) plus the thumbnail.
    check("tiling: 640x480, max 12 -> 4x3 + thumbnail", len(tiles) == 13,
          f"{len(tiles)} tiles")
    check("tiling: every tile is 448x448",
          all(t.size == (448, 448) for t in tiles))

    tiles = svc.dynamic_preprocess(img, max_num=1, image_size=448,
                                   use_thumbnail=True)
    check("tiling: max 1 -> one tile, no thumbnail", len(tiles) == 1)

    wide = Image.new("RGB", (3000, 500))
    tiles = svc.dynamic_preprocess(wide, max_num=6, image_size=448)
    check("tiling: a 6:1 panorama -> a 6x1 strip", len(tiles) == 6)

    arr = svc.tiles_to_array(tiles[:2], 448)
    check("tiles_to_array: (N, 3, S, S) float32",
          arr.shape == (2, 3, 448, 448) and arr.dtype == np.float32)
    white = svc.tiles_to_array([Image.new("RGB", (448, 448), (255, 255, 255))], 448)
    expect = (1.0 - svc.IMAGENET_MEAN) / svc.IMAGENET_STD
    check("tiles_to_array: ImageNet normalisation",
          np.allclose(white[0, :, 0, 0], expect, atol=1e-5))
    small = svc.tiles_to_array([Image.new("L", (100, 50))], 448)
    check("tiles_to_array: resizes and converts stray tiles",
          small.shape == (1, 3, 448, 448))


def test_questions():
    p = svc._parameters({})
    check("questions: none -> default for each image",
          svc._questions([], p, 3) == [svc.DEFAULT_QUESTION] * 3)
    check("questions: one -> shared", svc._questions(["q"], p, 2) == ["q", "q"])
    check("questions: one per image -> paired",
          svc._questions(["a", "b"], p, 2) == ["a", "b"])
    for bad, why in ((["a", "b"], 3), ):
        try:
            svc._questions(bad, p, why)
            check("questions: 2 for 3 images is an error", False)
        except ValueError:
            check("questions: 2 for 3 images is an error", True)
    p_q = svc._parameters({"question": "from params"})
    check("questions: parameters.question used",
          svc._questions([], p_q, 1) == ["from params"])
    try:
        svc._questions(["x"], p_q, 1)
        check("questions: texts AND parameters.question is an error", False)
    except ValueError:
        check("questions: texts AND parameters.question is an error", True)
    p_t = svc._parameters({"mode": "together"})
    try:
        svc._questions(["a", "b"], p_t, 2)
        check("questions: together takes exactly one", False)
    except ValueError:
        check("questions: together takes exactly one", True)


def test_parameters():
    p = svc._parameters({"max_tiles": "6", "parse_json": True})
    check("parameters: defaults merged, strings coerced",
          p["max_tiles"] == 6 and p["max_new_tokens"] == 512 and p["parse_json"])
    p = svc._parameters({"do_sample": "false", "parse_json": "True"})
    check("parameters: 'true'/'false' strings accepted (web console)",
          p["do_sample"] is False and p["parse_json"] is True)
    for raw, name in (({"max_tiles": 0}, "max_tiles below range"),
                      ({"parse_json": "yes"}, "a non-boolean string"),
                      ({"max_new_tokens": 10 ** 6}, "max_new_tokens above range"),
                      ({"mode": "all"}, "unknown mode"),
                      ({"do_sample": "yes"}, "non-boolean do_sample"),
                      ({"batch_size": "x"}, "non-integer batch_size")):
        try:
            svc._parameters(raw)
            check(f"parameters: rejects {name}", False)
        except ValueError:
            check(f"parameters: rejects {name}", True)


def test_parse_json():
    cases = [('{"type": "residential", "floors": 3}', {"type": "residential", "floors": 3}),
             ('```json\n{"a": 1}\n```', {"a": 1}),
             ('Sure! Here it is: {"a": [1, 2]} hope it helps', {"a": [1, 2]}),
             ('["brick", "glass"]', ["brick", "glass"]),
             ("A red house with three windows.", None)]
    check("parse_json: plain, fenced, embedded, list, prose",
          all(svc.parse_json_answer(t) == want for t, want in cases))


def test_split_map():
    m = svc.split_device_map(24, 2)
    layers = [m[f"language_model.model.layers.{i}"] for i in range(24)]
    check("split_device_map: every layer placed, GPU 0 lighter",
          len(layers) == 24 and layers.count(0) < layers.count(1))
    check("split_device_map: vision tower and head on GPU 0",
          m["vision_model"] == 0 and m["language_model.lm_head"] == 0)


# ------------------------------------------------------------------ envelope
def test_envelope(stub, engine):
    img = image_bytes()

    cfg, _ = call(stub, {"command": "reset"})
    check("reset: done", cfg.get("status") == "done" and cfg.get("action") == "reset")

    cfg, _ = call(stub, {"command": "info"})
    check("info: done, names the model, no data needed",
          cfg.get("status") == "done" and cfg.get("model"))

    cfg, _ = call(stub, {})
    check("ask, no images: empty_request", cfg.get("status") == "empty_request")

    cfg, _ = call(stub, {"command": "fly"}, {"images": [img]})
    check("unknown command: error", cfg.get("status") == "error")

    cfg, f = call(stub, {"command": "ask"}, {"images": img})
    answers = decode(f, cfg, "answers")
    check("ask, one bare image (Value.b): one answer",
          cfg.get("status") == "done" and len(answers) == 1, answers[0][:60])
    check("ask: answers declared as json", cfg.get("encoding") == {"answers": "json"})
    check("ask: tiles reported", cfg.get("tiles") == [13], str(cfg.get("tiles")))

    cfg, f = call(stub, {"command": "run",
                         "parameters": {"max_tiles": 4, "use_thumbnail": False}},
                  {"images": [img, img, img], "texts": "What is the building made of?"})
    answers = decode(f, cfg, "answers")
    check("'run' alias, 3 images + one bare text: 3 answers",
          cfg.get("status") == "done" and len(answers) == 3
          and all("made of" in a for a in answers))
    # 4:3 with at most 4 tiles: no grid of <= 4 is closer to 4:3 than 1x1
    # (2x2 ties, and the tie-break needs a bigger image), so one tile each.
    check("max_tiles honoured", cfg.get("tiles") == [1, 1, 1], str(cfg.get("tiles")))
    cfg, _ = call(stub, {"parameters": {"max_tiles": 6, "use_thumbnail": False}},
                  {"images": [image_bytes(1200, 400)]})
    check("max_tiles: 3:1 image, max 6 -> 3x1", cfg.get("tiles") == [3],
          str(cfg.get("tiles")))

    cfg, f = call(stub, {}, {"images": [img, img], "texts": ["first?", "second?"]})
    answers = decode(f, cfg, "answers")
    check("one question per image: paired in order",
          answers[0].startswith("first?") and answers[1].startswith("second?"))

    cfg, f = call(stub, {"parameters": {"mode": "together",
                                        "question": "Which is taller?"}},
                  {"images": [img, image_bytes(400, 600, "PNG", 1)]})
    answers = decode(f, cfg, "answers")
    check("together: one answer about 2 images (JPEG+PNG)",
          cfg.get("status") == "done" and len(answers) == 1
          and "images=2" in answers[0])

    cfg, _ = call(stub, {}, {"images": [img, img, img], "texts": ["a", "b"]})
    check("2 questions for 3 images: error naming the counts",
          cfg.get("status") == "error" and "3 images" in cfg.get("error", ""))

    cfg, _ = call(stub, {"parameters": {"max_tiles": 99}}, {"images": [img]})
    check("max_tiles out of range: error", cfg.get("status") == "error")

    cfg, _ = call(stub, {}, {"images": [b"not an image"]})
    check("undecodable image: error naming it",
          cfg.get("status") == "error" and "image #1" in cfg.get("error", ""))

    engine.answer = '```json\n{"type": "residential", "floors": 2}\n```'
    cfg, f = call(stub, {"parameters": {"parse_json": True}}, {"images": [img]})
    engine.answer = None
    parsed = decode(f, cfg, "parsed")
    check("parse_json: parsed field declared and decoded",
          cfg.get("encoding", {}).get("parsed") == "json"
          and parsed == [{"type": "residential", "floors": 2}]
          and cfg.get("num_parsed") == 1)


def test_client(host):
    """The same through visionist_client, if it is installed."""
    try:
        from visionist_client import Visionist
    except ImportError:
        skip("visionist_client round trip", "pip install visionist-client")
        return
    b = Visionist(host)
    res = b.run(data={"images": [image_bytes()], "texts": ["What is this?"]},
                config={"internvl": {}})
    answers = res.answers
    check("visionist_client: answers decoded to a list of str",
          isinstance(answers, list) and isinstance(answers[0], str), repr(answers)[:60])


# ------------------------------------------------------------------ real box
def test_live(host):
    stub = pipeline_pb2_grpc.PipelineServiceStub(grpc.insecure_channel(
        host, options=[("grpc.max_receive_message_length", 64 << 20),
                       ("grpc.max_send_message_length", 64 << 20)]))
    cfg, _ = call(stub, {"command": "info"})
    check("live info", cfg.get("status") == "done", json.dumps(cfg)[:120])
    cfg, _ = call(stub, {"command": "reset"})
    check("live reset", cfg.get("status") == "done")

    img = image_bytes()
    cfg, f = call(stub, {"parameters": {"max_tiles": 6, "max_new_tokens": 64}},
                  {"images": [img, img],
                   "texts": "Describe the building in one sentence."})
    ok = cfg.get("status") == "done"
    answers = decode(f, cfg, "answers") if ok else []
    check("live ask (each): 2 non-empty answers",
          ok and len(answers) == 2 and all(a.strip() for a in answers),
          (answers[0][:70] if answers else cfg.get("error", ""))
          + f"  [{cfg.get('device')}, {cfg.get('dtype')}, {cfg.get('runtime', 0):.1f}s]")

    cfg, f = call(stub, {"parameters": {
        "mode": "together", "max_tiles": 4, "max_new_tokens": 64, "parse_json": True,
        "question": 'Reply with JSON only: {"same_scene": true or false}'}},
        {"images": [img, image_bytes(seed=3)]})
    ok = cfg.get("status") == "done"
    check("live ask (together, parse_json)", ok,
          repr(decode(f, cfg, "parsed"))[:70] if ok else cfg.get("error", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.getenv("BOX_HOST"))
    args = ap.parse_args()

    if args.host:
        test_live(args.host)
        test_client(args.host)
    else:
        test_tiling()
        test_questions()
        test_parameters()
        test_parse_json()
        test_split_map()
        engine = FakeEngine()
        server, host = serve(engine)
        try:
            stub = pipeline_pb2_grpc.PipelineServiceStub(grpc.insecure_channel(host))
            test_envelope(stub, engine)
            test_client(host)
        finally:
            server.stop(0)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
