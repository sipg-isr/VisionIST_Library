#!/usr/bin/env python3
"""Test the phone box end to end: a real gRPC server and a real HTTP server,
in this process, no docker and no browser.

    python3 test/test_phone.py

What a phone does is played here by a few lines of http.client; what a program
does is played by the gRPC stub. The page's JavaScript is NOT run by this file
(see README.md, "Testing the page", for the browser check).

Layers:
  1. the gRPC contract    setup / get / show / status / reset, every bad input
  2. the web contract     pairing, upload validation, limits, the feed, media
  3. the store            eviction and `dropped`, TTL, sessions, isolation
  4. deployment modes     behind a proxy (BASE_PATH, TRUST_PROXY, ALLOWED_HOSTS),
                          direct TLS (self-signed), security headers
"""

import concurrent.futures as futures
import http.client
import io
import json
import pathlib
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import uuid

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "protos"))
sys.path.insert(0, str(ROOT / "src"))

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"{'ok  ' if condition else 'FAIL'} {name:62s} {detail}")


def _ensure_pb2():
    """pipeline_pb2 is generated, not committed: build it into a temp dir."""
    try:
        import pipeline_pb2  # noqa: F401
        return
    except ImportError:
        pass
    out = tempfile.mkdtemp(prefix="phone_pb2_")
    subprocess.run([sys.executable, "-m", "grpc_tools.protoc", f"-I{ROOT / 'protos'}",
                    f"--python_out={out}", f"--grpc_python_out={out}",
                    str(ROOT / "protos" / "pipeline.proto")], check=True)
    sys.path.insert(0, out)


_ensure_pb2()
import grpc                                                     # noqa: E402
import pipeline_pb2                                             # noqa: E402
import pipeline_pb2_grpc                                        # noqa: E402
from aux import wrap_value, unwrap_value                        # noqa: E402
import serve                                                    # noqa: E402

import phone_service as svc                                     # noqa: E402
from phone_settings import Settings, MB                         # noqa: E402
from phone_store import Store                                   # noqa: E402
from phone_web import WebServer, clean_meta, ApiError           # noqa: E402

# ------------------------------------------------------------------ fixtures
JPEG = b"\xff\xd8\xff\xe0" + bytes(range(256)) * 8          # the box never decodes
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
WEBP = b"RIFF\x10\x00\x00\x00WEBP" + b"\x00" * 16
GLB = b"glTF" + b"\x02\x00\x00\x00" + b"\x00" * 32
NOT_AN_IMAGE = b"GIF89a" + b"\x00" * 32


def jpeg(n=0, size=1024):
    return JPEG[:4] + bytes([n % 256]) * size


class FakeContext:
    def is_active(self):
        return True


def base_settings(**kw):
    kw.setdefault("web_host", "127.0.0.1")
    kw.setdefault("web_port", 0)
    kw.setdefault("message_cap", 64 * MB)
    return Settings(**kw)


class Rig:
    """One store + gRPC servicer + web server, with helpers for both faces."""

    def __init__(self, **kw):
        self.settings = base_settings(**kw)
        self.store = Store(self.settings)
        self.servicer = svc.PipelineService(self.store, self.settings)
        self.web = WebServer(self.store, self.settings).start()
        self.port = self.web.port
        self.tls = self.settings.tls

    def close(self):
        self.web.stop()

    # ---- the program's side (gRPC messages, called directly) --------------
    def call(self, command, params=None, data=None, session=None):
        section = {"command": command, "parameters": params or {}}
        if session is not None:
            section["session_id"] = session
        env = pipeline_pb2.Envelope(config_json=json.dumps({"phone": section}))
        for k, v in (data or {}).items():
            env.data[k].CopyFrom(wrap_value(v))
        return unpack(self.servicer.Process(env, FakeContext()))

    # ---- the phone's side (HTTP) -------------------------------------------
    def http(self, method, path, token=None, headers=None, body=None):
        if self.tls:
            conn = http.client.HTTPSConnection("127.0.0.1", self.port, timeout=15,
                                               context=ssl._create_unverified_context())
        else:
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        h = dict(headers or {})
        if token:
            h["Authorization"] = "Bearer " + token
        try:
            conn.request(method, path, body=body, headers=h)
            r = conn.getresponse()
            raw = r.read()
            return r.status, {k.lower(): v for k, v in r.getheaders()}, raw
        finally:
            conn.close()

    def jget(self, path, token=None, headers=None):
        status, hdr, raw = self.http("GET", path, token, headers)
        try:
            return status, json.loads(raw.decode())
        except ValueError:
            return status, raw

    def submit(self, token, images, meta=None, headers=None):
        parts = [("image", f"s{i}.jpg", "image/jpeg", b) for i, b in enumerate(images)]
        if meta is not None:
            parts.append(("meta", None, "application/json",
                          meta if isinstance(meta, bytes) else json.dumps(meta).encode()))
        body, ctype = multipart(parts)
        h = {"Content-Type": ctype, **(headers or {})}
        status, hdr, raw = self.http("POST", "/api/submit", token, h, body)
        try:
            return status, json.loads(raw.decode())
        except ValueError:
            return status, raw


def multipart(parts):
    boundary = uuid.uuid4().hex
    out = io.BytesIO()
    for name, filename, ctype, blob in parts:
        out.write(f"--{boundary}\r\n".encode())
        disp = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
        out.write(f"Content-Disposition: {disp}\r\n".encode())
        if ctype:
            out.write(f"Content-Type: {ctype}\r\n".encode())
        out.write(b"\r\n" + blob + b"\r\n")
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


def unpack(reply):
    cfg = json.loads(reply.config_json)["phone"]
    fields = {k: unwrap_value(v) for k, v in reply.data.items()}
    enc = cfg.get("encoding") or {}
    for name, value in list(fields.items()):
        if enc.get(name) == "json":
            fields[name] = json.loads(value.decode())
    return cfg, fields


def setup(rig, token_only=True, **params):
    cfg, f = rig.call("setup", {"public_url": "https://phone.example.org", **params})
    assert cfg["status"] == "done", cfg
    return cfg, f


# ============================================================ 1. gRPC contract
def test_grpc():
    print("\n-- gRPC contract")
    rig = Rig()
    try:
        cfg, f = rig.call("setup", {"labels": ["door", "valve", "crack"], "mode": "burst",
                                    "burst_max": 4, "title": "Plant survey",
                                    "sensors": ["location", "orientation"],
                                    "public_url": "https://phone.example.org/phone/"})
        check("setup: done", cfg["status"] == "done", str(cfg.get("error")))
        check("setup: token is long and url-safe",
              len(cfg["token"]) >= 30 and cfg["token"].replace("-", "").replace("_", "").isalnum())
        check("setup: url carries the token in the FRAGMENT",
              cfg["url"] == f"https://phone.example.org/phone/#t={cfg['token']}", cfg["url"])
        check("setup: no query string leaks the token", "?" not in cfg["url"])
        check("setup: https url -> https true, no warnings", cfg["https"] is True and not cfg["warnings"])
        check("setup: QR is a PNG", f.get("qr", b"")[:8] == b"\x89PNG\r\n\x1a\n")
        check("setup: encoding declares the QR", cfg["encoding"] == {"qr": "identity"})
        check("setup: normalized config echoed",
              cfg["config"]["labels"] == ["door", "valve", "crack"] and cfg["config"]["burst_max"] == 4)
        check("setup: created first time", cfg["created"] is True and cfg["config_version"] == 1)

        cfg2, _ = rig.call("setup", {"labels": ["door", "valve", "crack"], "mode": "burst",
                                     "burst_max": 4, "title": "Plant survey",
                                     "sensors": ["location", "orientation"]})
        check("setup again: same token, same version when nothing changed",
              cfg2["token"] == cfg["token"] and cfg2["config_version"] == 1 and cfg2["created"] is False)
        check("setup without a public url: path + warning, no QR",
              cfg2["url"] is None and cfg2["path"].endswith("#t=" + cfg["token"])
              and "qr" not in (_ := rig.call("setup", {"labels": ["a"]})[1])
              and any("public URL" in w for w in rig.call("setup", {"labels": ["a"]})[0]["warnings"]))
        rig.call("setup", {"labels": ["door", "valve", "crack"], "mode": "burst",
                           "burst_max": 4, "title": "Plant survey",
                           "sensors": ["location", "orientation"]})
        cfg3, _ = rig.call("setup", {"labels": ["door"], "public_url": "http://192.168.1.20:9080"})
        check("setup: changed config bumps the version", cfg3["config_version"] > 1)
        check("setup: plain http to a LAN host warns about the camera",
              cfg3["https"] is False and any("https" in w for w in cfg3["warnings"]))
        cfg4, _ = rig.call("setup", {"labels": ["door"], "public_url": "http://localhost:9080"})
        check("setup: plain http to localhost does not warn", not cfg4["warnings"])
        cfg5, _ = rig.call("setup", {"labels": ["door"], "rotate": True,
                                     "public_url": "https://phone.example.org"})
        check("setup rotate: new token", cfg5["token"] != cfg["token"])

        bad = [
            ({"mode": "triple"}, "mode"), ({"labels": 3}, "labels"),
            ({"labels": ["x" * 65]}, "labels"), ({"burst_max": 0}, "burst_max"),
            ({"burst_max": 99}, "burst_max"), ({"camera": "rear"}, "camera"),
            ({"jpeg_quality": 2}, "jpeg_quality"), ({"sensors": ["gps"]}, "sensors"),
            ({"note": "maybe"}, "note"), ({"public_url": "ftp://x"}, "public_url"),
            ({"public_url": "https://x/?a=1"}, "public_url"), ({"max_side": -1}, "max_side"),
            ({"title": 5}, "title"),
        ]
        for params, key in bad:
            c, _ = rig.call("setup", params)
            check(f"setup rejects {json.dumps(params)[:34]}",
                  c["status"] == "error" and key in c["error"], c.get("error", "")[:50])

        c, _ = rig.call("setup", {"note": "false", "custom_labels": "True", "require_label": 1,
                                  "public_url": "https://phone.example.org"})
        check("booleans accept 'true'/'false' text (a form's select) and 0/1",
              c["status"] == "done" and c["config"]["note"] is False
              and c["config"]["custom_labels"] is True and c["config"]["require_label"] is True)
        c, _ = rig.call("explode")
        check("unknown command is an error", c["status"] == "error" and "explode" in c["error"])
        c, _ = rig.call("get", session="bad id!")
        check("bad session_id is an error", c["status"] == "error" and "session_id" in c["error"])
        c, _ = rig.call("get", session="nobody")
        check("get on an unknown session says to call setup", c["status"] == "error" and "setup" in c["error"])
        c, _ = rig.call("show", {"text": "x"}, session="nobody")
        check("show on an unknown session is an error", c["status"] == "error")

        env = pipeline_pb2.Envelope(config_json=json.dumps({"yolo": {}}))
        c = json.loads(rig.servicer.Process(env, FakeContext()).config_json)["phone"]
        check("foreign config section is an error", c["status"] == "error")
        c = json.loads(rig.servicer.Process(pipeline_pb2.Envelope(), FakeContext()).config_json)["phone"]
        check("empty config is an error", c["status"] == "error")
        env = pipeline_pb2.Envelope(config_json=json.dumps({"phone": {"parameters": []}}))
        c = json.loads(rig.servicer.Process(env, FakeContext()).config_json)["phone"]
        check("parameters must be an object", c["status"] == "error")
        c, _ = rig.call("status")
        check("status without a section command defaults... status works", c["status"] == "done")
        rig.call("reset")
        env = pipeline_pb2.Envelope(config_json=json.dumps({"phone": {}}))
        c = json.loads(rig.servicer.Process(env, FakeContext()).config_json)["phone"]
        check("no command defaults to get (needs a session)", c["status"] == "error" and "setup" in c["error"])
        rig.call("setup", {"labels": ["a"]})
        c = json.loads(rig.servicer.Process(env, FakeContext()).config_json)["phone"]
        check("... and with a session it is a get", c["status"] == "done" and c["action"] == "get")
    finally:
        rig.close()


# ============================================== 2. phone <-> program round trip
def test_roundtrip():
    print("\n-- round trip: phone submits, program polls, program shows, phone sees it")
    rig = Rig()
    try:
        cfg, _ = setup(rig, labels=["door", "valve", "crack"], mode="burst", burst_max=3,
                       sensors=["location", "orientation"], title="Survey")
        tok = cfg["token"]

        st, _ = rig.jget("/api/config")
        check("web: no token -> 401", st == 401)
        st, _ = rig.jget("/api/config", token="x" * 32)
        check("web: wrong token -> 401", st == 401)
        st, body = rig.jget("/healthz")
        check("web: healthz needs no token", st == 200 and body == {"ok": True})
        st, body = rig.jget("/api/config?device=d1&name=Pixel", token=tok)
        check("web: config with the token", st == 200 and body["config"]["title"] == "Survey"
              and body["config"]["mode"] == "burst" and body["limits"]["max_images"] == 12)

        cfgr, f = rig.call("get")
        check("get: nothing yet -> done, count 0, empty meta",
              cfgr["status"] == "done" and cfgr["count"] == 0 and f["meta"] == [] and "images" not in f)

        sensors = [{"location": {"lat": 38.7369, "lon": -9.1386, "acc": 5.0, "t": 1.0, "junk": float("nan")},
                    "orientation": {"alpha": 10, "beta": 20, "gamma": 30},
                    "motion": {"ax": 1.0}},          # motion not enabled -> dropped
                   None, {"location": {"lat": 1.0}}]
        st, r = rig.submit(tok, [jpeg(1), PNG, WEBP], {
            "labels": ["door", "crack"], "note": "  left wall  ", "sensors": sensors,
            "device": {"id": "d1", "name": "Pixel"}, "client_t": 1700000000.5})
        check("submit: 3 images accepted", st == 200 and r["ok"] and r["seq"] == 1 and r["images"] == 3, str(r))
        st, r2 = rig.submit(tok, [jpeg(2)], {"labels": ["valve"]})
        check("submit: second gets seq 2", st == 200 and r2["seq"] == 2)

        c, f = rig.call("get")
        meta = f["meta"]
        check("get: both submissions, oldest first", c["count"] == 2 and [m["seq"] for m in meta] == [1, 2])
        check("get: images flattened, first_image/n_images locate them",
              len(f["images"]) == 4 and meta[0]["first_image"] == 0 and meta[0]["n_images"] == 3
              and meta[1]["first_image"] == 3 and f["images"][0] == jpeg(1) and f["images"][3] == jpeg(2))
        check("get: bytes are exactly what the phone sent", f["images"][1] == PNG and f["images"][2] == WEBP)
        check("get: labels, note (trimmed), device, client_t",
              meta[0]["labels"] == ["door", "crack"] and meta[0]["note"] == "left wall"
              and meta[0]["device"] == {"id": "d1", "name": "Pixel"} and meta[0]["client_t"] == 1700000000.5)
        check("get: mime types sniffed", meta[0]["mime"] == ["image/jpeg", "image/png", "image/webp"])
        s0 = meta[0]["sensors"]
        check("get: sensors aligned with images, NaN and disabled groups dropped",
              len(s0) == 3 and s0[0] == {"location": {"lat": 38.7369, "lon": -9.1386, "acc": 5.0, "t": 1.0},
                                         "orientation": {"alpha": 10, "beta": 20, "gamma": 30}}
              and s0[1] is None and s0[2] == {"location": {"lat": 1.0}}, str(s0))
        check("get: no sensors -> null", meta[1]["sensors"] is None)
        check("get: encoding map", c["encoding"] == {"images": "identity", "meta": "json"})
        check("get: seq / next_after / pending / dropped",
              c["seq"] == 2 and c["next_after"] == 2 and c["pending"] == 0 and c["dropped"] == 0)
        check("get is idempotent (pop false): same again", rig.call("get")[0]["count"] == 2)

        c, f = rig.call("get", {"after": 1})
        check("get after=1 -> only seq 2", c["count"] == 1 and f["meta"][0]["seq"] == 2)
        c, f = rig.call("get", {"max_items": 1})
        check("get max_items=1 -> one, pending 1, next_after 1",
              c["count"] == 1 and c["pending"] == 1 and c["next_after"] == 1)
        c, f = rig.call("get", {"latest": True})
        check("get latest -> only the newest", c["count"] == 1 and f["meta"][0]["seq"] == 2 and c["pending"] == 0)
        c, f = rig.call("get", {"latest": True, "after": 2})
        check("get latest after the newest -> nothing", c["count"] == 0)

        c, f = rig.call("get", {"pop": True, "max_items": 1})
        check("get pop deletes what it returned", c["count"] == 1 and rig.call("get")[1]["meta"][0]["seq"] == 2)
        rig.call("get", {"pop": True})
        c, _ = rig.call("get")
        check("after pop the queue is empty", c["count"] == 0)

        # --- long poll: a phone submits half a second into a 5 s wait ---------
        threading.Timer(0.5, lambda: rig.submit(tok, [jpeg(9)], {"labels": []})).start()
        t0 = time.time()
        c, f = rig.call("get", {"wait_s": 5, "after": 2})
        dt = time.time() - t0
        check("long poll wakes when a photo arrives", c["count"] == 1 and 0.3 < dt < 3, f"{dt:.2f}s")
        t0 = time.time()
        c, _ = rig.call("get", {"wait_s": 0.6, "after": 99})
        dt = time.time() - t0
        check("long poll times out with count 0", c["count"] == 0 and 0.5 < dt < 2, f"{dt:.2f}s")
        c, _ = rig.call("get", {"wait_s": 999})
        check("wait_s over MAX_WAIT_S is rejected", c["status"] == "error")

        # --- show -> feed -> media ---------------------------------------------
        c, _ = rig.call("show", {"title": "Doors", "text": "2 doors found", "ref_seq": 1},
                        data={"image": jpeg(7, 500), "glb": GLB})
        check("show: done, card_id 1", c["status"] == "done" and c["card_id"] == 1, str(c.get("error")))
        c, _ = rig.call("show", {"text": "second"})
        check("show: text only", c["status"] == "done" and c["card_id"] == 2)
        c, _ = rig.call("show", data={"text": "from data"})
        check("show: text from data.text", c["status"] == "done")
        c, _ = rig.call("show")
        check("show: nothing -> empty_request", c["status"] == "empty_request")
        c, _ = rig.call("show", data={"image": NOT_AN_IMAGE})
        check("show: GIF is refused", c["status"] == "error" and "JPEG" in c["error"])
        c, _ = rig.call("show", data={"glb": b"not a glb"})
        check("show: non-glb refused", c["status"] == "error" and "glTF" in c["error"])
        c, _ = rig.call("show", {"text": "x" * 20001})
        check("show: huge text refused", c["status"] == "error")
        c, _ = rig.call("show", {"text": "x", "ref_seq": -1})
        check("show: bad ref_seq refused", c["status"] == "error")

        st, feed = rig.jget("/api/feed?after=0", token=tok)
        check("feed: three cards", st == 200 and [x["id"] for x in feed["cards"]] == [1, 2, 3], str(feed)[:100])
        c1 = feed["cards"][0]
        check("feed: card fields", c1["title"] == "Doors" and c1["text"] == "2 doors found"
              and c1["ref_seq"] == 1 and c1["image"] and c1["glb"] and c1["image_mime"] == "image/jpeg")
        check("feed: reports the program's last poll", feed["program_age_s"] is not None and feed["program_age_s"] < 10)
        st, hdr, raw = rig.http("GET", "/api/media/1/image", tok)
        check("media: image bytes identical, content-type", st == 200 and raw == jpeg(7, 500)
              and hdr["content-type"] == "image/jpeg")
        st, hdr, raw = rig.http("GET", "/api/media/1/glb", tok)
        check("media: glb bytes identical, model type", st == 200 and raw == GLB and hdr["content-type"] == "model/gltf-binary")
        check("media: text-only card has no image", rig.http("GET", "/api/media/2/image", tok)[0] == 404)
        check("media: unknown card 404", rig.http("GET", "/api/media/99/image", tok)[0] == 404)
        check("media: bad kind 404", rig.http("GET", "/api/media/1/exe", tok)[0] == 404)
        check("media: needs the token", rig.http("GET", "/api/media/1/image")[0] == 401)
        st, feed2 = rig.jget("/api/feed?after=2", token=tok)
        check("feed after=2 -> card 3 only", [x["id"] for x in feed2["cards"]] == [3])

        threading.Timer(0.5, lambda: rig.call("show", {"text": "late"})).start()
        t0 = time.time()
        st, feed3 = rig.jget("/api/feed?after=3&wait=5", token=tok)
        dt = time.time() - t0
        check("feed long-poll wakes on a new card", st == 200 and len(feed3["cards"]) == 1 and dt < 3, f"{dt:.2f}s")
        c, _ = rig.call("show", {"text": "only me", "replace": True})
        st, feed4 = rig.jget("/api/feed?after=0", token=tok)
        check("show replace clears earlier cards", [x["text"] for x in feed4["cards"]] == ["only me"])

        # --- config change wakes the long poll ---------------------------------
        cv = rig.jget("/api/config", token=tok)[1]["config_version"]
        threading.Timer(0.4, lambda: rig.call("setup", {"labels": ["other"], "mode": "burst", "burst_max": 3,
                                                        "sensors": ["location", "orientation"], "title": "Survey",
                                                        "public_url": "https://phone.example.org"})).start()
        t0 = time.time()
        st, feed5 = rig.jget(f"/api/feed?after=99&wait=5&cv={cv}", token=tok)
        check("feed returns early with the new config_version", feed5["config_version"] == cv + 1
              and time.time() - t0 < 3)
        st, body = rig.jget("/api/config", token=tok)
        check("config reflects the new labels", body["config"]["labels"] == ["other"])
        st, r = rig.submit(tok, [jpeg(1)], {"labels": ["door"]})
        check("a label the session no longer offers is refused", st == 400 and "door" in r["error"])

        c, _ = rig.call("status")
        check("status: queue, devices, phone and program ages",
              c["exists"] and c["queue"]["count"] >= 1 and any(d["id"] == "d1" for d in c["devices"])
              and c["phone_last_seen_s"] is not None and c["program_last_poll_s"] is not None
              and c["sessions"] == ["default"] and c["web"]["tls"] is False, str(c)[:140])
        c, _ = rig.call("status", session="ghost")
        check("status of an unknown session: exists false", c["status"] == "done" and c["exists"] is False)

        # --- reset ---------------------------------------------------------------
        c, _ = rig.call("reset", session="default")
        check("reset one session", c["status"] == "done" and c["sessions_removed"] == 1)
        check("after reset the phone's token is refused", rig.jget("/api/config", token=tok)[0] == 401)
        c, _ = rig.call("get")
        check("after reset get says to call setup", c["status"] == "error")
        c, _ = rig.call("reset")
        check("reset with nothing to reset is fine", c["status"] == "done" and c["sessions_removed"] == 0)
    finally:
        rig.close()


# ====================================================== 3. upload validation
def test_upload_validation():
    print("\n-- upload validation and limits")
    rig = Rig(message_cap=2 * MB, rate_per_min=1000)
    try:
        cfg, _ = setup(rig, labels=["a", "b"], mode="single", require_label=True)
        tok = cfg["token"]
        ok = {"labels": ["a"]}
        st, r = rig.submit(tok, [jpeg()], ok)
        check("accepts a valid single photo", st == 200)
        st, r = rig.submit(tok, [jpeg(), jpeg()], ok)
        check("single mode refuses a second photo", st == 400 and "at most 1" in r["error"])
        st, r = rig.submit(tok, [], ok)
        check("refuses a submission with no image", st == 400)
        st, r = rig.submit(tok, [NOT_AN_IMAGE], ok)
        check("refuses non-image bytes (magic number)", st == 415)
        st, r = rig.submit(tok, [b"<html>" + b"x" * 40], ok)
        check("refuses html posing as an image", st == 415)
        st, r = rig.submit(tok, [jpeg()], {"labels": []})
        check("require_label enforced", st == 400 and "label" in r["error"])
        st, r = rig.submit(tok, [jpeg()], {"labels": ["zzz"]})
        check("a label not offered is refused", st == 400)
        st, r = rig.submit(tok, [jpeg()], {"labels": "a"})
        check("labels must be a list", st == 400)
        st, r = rig.submit(tok, [jpeg()], {"labels": [1]})
        check("labels must be strings", st == 400)
        st, r = rig.submit(tok, [jpeg()], b"{not json")
        check("meta must be JSON", st == 400)
        st, r = rig.submit(tok, [jpeg()], b"[1,2]")
        check("meta must be an object", st == 400)
        st, r = rig.submit(tok, [jpeg()], {"labels": ["a"], "note": 5})
        check("note must be a string", st == 400)
        st, r = rig.submit(tok, [jpeg()], b"{" + b" " * 20000 + b"}")
        check("oversized meta refused", st == 413)
        st, r = rig.submit(tok, [jpeg(size=int(1.7 * MB))], ok)
        check("over 80% of the gRPC message cap -> 413", st == 413, str(r)[:60])
        st, r = rig.submit(tok, [jpeg(size=int(1.4 * MB))], ok)
        check("just under the cap is accepted", st == 200)

        body, ctype = multipart([("image", "a.jpg", "image/jpeg", jpeg()), ("evil", None, None, b"x")])
        st, _, raw = rig.http("POST", "/api/submit", tok, {"Content-Type": ctype}, body)
        check("unexpected multipart part refused", st == 400)
        st, _, raw = rig.http("POST", "/api/submit", tok, {"Content-Type": "application/json"}, b"{}")
        check("non-multipart refused (415)", st == 415)
        st, _, raw = rig.http("POST", "/api/submit", None, {"Content-Type": ctype}, body)
        check("submit without a token -> 401", st == 401)
        st, _, raw = rig.http("GET", "/api/submit", tok)
        check("GET on submit -> 405", st == 405)

        # the reply must always fit one gRPC message: a stored submission is gettable
        c, f = rig.call("get", {"after": 0, "max_items": 64, "max_mb": 4096})
        check("everything stored can be read back in one reply", c["status"] == "done" and c["count"] >= 2)
        c, f = rig.call("get", {"max_mb": 0.1})
        check("get max_mb stops before a submission that would exceed it",
              c["count"] == 1 and c["pending"] >= 1, f"count {c['count']} pending {c['pending']}")
        c, f = rig.call("get", {"max_mb": 0.1, "after": 1})
        check("... but a lone submission over max_mb is still returned (never stuck)",
              c["count"] == 1 and len(f["images"][0]) > 0.1 * MB)

        # labels are free-form when custom_labels is on
        setup(rig, labels=["a"], custom_labels=True)
        st, r = rig.submit(tok, [jpeg()], {"labels": ["  my  own   label "]})
        check("custom_labels accepts a new label (whitespace normalized)", st == 200)
        c, f = rig.call("get", {"latest": True})
        check("custom label stored normalized", f["meta"][0]["labels"] == ["my own label"])
        setup(rig, labels=["a"], note=False)
        st, r = rig.submit(tok, [jpeg()], {"labels": ["a"], "note": "hidden"})
        c, f = rig.call("get", {"latest": True})
        check("note dropped when the session has notes off", f["meta"][0]["note"] == "")

        # unit: clean_meta directly
        labels = clean_meta(json.dumps({"labels": ["a", " a ", "b", "a"]}).encode(),
                            {"labels": ["a", "b"]})[0]
        check("clean_meta: duplicates collapse", labels == ["a", "b"], str(labels))
        try:
            clean_meta(json.dumps({"labels": [str(i) for i in range(40)]}).encode(), {"custom_labels": True})
            check("clean_meta: more than 32 labels refused", False)
        except ApiError:
            check("clean_meta: more than 32 labels refused", True)
    finally:
        rig.close()


# ===================================================== 4. limits and security
def test_security():
    print("\n-- security: lockout, rate limit, isolation, headers, paths")
    rig = Rig(auth_fail_limit=3, rate_per_min=4)
    try:
        a, _ = setup(rig, labels=["a"], session_id=None) if False else (None, None)
        cfg_a, _ = rig.call("setup", {"labels": ["a"], "public_url": "https://x.example"}, session="alice")
        cfg_b, _ = rig.call("setup", {"labels": ["b"], "public_url": "https://x.example"}, session="bob")
        ta, tb = cfg_a["token"], cfg_b["token"]
        check("two sessions, two tokens", ta != tb)
        rig.call("show", {"text": "alice only"}, session="alice")
        check("bob cannot read alice's card", rig.http("GET", "/api/media/1/image", tb)[0] == 404
              and rig.jget("/api/feed?after=0", tb)[1]["cards"] == [])
        st, r = rig.submit(ta, [jpeg()], {"labels": ["b"]})
        check("alice cannot use bob's labels", st == 400)
        st, r = rig.submit(ta, [jpeg()], {"labels": ["a"]})
        check("alice submits", st == 200)
        c, f = rig.call("get", session="bob")
        check("bob's queue stays empty", c["count"] == 0)
        c, f = rig.call("get", session="alice")
        check("alice's queue has her photo", c["count"] == 1)
        c, _ = rig.call("status")
        check("status lists both sessions", c["sessions"] == ["alice", "bob"])

        for i in range(3):
            st, _ = rig.jget("/api/config", token="w" * 32)
            check(f"wrong token #{i + 1} -> 401", st == 401)
        st, hdr, raw = rig.http("GET", "/api/config", token="w" * 32)
        check("4th wrong token -> 429 with Retry-After", st == 429 and hdr.get("retry-after") == "300")
        st, hdr, raw = rig.http("GET", "/api/config", token=ta)
        check("a locked-out address is refused even with a good token", st == 429)
        check("healthz and the page still answer while locked out",
              rig.http("GET", "/healthz")[0] == 200 and rig.http("GET", "/")[0] == 200)
    finally:
        rig.close()

    rig = Rig(rate_per_min=3)
    try:
        cfg, _ = rig.call("setup", {"labels": ["a"], "public_url": "https://x.example"})
        tok = cfg["token"]
        res = [rig.submit(tok, [jpeg()], {"labels": ["a"]})[0] for _ in range(5)]
        check("rate limit: 3 per minute then 429", res == [200, 200, 200, 429, 429], str(res))

        st, hdr, raw = rig.http("GET", "/")
        check("page: served as html", st == 200 and hdr["content-type"].startswith("text/html") and b"VisionIST phone" in raw)
        csp = hdr.get("content-security-policy", "")
        check("page: CSP forbids third-party and inline script",
              "default-src 'none'" in csp and "script-src 'self'" in csp and "unsafe-inline" not in csp
              and "connect-src 'self' blob:" in csp
              and "unsafe-eval" not in csp)
        check("page: camera/geolocation allowed for self only",
              "camera=(self)" in hdr["permissions-policy"] and "geolocation=(self)" in hdr["permissions-policy"])
        check("page: nosniff, no-referrer, frame-ancestors none",
              hdr["x-content-type-options"] == "nosniff" and hdr["referrer-policy"] == "no-referrer"
              and "frame-ancestors 'none'" in csp)
        check("page: no HSTS over plain http without a proxy", "strict-transport-security" not in hdr)
        check("page: not cached (html)", hdr.get("cache-control") == "no-cache")
        st, hdr, raw = rig.http("GET", "/api/config", tok)
        check("api: no-store", hdr.get("cache-control") == "no-store")
        st, hdr, raw = rig.http("GET", "/assets/app.js")
        check("asset: app.js served as javascript", st == 200 and hdr["content-type"].startswith("text/javascript"))
        st, hdr, raw = rig.http("GET", "/assets/style.css")
        check("asset: style.css served as css", st == 200 and hdr["content-type"].startswith("text/css"))
        for path in ("/assets/..%2fphone_store.py", "/assets/%2e%2e/phone_store.py", "/assets/../phone_store.py",
                     "/assets/.hidden", "/assets/index.html", "/assets/vendor/", "/assets/nope.js",
                     "/assets/%00", "/assets/..\\phone_store.py", "/phone_store.py"):
            check(f"path traversal / hidden: {path}", rig.http("GET", path)[0] in (400, 404))
        check("asset: viewer bundle absent in a source checkout -> 404 (page falls back)",
              rig.http("GET", "/assets/vendor/viewer.js")[0] in (200, 404))
        check("unknown route 404", rig.http("GET", "/nothing")[0] == 404)
        st, _ = rig.jget("/api/feed?after=abc", token=tok)
        check("feed: bad numbers -> 400", st == 400)
    finally:
        rig.close()


# ====================================================== 5. store: limits, TTL
def test_store():
    print("\n-- store: eviction, `dropped`, TTL, byte cap")
    rig = Rig(max_queue=3, rate_per_min=10000)
    try:
        cfg, _ = setup(rig, labels=["a"])
        tok = cfg["token"]
        for i in range(5):
            assert rig.submit(tok, [jpeg(i)], {"labels": ["a"]})[0] == 200
        c, f = rig.call("get")
        check("max_queue keeps the newest 3", [m["seq"] for m in f["meta"]] == [3, 4, 5])
        check("dropped counts what the reader missed", c["dropped"] == 2, str(c["dropped"]))
        c, f = rig.call("get", {"after": 3})
        check("a reader who is up to date sees dropped 0", c["dropped"] == 0 and c["count"] == 2)
        rig.call("get", {"pop": True})
        c, f = rig.call("get", {"after": 5})
        check("popped submissions are not 'dropped'", c["dropped"] == 0)
    finally:
        rig.close()

    s = base_settings(max_store_mb=1, max_queue=100, max_submit_mb=1)
    store = Store(s)
    sess, _ = store.setup("s", {"labels": []})
    for i in range(4):
        store.add_submission(sess, [jpeg(i, 400 * 1024)], ["image/jpeg"], [], "", None, {}, None)
    st = store.status(sess)
    check("MAX_STORE_MB evicts the oldest first",
          st["queue"]["oldest_seq"] >= 2 and st["store_bytes"] <= MB, str(st["queue"]))
    store.add_card(sess, "t", "", None, jpeg(1, 300 * 1024), "image/jpeg", None)
    check("cards count toward the cap too", store.status(sess)["store_bytes"] <= MB)
    try:
        store.add_submission(sess, [jpeg(1, 2 * MB)], ["image/jpeg"], [], "", None, {}, None)
        check("one submission over the per-submission cap is refused", False)
    except ValueError:
        check("one submission over the per-submission cap is refused", True)

    now = [1000.0]
    s = base_settings(ttl_s=60)
    store = Store(s, clock=lambda: now[0])
    sess, _ = store.setup("s", {"labels": []})
    store.add_submission(sess, [jpeg()], ["image/jpeg"], [], "", None, {}, None)
    store.add_card(sess, "t", "x", None, None, None, None)
    now[0] += 30
    store.add_submission(sess, [jpeg()], ["image/jpeg"], [], "", None, {}, None)
    now[0] += 40
    check("TTL expires only the old ones", store.sweep() == 2 and len(sess.queue) == 1 and not sess.cards)
    subs, info = store.fetch(sess, after=0)
    check("TTL-expired submissions are reported as dropped", info["dropped"] == 1 and len(subs) == 1)

    store = Store(base_settings(max_sessions=2))
    store.setup("a", {}); store.setup("b", {})
    try:
        store.setup("c", {})
        check("MAX_SESSIONS enforced", False)
    except ValueError:
        check("MAX_SESSIONS enforced", True)
    store.reset("a"); store.setup("c", {})
    check("a freed slot can be used", store.session_ids() == ["b", "c"])

    # a reader waiting on a session that is reset must not hang forever
    store = Store(base_settings())
    sess, _ = store.setup("w", {})
    result = []
    t = threading.Thread(target=lambda: result.append(_try_fetch(store, sess)))
    t.start()
    time.sleep(0.3)
    store.reset("w")
    t.join(3)
    check("a waiting get is released by reset (no hung worker)", not t.is_alive() and result == ["reset"])

    # a cancelled RPC frees its worker
    store = Store(base_settings())
    sess, _ = store.setup("c", {})
    stop = threading.Event()
    done = []
    t = threading.Thread(target=lambda: done.append(store.fetch(sess, wait_s=30, should_stop=stop.is_set)))
    t.start()
    time.sleep(0.3)
    stop.set()
    t.join(3)
    check("a cancelled get stops waiting", not t.is_alive() and done and done[0][0] == [])


def _try_fetch(store, sess):
    try:
        store.fetch(sess, wait_s=20)
        return "returned"
    except ValueError:
        return "reset"


# ===================================================== 6. deployment modes
def test_deployment():
    print("\n-- deployment: behind a proxy, and direct TLS")
    # behind a proxy that does NOT strip its prefix
    rig = Rig(base_path="/phone", allowed_hosts=("lab.example.org",), trust_proxy=1,
              auth_fail_limit=2)
    try:
        H = {"Host": "lab.example.org"}
        cfg, _ = rig.call("setup", {"labels": ["a"], "public_url": "https://lab.example.org/phone"})
        tok = cfg["token"]
        check("proxy: setup returns the PUBLIC url, prefix included",
              cfg["url"] == f"https://lab.example.org/phone/#t={tok}")
        check("proxy: path is the in-box path with BASE_PATH", cfg["path"] == f"/phone/#t={tok}")
        st, hdr, _ = rig.http("GET", "/phone", headers=H)
        check("proxy: bare prefix redirects to the slash form", st == 308 and hdr["location"] == "/phone/")
        st, hdr, raw = rig.http("GET", "/phone/", headers=H)
        check("proxy: page under the prefix", st == 200 and b"VisionIST phone" in raw)
        check("proxy: guard headers present once (CSP)", "content-security-policy" in hdr)
        check("proxy: assets under the prefix", rig.http("GET", "/phone/assets/app.js", headers=H)[0] == 200)
        check("proxy: healthz under the prefix", rig.http("GET", "/phone/healthz", headers=H)[0] == 200)
        check("proxy: nothing outside the prefix", rig.http("GET", "/healthz", headers=H)[0] == 404)
        st, body = rig.jget("/phone/api/config", token=tok, headers=H)
        check("proxy: api under the prefix with the token", st == 200 and body["session"] == "default")
        st, r = rig.submit(tok, [jpeg()], {"labels": ["a"]}, headers=H) if False else (None, None)
        body_, ctype = multipart([("image", "a.jpg", "image/jpeg", jpeg()),
                                  ("meta", None, "application/json", b'{"labels":["a"]}')])
        st, _, raw = rig.http("POST", "/phone/api/submit", tok, {**H, "Content-Type": ctype}, body_)
        check("proxy: upload under the prefix", st == 200, raw[:60].decode())
        check("proxy: unknown Host header refused (421)", rig.http("GET", "/phone/", headers={"Host": "evil.example"})[0] == 421)
        check("proxy: Host with a port is matched on the name", rig.http("GET", "/phone/healthz", headers={"Host": "lab.example.org:8443"})[0] == 200)
        st, hdr, _ = rig.http("GET", "/phone/", headers={**H, "X-Forwarded-Proto": "https"})
        check("proxy: HSTS when the proxy says https", "strict-transport-security" in hdr)

        # lockout is per client address as the PROXY reports it
        bad = lambda xff: rig.http("GET", "/phone/api/config", "w" * 32, {**H, "X-Forwarded-For": xff})[0]
        r1 = [bad("203.0.113.5"), bad("203.0.113.5"), bad("203.0.113.5")]
        check("proxy: lockout keys on X-Forwarded-For (last hop)", r1 == [401, 401, 429], str(r1))
        check("proxy: another client is not locked out", bad("198.51.100.7") == 401)
        check("proxy: a spoofed earlier hop does not dodge it", bad("1.2.3.4, 203.0.113.5") == 429)
    finally:
        rig.close()

    # NOT behind a proxy: X-Forwarded-For must be ignored
    rig = Rig(auth_fail_limit=2)
    try:
        bad = lambda xff: rig.http("GET", "/api/config", "w" * 32, {"X-Forwarded-For": xff})[0]
        r = [bad("1.1.1.1"), bad("2.2.2.2"), bad("3.3.3.3")]
        check("direct: spoofed X-Forwarded-For does not evade the lockout", r == [401, 401, 429], str(r))
    finally:
        rig.close()

    # direct TLS, self-signed
    tmp = tempfile.mkdtemp(prefix="phone_tls_")
    rig = Rig(tls_self_signed=True, tls_dir=tmp, tls_san=("localhost", "127.0.0.1"))
    try:
        cfg, _ = rig.call("setup", {"labels": ["a"], "public_url": "https://127.0.0.1:9080"})
        st, hdr, raw = rig.http("GET", "/healthz")
        check("tls: serves https", st == 200 and rig.tls)
        check("tls: HSTS sent", "strict-transport-security" in hdr)
        st, _, _ = rig.http("GET", "/api/config", cfg["token"])
        check("tls: api works with the token", st == 200)
        pem = pathlib.Path(tmp, "cert.pem").read_bytes()
        from cryptography import x509
        cert = x509.load_pem_x509_certificate(pem)
        sans = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        check("tls: certificate has the requested names",
              "localhost" in sans.get_values_for_type(x509.DNSName)
              and any(str(i) == "127.0.0.1" for i in sans.get_values_for_type(x509.IPAddress)))
        check("tls: key file is private", oct(pathlib.Path(tmp, "key.pem").stat().st_mode & 0o777) == "0o600")
        try:
            plain = http.client.HTTPConnection("127.0.0.1", rig.port, timeout=5)
            plain.request("GET", "/healthz"); plain.getresponse().read()
            check("tls: plain http to the TLS port is not served", False)
        except Exception:
            check("tls: plain http to the TLS port is not served", True)
    finally:
        rig.close()
    s1 = Settings.from_env({"TLS_CERT": "/x"}, message_cap=-1)
    check("settings: TLS_CERT without TLS_KEY falls back to plain http", not s1.tls)
    s2 = Settings.from_env({"BASE_PATH": "phone/", "PUBLIC_URL": "https://a.b/phone/", "TRUST_PROXY": "1",
                            "ALLOWED_HOSTS": "A.b, c.d", "MAX_QUEUE": "nope"}, message_cap=-1)
    check("settings: BASE_PATH, PUBLIC_URL, hosts normalized",
          s2.base_path == "/phone" and s2.public_url == "https://a.b/phone"
          and s2.allowed_hosts == ("a.b", "c.d") and s2.trust_proxy == 1 and s2.max_queue == 200)
    try:
        Settings.from_env({"BASE_PATH": "/a/../b"}, message_cap=-1)
        check("settings: BASE_PATH with .. refused", False)
    except ValueError:
        check("settings: BASE_PATH with .. refused", True)
    check("settings: submission cap is 80% of the gRPC message cap",
          Settings.from_env({}, message_cap=10 * MB).max_submit_bytes == int(8 * MB))


# =============================================================== 7. real gRPC
def test_over_grpc():
    print("\n-- over a real gRPC channel (transport, reflection-free)")
    rig = Rig()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=serve.server_options())
    pipeline_pb2_grpc.add_PipelineServiceServicer_to_server(rig.servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        stub = pipeline_pb2_grpc.PipelineServiceStub(grpc.insecure_channel(f"127.0.0.1:{port}"))

        def call(command, params=None, data=None):
            env = pipeline_pb2.Envelope(config_json=json.dumps({"phone": {"command": command, "parameters": params or {}}}))
            for k, v in (data or {}).items():
                env.data[k].CopyFrom(wrap_value(v))
            return unpack(stub.Process(env, timeout=30))

        cfg, f = call("setup", {"labels": ["a"], "public_url": "https://p.example"})
        check("grpc: setup", cfg["status"] == "done" and f["qr"][:4] == b"\x89PNG")
        rig.submit(cfg["token"], [jpeg(3)], {"labels": ["a"]})
        threading.Timer(0.3, lambda: rig.submit(cfg["token"], [jpeg(4)], {"labels": ["a"]})).start()
        cfg2, f2 = call("get", {"wait_s": 5, "after": 1})
        check("grpc: long-poll get returns the second photo", cfg2["count"] == 1 and f2["images"][0] == jpeg(4))
        cfg3, f3 = call("get", {"max_items": 64})
        check("grpc: images arrive as a BytesList", cfg3["count"] == 2 and len(f3["images"]) == 2)

        # a client that cancels a long poll frees the server worker
        t0 = time.time()
        env = pipeline_pb2.Envelope(config_json=json.dumps(
            {"phone": {"command": "get", "parameters": {"wait_s": 20, "after": 99}}}))
        fut = stub.Process.future(env, timeout=30)
        time.sleep(0.5)
        fut.cancel()
        time.sleep(1.2)
        c, _ = call("status")
        check("grpc: server still answers after a cancelled long poll", c["status"] == "done" and time.time() - t0 < 10)
    finally:
        server.stop(0)
        rig.close()


def main():
    t0 = time.time()
    for fn in (test_grpc, test_roundtrip, test_upload_validation, test_security,
               test_store, test_deployment, test_over_grpc):
        try:
            fn()
        except Exception as e:                              # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(f"{fn.__name__} ran without an exception", False, repr(e)[:60])
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed in {time.time() - t0:.1f}s")
    for name in FAIL:
        print("  FAILED:", name)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
