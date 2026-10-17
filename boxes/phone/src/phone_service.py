"""phone box - a phone (or any browser) as a capture-and-annotate terminal.

The box has two faces on one in-memory store:

* a **web page** (``phone_web.py``) a phone opens: live camera, label buttons,
  a send button, and a feed where results come back;
* the standard **gRPC** ``Process`` call, which is how a program uses it:
  ``setup`` the page, poll ``get`` for what was captured, ``show`` results on
  the phone.

Nothing in the box calls another box: it stores and forwards. The program that
polls it decides what to run on the images.

The shared-envelope rules, as every box follows them:

* ONE RPC, ``Process``; the work is in ``config_json`` under ``"phone"``:
  ``{"phone": {"command": ..., "session_id": ..., "parameters": {...}}}``.
* The reply is namespaced the same way, with ``status`` done | empty_request |
  error, and ``encoding`` naming the codec of every bytes field.
* ``reset`` is accepted (see below). Port 8061, overridable with ``PORT``;
  server limits from ``serve``.
* ``session_id`` sits at the TOP of the section (like yolo and tapnext),
  default ``"default"``. One session = one pairing of a program with phones.

Commands: ``setup`` · ``get`` (the default) · ``show`` · ``status`` · ``reset``.
See README.md for the request and reply shapes.
"""

import concurrent.futures as futures
import io
import json
import logging
import os
import re
import sys
import threading
import time

sys.path.append("./protos")
import pipeline_pb2 as folder_wd_pb2  # noqa: E402
import pipeline_pb2_grpc as folder_wd_pb2_grpc  # noqa: E402
from aux import wrap_value, unwrap_value  # noqa: E402
import serve  # noqa: E402

from phone_settings import MB, Settings  # noqa: E402
from phone_store import Store, StoreError, is_glb, sniff_image  # noqa: E402

_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_PORT_ENV_VAR = "PORT"

#: The config section this box answers to. Must equal `key` in box.yaml, and
#: this file must be named <key>_service.py - the Dockerfile resolves it.
BOX_KEY = "phone"

COMMANDS = ("setup", "get", "show", "status", "reset")
SENSOR_GROUPS = ("location", "orientation", "motion")
_SESSION_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_URL_RE = re.compile(r"^https?://[^\s#?]+$")
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]")

MAX_TEXT_CHARS = 20000
MAX_TITLE_CHARS = 200

_CONFIG_DEFAULTS = {
    "title": "VisionIST phone",
    "labels": [],
    "mode": "single",
    "burst_max": 8,
    "camera": "environment",
    "max_side": 1600,
    "jpeg_quality": 0.9,
    "sensors": [],
    "note": True,
    "custom_labels": False,
    "require_label": False,
}


class _BadRequest(Exception):
    pass


def _done(extra, data=None, encoding=None):
    section = {"status": "done", **extra}
    if encoding:
        section["encoding"] = encoding
    return folder_wd_pb2.Envelope(
        config_json=json.dumps({BOX_KEY: section}), data=data or {})


def _empty_request(why=None):
    section = {"status": "empty_request"}
    if why:
        section["error"] = why
    return folder_wd_pb2.Envelope(config_json=json.dumps({BOX_KEY: section}))


def _error(message):
    return folder_wd_pb2.Envelope(
        config_json=json.dumps({BOX_KEY: {"status": "error", "error": message}}))


# ------------------------------------------------------- parameter parsing
def _p_int(params, name, default, lo, hi):
    value = params.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or int(value) != value:
        raise _BadRequest(f"parameters.{name} must be an integer")
    value = int(value)
    if not lo <= value <= hi:
        raise _BadRequest(f"parameters.{name} must be between {lo} and {hi}")
    return value


def _p_float(params, name, default, lo, hi):
    value = params.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _BadRequest(f"parameters.{name} must be a number")
    if not lo <= float(value) <= hi:
        raise _BadRequest(f"parameters.{name} must be between {lo} and {hi}")
    return float(value)


def _p_bool(params, name, default):
    value = params.get(name, default)
    if isinstance(value, bool):
        return value
    if value in (0, 1) and isinstance(value, (int, float)):    # MATLAB/Octave send 0/1
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"                 # a form's select sends text
    raise _BadRequest(f"parameters.{name} must be true or false")


def _p_enum(params, name, default, choices):
    value = params.get(name, default)
    if value not in choices:
        raise _BadRequest(f"parameters.{name} must be one of {list(choices)}")
    return value


def _p_str(params, name, default, limit):
    value = params.get(name, default)
    if not isinstance(value, str):
        raise _BadRequest(f"parameters.{name} must be a string")
    return value.strip()[:limit]


def _p_str_list(params, name, default, max_items, max_chars, unique=True):
    value = params.get(name, default)
    if isinstance(value, str):                    # one string = a list of one
        value = [value]
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise _BadRequest(f"parameters.{name} must be a list of strings")
    out = []
    for item in value:
        item = " ".join(item.split())
        if not item:
            continue
        if len(item) > max_chars:
            raise _BadRequest(f"parameters.{name}: {item[:20]!r}... is over "
                              f"{max_chars} characters")
        if not unique or item not in out:
            out.append(item)
    if len(out) > max_items:
        raise _BadRequest(f"parameters.{name} takes at most {max_items} entries")
    return out


def normalize_config(params, settings):
    """The page's configuration from ``setup`` parameters.

    A ``setup`` call states the WHOLE config: a key left out returns to its
    default, so the page always matches what the last call said.
    """
    sensors = _p_str_list(params, "sensors", [], len(SENSOR_GROUPS), 16)
    for s in sensors:
        if s not in SENSOR_GROUPS:
            raise _BadRequest(f"parameters.sensors: {s!r} is not one of {list(SENSOR_GROUPS)}")
    return {
        "title": _p_str(params, "title", _CONFIG_DEFAULTS["title"], 80) or _CONFIG_DEFAULTS["title"],
        "labels": _p_str_list(params, "labels", [], 64, 64),
        "mode": _p_enum(params, "mode", "single", ("single", "burst")),
        "burst_max": _p_int(params, "burst_max", 8, 1, settings.max_images),
        "camera": _p_enum(params, "camera", "environment", ("environment", "user")),
        "max_side": _p_int(params, "max_side", 1600, 0, 8192),
        "jpeg_quality": _p_float(params, "jpeg_quality", 0.9, 0.3, 1.0),
        "sensors": sensors,
        "note": _p_bool(params, "note", True),
        "custom_labels": _p_bool(params, "custom_labels", False),
        "require_label": _p_bool(params, "require_label", False),
    }


def make_qr_png(text, scale=8):
    import segno
    buf = io.BytesIO()
    segno.make(text, error="m").save(buf, kind="png", scale=scale, border=3)
    return buf.getvalue()


def pairing_url(base, token):
    """The link a phone opens. The token rides in the FRAGMENT: a browser never
    sends it to the server, a proxy never logs it, a Referer never carries it."""
    return f"{base.rstrip('/')}/#t={token}"


# ----------------------------------------------------------------- service
class PipelineService(folder_wd_pb2_grpc.PipelineServiceServicer):

    def __init__(self, store=None, settings=None):
        self.settings = settings or Settings.from_env()
        self.store = store or Store(self.settings)

    # ---------------------------------------------------------- dispatch
    def Process(self, request, context):
        try:
            if not request.config_json:
                return _error("No config JSON")
            config = json.loads(request.config_json)
            if not isinstance(config, dict) or not isinstance(config.get(BOX_KEY), dict):
                return _error(f"config section {BOX_KEY!r} missing or not an object")
            section = config[BOX_KEY]
            command = section.get("command") or "get"
            parameters = section.get("parameters", {})
            if not isinstance(parameters, dict):
                return _error("parameters must be an object")
            if command not in COMMANDS:
                return _error(f"unknown command {command!r} (expected one of "
                              f"{', '.join(COMMANDS)})")
            sid = section.get("session_id")
            if sid is not None and not (isinstance(sid, str) and _SESSION_RE.match(sid)):
                return _error("session_id must be 1-64 characters of A-Z a-z 0-9 . _ -")
            handler = getattr(self, "_cmd_" + command)
            return handler(sid, parameters, request, context)
        except _BadRequest as e:
            return _error(str(e))
        except StoreError as e:
            return _error(str(e))
        except Exception as e:                              # noqa: BLE001
            logging.exception("Error in Process")
            return _error(str(e))

    def _need_session(self, sid):
        sess = self.store.session(sid or "default")
        if sess is None:
            raise _BadRequest(f"no session {sid or 'default'!r}: call setup first "
                              f"(it was reset, or never created)")
        return sess

    # ------------------------------------------------------------- setup
    def _cmd_setup(self, sid, params, request, context):
        sid = sid or "default"
        cfg = normalize_config(params, self.settings)
        rotate = _p_bool(params, "rotate", False)
        want_qr = _p_bool(params, "qr", True)
        qr_scale = _p_int(params, "qr_scale", 8, 2, 32)
        public = params.get("public_url", "")
        if not isinstance(public, str):
            raise _BadRequest("parameters.public_url must be a string")
        public = public.strip().rstrip("/")
        if public and not _URL_RE.match(public):
            raise _BadRequest("parameters.public_url must be an http(s) URL "
                              "without a query or fragment")
        sess, created = self.store.setup(sid, cfg, rotate=rotate)

        warnings = []
        base, source = (public, "parameters") if public else \
            (self.settings.public_url, "PUBLIC_URL") if self.settings.public_url \
            else ("", None)
        url = pairing_url(base, sess.token) if base else None
        https = None
        if url:
            https = base.lower().startswith("https://")
            host = base.split("//", 1)[1].split("/", 1)[0].rsplit(":", 1)[0].lower()
            if not https and host not in _LOCAL_HOSTS:
                warnings.append("this URL is plain http: a phone browser only offers its "
                                "live camera on https (or localhost), so the page falls "
                                "back to the phone's own camera app. Put the box behind "
                                "https, or set TLS_CERT/TLS_KEY.")
        else:
            warnings.append("no public URL known: pass parameters.public_url or set the "
                            "PUBLIC_URL environment variable, then open `path` on it. "
                            "No QR code was made.")
        data, encoding = {}, {}
        if url and want_qr:
            try:
                data["qr"] = wrap_value(make_qr_png(url, qr_scale))
                encoding["qr"] = "identity"
            except Exception as e:                          # noqa: BLE001
                warnings.append(f"QR code not made: {e}")
        return _done({
            "action": "setup", "session_id": sid, "created": created,
            "url": url, "url_source": source, "path": f"{self.settings.base_path}/#t={sess.token}",
            "token": sess.token, "https": https,
            "config": cfg, "config_version": sess.config_version,
            "web": {"port": self.settings.web_port, "tls": self.settings.tls,
                    "base_path": self.settings.base_path},
            "warnings": warnings}, data=data, encoding=encoding or None)

    # --------------------------------------------------------------- get
    def _cmd_get(self, sid, params, request, context):
        sess = self._need_session(sid)
        after = _p_int(params, "after", 0, 0, 2 ** 53)
        max_items = _p_int(params, "max_items", 8, 1, 64)
        wait_s = _p_float(params, "wait_s", 0.0, 0.0, float(self.settings.max_wait_s))
        latest = _p_bool(params, "latest", False)
        pop = _p_bool(params, "pop", False)
        cap = self.settings.message_cap
        default_mb = 32 if cap <= 0 else max(1, int(cap * 0.8) // MB)
        max_bytes = int(_p_float(params, "max_mb", float(default_mb), 0.1, 4096.0) * MB)

        subs, info = self.store.fetch(
            sess, after=after, max_items=max_items, max_bytes=max_bytes,
            latest=latest, pop=pop, wait_s=wait_s,
            should_stop=lambda: not context.is_active())
        images, meta = [], []
        for s in subs:
            meta.append(s.meta(first_image=len(images)))
            images.extend(s.images)
        extra = {"action": "get", "session_id": sess.id, "count": len(subs),
                 "images": len(images), **info}
        data = {"meta": wrap_value(json.dumps(meta).encode())}
        if images:
            data["images"] = wrap_value(images)
        return _done(extra, data=data, encoding={"images": "identity", "meta": "json"})

    # -------------------------------------------------------------- show
    def _cmd_show(self, sid, params, request, context):
        sess = self._need_session(sid)
        title = _p_str(params, "title", "", MAX_TITLE_CHARS)
        text = params.get("text")
        if text is None and "text" in request.data:
            text = unwrap_value(request.data["text"])
        if text is not None:
            if isinstance(text, bytes):
                text = text.decode("utf-8", "replace")
            if not isinstance(text, str):
                raise _BadRequest("text must be a string")
            if len(text) > MAX_TEXT_CHARS:
                raise _BadRequest(f"text is over {MAX_TEXT_CHARS} characters")
        ref_seq = params.get("ref_seq")
        if ref_seq is not None:
            if isinstance(ref_seq, bool) or not isinstance(ref_seq, (int, float)) \
                    or int(ref_seq) != ref_seq or ref_seq < 0:
                raise _BadRequest("parameters.ref_seq must be a non-negative integer")
            ref_seq = int(ref_seq)
        replace = _p_bool(params, "replace", False)

        image = image_mime = glb = None
        if "image" in request.data:
            image = unwrap_value(request.data["image"])
            if isinstance(image, list):
                image = image[0] if image else None
        if image is not None:
            image = bytes(image)
            image_mime = sniff_image(image)
            if image_mime is None:
                raise _BadRequest("image must be JPEG, PNG or WebP bytes")
        if "glb" in request.data:
            glb = unwrap_value(request.data["glb"])
            if glb is not None:
                glb = bytes(glb)
                if not is_glb(glb):
                    raise _BadRequest("glb must be a binary glTF (it starts with 'glTF')")
        if not (text or image or glb):
            return _empty_request("show needs at least one of: text, image, glb")
        for blob in (image, glb):
            if blob is not None and len(blob) > self.settings.max_store_mb * MB // 4:
                raise _BadRequest("that result is too large for this box's MAX_STORE_MB")
        card = self.store.add_card(sess, title, text or "", ref_seq, image, image_mime,
                                   glb, replace=replace)
        return _done({"action": "show", "session_id": sess.id, "card_id": card})

    # ------------------------------------------------------------ status
    def _cmd_status(self, sid, params, request, context):
        sessions = self.store.session_ids()
        sess = self.store.session(sid or "default")
        extra = {"action": "status", "session_id": sid or "default",
                 "exists": sess is not None, "sessions": sessions,
                 "web": {"port": self.settings.web_port, "tls": self.settings.tls,
                         "base_path": self.settings.base_path,
                         "public_url": self.settings.public_url or None}}
        if sess is not None:
            extra.update(self.store.status(sess))
        return _done(extra)

    # ------------------------------------------------------------- reset
    def _cmd_reset(self, sid, params, request, context):
        n = self.store.reset(sid)
        return _done({"action": "reset", "session_id": sid, "sessions_removed": n})


# --------------------------------------------------------------- bootstrap
def get_port():
    try:
        port = int(os.getenv(_PORT_ENV_VAR, _PORT_DEFAULT))
        return port if port > 0 else None
    except ValueError:
        logging.exception("Invalid port value")
        return None


def run_server(server, sweeper_stop):
    port = get_port()
    if not port:
        return
    target = f"[::]:{port}"
    server.add_insecure_port(target)
    server.start()
    logging.info(f"Server started at {target}")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        sweeper_stop.set()
        server.stop(0)


def start_sweeper(store, interval=30):
    stop = threading.Event()

    def loop():
        while not stop.wait(interval):
            try:
                n = store.sweep()
                if n:
                    logging.info("expired %d item(s) past TTL_S", n)
            except Exception:                               # noqa: BLE001
                logging.exception("sweep failed")

    threading.Thread(target=loop, name="phone-sweeper", daemon=True).start()
    return stop


if __name__ == "__main__":
    import grpc
    import grpc_reflection.v1alpha.reflection as grpc_reflection

    from phone_web import WebServer

    logging.basicConfig(
        format="[ %(levelname)s ] %(asctime)s (%(module)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    settings = Settings.from_env()
    store = Store(settings)
    web = WebServer(store, settings).start()
    logging.info("%s", serve.describe(8))
    if settings.tls or settings.public_url.startswith("https://"):
        logging.warning("the gRPC port has NO authentication. On a public host publish "
                        "only the web port (%s) and keep the gRPC port on a private "
                        "network.", web.port)
    sweeper_stop = start_sweeper(store)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=serve.max_workers(8)),
        options=serve.server_options(),
        maximum_concurrent_rpcs=serve.max_concurrent_rpcs(serve.max_workers(8)))
    folder_wd_pb2_grpc.add_PipelineServiceServicer_to_server(
        PipelineService(store, settings), server)

    grpc_reflection.enable_server_reflection(
        (folder_wd_pb2.DESCRIPTOR.services_by_name["PipelineService"].full_name,
         grpc_reflection.SERVICE_NAME), server)

    run_server(server, sweeper_stop)
