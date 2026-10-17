"""The HTTP side of the phone box: the page, the upload endpoint, the results feed.

Routes (all relative to ``BASE_PATH``, if one is set):

    GET  /                  the capture page
    GET  /assets/<file>     its script, stylesheet and (built) 3D viewer
    GET  /healthz           liveness for a proxy / orchestrator, no token
    GET  /api/config        the session's config + limits          [token]
    POST /api/submit        multipart: image parts + one meta part [token]
    POST /api/frame         one streamed frame: image + meta (+ depth); it
                            replaces the session's "latest" slot    [token]
    GET  /api/feed          results pushed with `show`; long-poll  [token]
    GET  /api/media/<id>/image|glb                                 [token]

The token travels in an ``Authorization: Bearer`` header, never in a URL: the
pairing link carries it in the fragment (``#t=...``), which a browser does not
send to the server and does not leak in a Referer. Everything is plain HTTP
request/response (no WebSocket), so any reverse proxy passes it unchanged.
"""

import asyncio
import collections
import datetime
import ipaddress
import json
import logging
import math
import mimetypes
import os
import pathlib
import socket
import ssl
import threading
import time

from aiohttp import web

from phone_settings import MB
from phone_store import StoreError, sniff_image

WEB_DIR = pathlib.Path(os.environ.get("PHONE_WEB_DIR")
                       or pathlib.Path(__file__).resolve().parent / "web")

MAX_META_BYTES = 16 * 1024
MAX_LABELS = 32
MAX_LABEL_CHARS = 64
MAX_NOTE_CHARS = 2000
SENSOR_GROUPS = ("location", "orientation", "motion", "lidar")
DEPTH_DTYPES = {"float32": 4, "uint16": 2}
DEPTH_UNITS = ("m", "mm")
MAX_DEPTH_SIDE = 4096

_CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; "
        "img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self' blob:; "
        "worker-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'; "
        "form-action 'none'")
_PERMISSIONS = ("camera=(self), geolocation=(self), accelerometer=(self), "
                "gyroscope=(self), magnetometer=(self), microphone=(), payment=()")


class ApiError(Exception):
    def __init__(self, status, message, headers=None):
        super().__init__(message)
        self.status, self.message, self.headers = status, message, headers or {}


# ------------------------------------------------------------ validation
def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _clean_sensor_reading(raw, allowed_groups):
    """One image's sensors: plain numbers under the groups the session asked for."""
    if not isinstance(raw, dict):
        return None
    out = {}
    for group in SENSOR_GROUPS:
        values = raw.get(group)
        if group not in allowed_groups or not isinstance(values, dict):
            continue
        kept = {str(k)[:24]: v for k, v in list(values.items())[:16] if _finite(v)}
        if kept:
            out[group] = kept
    return out or None


def _clean_sensors(raw, allowed_groups, n_images):
    """``sensors`` is a list aligned with the images (one reading each, or null);
    a single object is accepted for a one-image submission. Returns a list of
    length ``n_images`` or None when no image has a reading."""
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return None
    readings = [_clean_sensor_reading(r, allowed_groups) for r in raw[:n_images]]
    readings += [None] * (n_images - len(readings))
    return readings if any(readings) else None


def load_meta(raw):
    try:
        meta = json.loads(raw.decode("utf-8")) if raw else {}
    except (UnicodeDecodeError, ValueError):
        raise ApiError(400, "meta is not valid JSON")
    if not isinstance(meta, dict):
        raise ApiError(400, "meta must be an object")
    return meta


def clean_size(raw):
    """``[width, height]`` of the streamed frame as the phone encoded it, or None."""
    if (isinstance(raw, list) and len(raw) == 2
            and all(isinstance(x, int) and not isinstance(x, bool) and 0 < x <= 65536
                    for x in raw)):
        return list(raw)
    return None


def clean_depth(raw, nbytes):
    """Describe a depth map that came as its own part (e.g. LiDAR from a native
    ARKit client). The box does not decode it, but it checks that the declared
    shape matches the byte count so a consumer can ``frombuffer`` it blindly.

    ``{width, height, dtype: float32|uint16, unit: m|mm}`` plus optional
    ``intrinsics`` (fx, fy, cx, cy... as plain numbers). Row-major,
    little-endian, no header.
    """
    if not isinstance(raw, dict):
        raise ApiError(400, "meta.depth is required when a depth part is sent")
    dtype, unit = raw.get("dtype"), raw.get("unit", "m")
    w, h = raw.get("width"), raw.get("height")
    if dtype not in DEPTH_DTYPES:
        raise ApiError(400, f"depth.dtype must be one of {sorted(DEPTH_DTYPES)}")
    if unit not in DEPTH_UNITS:
        raise ApiError(400, f"depth.unit must be one of {list(DEPTH_UNITS)}")
    if not all(isinstance(x, int) and not isinstance(x, bool) and 0 < x <= MAX_DEPTH_SIDE
               for x in (w, h)):
        raise ApiError(400, f"depth.width and depth.height must be integers in 1..{MAX_DEPTH_SIDE}")
    if w * h * DEPTH_DTYPES[dtype] != nbytes:
        raise ApiError(400, f"depth part is {nbytes} bytes; {w}x{h} {dtype} needs "
                            f"{w * h * DEPTH_DTYPES[dtype]}")
    info = {"width": w, "height": h, "dtype": dtype, "unit": unit, "bytes": nbytes}
    intr = raw.get("intrinsics")
    if isinstance(intr, dict):
        kept = {str(k)[:16]: v for k, v in list(intr.items())[:12] if _finite(v)}
        if kept:
            info["intrinsics"] = kept
    return info


def clean_meta(raw, config, n_images=1, frame=False):
    """Validate the phone's ``meta`` part against the session config.

    Returns ``(labels, note, sensors, device, client_t)`` or raises ApiError.
    ``frame=True`` (a streamed frame) skips ``require_label``: that rule is about
    a deliberate submission, not about every frame of a stream.
    """
    meta = raw if isinstance(raw, dict) else load_meta(raw)

    labels = meta.get("labels", [])
    if not isinstance(labels, list) or len(labels) > MAX_LABELS:
        raise ApiError(400, f"labels must be a list of at most {MAX_LABELS}")
    cleaned = []
    for item in labels:
        if not isinstance(item, str):
            raise ApiError(400, "labels must be strings")
        item = " ".join(item.split())[:MAX_LABEL_CHARS]
        if item and item not in cleaned:
            cleaned.append(item)
    if not config.get("custom_labels"):
        allowed = set(config.get("labels") or [])
        stray = [x for x in cleaned if x not in allowed]
        if stray:
            raise ApiError(400, f"label not offered by this session: {stray[0]!r}")
    if config.get("require_label") and not cleaned and not frame:
        raise ApiError(400, "this session requires at least one label")

    note = meta.get("note", "")
    if not isinstance(note, str):
        raise ApiError(400, "note must be a string")
    note = note.strip()[:MAX_NOTE_CHARS] if config.get("note", True) else ""

    sensors = _clean_sensors(meta.get("sensors"), set(config.get("sensors") or []),
                             n_images)

    device = meta.get("device") if isinstance(meta.get("device"), dict) else {}
    device = {"id": str(device.get("id") or "")[:64],
              "name": str(device.get("name") or "phone")[:64]}
    client_t = meta.get("client_t")
    client_t = float(client_t) if _finite(client_t) else None
    return cleaned, note, sensors, device, client_t


# ------------------------------------------------------------ rate limits
class Window:
    """Sliding-window counter per key."""

    def __init__(self, limit, seconds):
        self.limit, self.seconds = limit, seconds
        self._hits = collections.defaultdict(collections.deque)

    def _trim(self, key, now):
        dq = self._hits[key]
        while dq and dq[0] <= now - self.seconds:
            dq.popleft()
        if not dq:
            self._hits.pop(key, None)
        return dq

    def blocked(self, key, now):
        return len(self._trim(key, now)) >= self.limit if key in self._hits else False

    def hit(self, key, now):
        """Record a hit; False if the key is over its limit (hit not recorded)."""
        if len(self._hits) > 10000:
            for k in list(self._hits):
                self._trim(k, now)
        self._trim(key, now)          # drops the key when it has emptied...
        dq = self._hits[key]          # ...so take the deque only afterwards
        if len(dq) >= self.limit:
            return False
        dq.append(now)
        return True


def client_ip(request, trust):
    peer = request.transport.get_extra_info("peername") if request.transport else None
    ip = peer[0] if peer else "unknown"
    if trust > 0:
        hops = [h.strip() for h in request.headers.get("X-Forwarded-For", "").split(",")
                if h.strip()]
        if len(hops) >= trust:
            return hops[-trust]
    return ip


# ------------------------------------------------------------------ app
def make_app(store, settings, clock=time.time):
    """The aiohttp application (mounted under BASE_PATH when one is set)."""
    fail_window = Window(settings.auth_fail_limit, 300)
    submit_window = Window(settings.rate_per_min, 60)
    frame_window = Window(settings.frame_rate_per_s, 1)
    index_bytes = None

    def load_index():
        nonlocal index_bytes
        if index_bytes is None:
            index_bytes = (WEB_DIR / "index.html").read_bytes()
        return index_bytes

    @web.middleware
    async def guard(request, handler):
        host = request.headers.get("Host", "").split(":")[0].strip("[]").lower()
        if settings.allowed_hosts and host not in settings.allowed_hosts:
            return web.json_response({"error": "unknown host"}, status=421)
        try:
            response = await handler(request)
        except ApiError as e:
            response = web.json_response({"error": e.message}, status=e.status,
                                         headers=e.headers)
        except web.HTTPRequestEntityTooLarge:
            response = web.json_response({"error": "request too large"}, status=413)
        except web.HTTPException as e:
            if "/api/" in request.path:
                response = web.json_response({"error": e.reason}, status=e.status)
            else:
                raise
        h = response.headers
        h["Content-Security-Policy"] = _CSP
        h["Permissions-Policy"] = _PERMISSIONS
        h["Referrer-Policy"] = "no-referrer"
        h["X-Content-Type-Options"] = "nosniff"
        h["Cross-Origin-Opener-Policy"] = "same-origin"
        if "/api/" in request.path:
            h["Cache-Control"] = "no-store"
        secure = (settings.tls or request.secure or (
            settings.trust_proxy > 0
            and request.headers.get("X-Forwarded-Proto", "").lower() == "https"))
        if secure:
            h["Strict-Transport-Security"] = "max-age=31536000"
        return response

    def authorize(request):
        now = clock()
        ip = client_ip(request, settings.trust_proxy)
        if fail_window.blocked(ip, now):
            raise ApiError(429, "too many wrong tokens; try again later",
                           {"Retry-After": "300"})
        header = request.headers.get("Authorization", "")
        token = header[7:].strip() if header[:7].lower() == "bearer " else ""
        sess = store.find_by_token(token)
        if sess is None:
            fail_window.hit(ip, now)
            raise ApiError(401, "invalid or expired token")
        return sess, ip

    # ----------------------------------------------------------- handlers
    async def index(request):
        return web.Response(body=load_index(), content_type="text/html",
                            charset="utf-8", headers={"Cache-Control": "no-cache"})

    async def asset(request):
        name = request.match_info["name"]
        parts = name.split("/")
        if any(p in ("", ".", "..") or p.startswith(".") or "\\" in p or "\x00" in p
               for p in parts) or name == "index.html":
            raise web.HTTPNotFound()
        path = (WEB_DIR / name).resolve()
        try:
            path.relative_to(WEB_DIR.resolve())
        except ValueError:
            raise web.HTTPNotFound()
        if not path.is_file():
            raise web.HTTPNotFound()
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix == ".js":
            ctype = "text/javascript"
        return web.Response(body=path.read_bytes(), content_type=ctype.split(";")[0],
                            headers={"Cache-Control": "public, max-age=300"})

    async def healthz(request):
        return web.json_response({"ok": True})

    async def config(request):
        sess, _ = authorize(request)
        store.touch_device(sess, request.query.get("device", "")[:64],
                           request.query.get("name", "")[:64])
        return web.json_response({
            "session": sess.id, "config": sess.config,
            "config_version": sess.config_version,
            "limits": {"max_submit_bytes": settings.max_submit_bytes,
                       "max_images": settings.max_images,
                       "max_frame_bytes": settings.max_frame_bytes,
                       "max_wait_s": settings.max_wait_s},
            "server_time": clock()})

    async def submit(request):
        sess, ip = authorize(request)
        if not submit_window.hit(ip, clock()):
            raise ApiError(429, "too many submissions; slow down", {"Retry-After": "30"})
        if not (request.content_type or "").startswith("multipart/"):
            raise ApiError(415, "send multipart/form-data")
        cap = settings.max_submit_bytes
        if request.content_length and request.content_length > cap + 64 * 1024:
            raise ApiError(413, f"submission over {cap // MB} MB")
        cfg = dict(sess.config)
        limit_images = min(cfg.get("burst_max", 1) if cfg.get("mode") == "burst" else 1,
                           settings.max_images)
        images, mimes, raw_meta, total = [], [], b"", 0
        reader = await request.multipart()
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name == "image":
                if len(images) >= limit_images:
                    raise ApiError(400, f"at most {limit_images} image(s) per submission")
                buf = bytearray()
                while True:
                    chunk = await part.read_chunk(65536)
                    if not chunk:
                        break
                    buf += chunk
                    total += len(chunk)
                    if total > cap:
                        raise ApiError(413, f"submission over {cap // MB} MB")
                mime = sniff_image(buf)
                if mime is None:
                    raise ApiError(415, "images must be JPEG, PNG or WebP")
                images.append(bytes(buf))
                mimes.append(mime)
            elif part.name == "meta":
                while True:
                    chunk = await part.read_chunk(4096)
                    if not chunk:
                        break
                    raw_meta += chunk
                    if len(raw_meta) > MAX_META_BYTES:
                        raise ApiError(413, "meta too large")
            else:
                raise ApiError(400, f"unexpected part {part.name!r}")
        if not images:
            raise ApiError(400, "no image in the submission")
        labels, note, sensors, device, client_t = clean_meta(raw_meta, cfg, len(images))
        try:
            seq = store.add_submission(sess, images, mimes, labels, note, sensors,
                                       device, client_t)
        except StoreError as e:
            raise ApiError(413 if "large" in str(e) else 409, str(e))
        store.touch_device(sess, device["id"], device["name"])
        return web.json_response({"ok": True, "seq": seq, "bytes": total,
                                  "images": len(images)})

    async def frame(request):
        sess, ip = authorize(request)
        if not frame_window.hit(ip, clock()):
            raise ApiError(429, "too many frames; lower the stream rate", {"Retry-After": "1"})
        if not (request.content_type or "").startswith("multipart/"):
            raise ApiError(415, "send multipart/form-data")
        cap = settings.max_frame_bytes
        if request.content_length and request.content_length > cap + 64 * 1024:
            raise ApiError(413, f"frame over {cap // MB} MB")
        image = depth = None
        mime, raw_meta, total = None, b"", 0
        reader = await request.multipart()
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name in ("image", "depth"):
                if (image if part.name == "image" else depth) is not None:
                    raise ApiError(400, f"one {part.name} per frame")
                buf = bytearray()
                while True:
                    chunk = await part.read_chunk(65536)
                    if not chunk:
                        break
                    buf += chunk
                    total += len(chunk)
                    if total > cap:
                        raise ApiError(413, f"frame over {cap // MB} MB")
                if part.name == "image":
                    mime = sniff_image(buf)
                    if mime is None:
                        raise ApiError(415, "images must be JPEG, PNG or WebP")
                    image = bytes(buf)
                else:
                    depth = bytes(buf)
            elif part.name == "meta":
                while True:
                    chunk = await part.read_chunk(4096)
                    if not chunk:
                        break
                    raw_meta += chunk
                    if len(raw_meta) > MAX_META_BYTES:
                        raise ApiError(413, "meta too large")
            else:
                raise ApiError(400, f"unexpected part {part.name!r}")
        if image is None:
            raise ApiError(400, "no image in the frame")
        meta = load_meta(raw_meta)
        labels, note, sensors, device, client_t = clean_meta(
            meta, dict(sess.config), 1, frame=True)
        depth_info = clean_depth(meta.get("depth"), len(depth)) if depth else None
        try:
            n = store.add_frame(sess, image, mime, labels, note, sensors, device,
                                client_t, size=clean_size(meta.get("size")),
                                depth=depth, depth_info=depth_info)
        except StoreError as e:
            raise ApiError(413 if "large" in str(e) else 409, str(e))
        store.touch_device(sess, device["id"], device["name"])
        return web.json_response({"ok": True, "n": n, "bytes": total})

    async def feed(request):
        sess, _ = authorize(request)
        q = request.query
        try:
            after = max(0, int(q.get("after", "0")))
            wait = min(max(0.0, float(q.get("wait", "0"))), settings.max_wait_s)
            known = int(q.get("cv", sess.config_version))
        except ValueError:
            raise ApiError(400, "after, wait and cv must be numbers")
        store.touch_device(sess, q.get("device", "")[:64], q.get("name", "")[:64])
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait
        while True:
            cards = store.cards_since(sess, after)
            if (cards or wait <= 0 or loop.time() >= deadline
                    or store.session(sess.id) is not sess
                    or sess.config_version != known):
                break
            await asyncio.sleep(0.1)
        if store.session(sess.id) is not sess:
            raise ApiError(401, "session ended")
        age = store.program_age(sess)
        return web.json_response({
            "cards": cards, "last": cards[-1]["id"] if cards else after,
            "config_version": sess.config_version,
            "program_age_s": None if age is None else round(age, 1)})

    async def media(request):
        sess, _ = authorize(request)
        try:
            card = store.card(sess, int(request.match_info["id"]))
        except ValueError:
            raise ApiError(404, "no such card")
        kind = request.match_info["kind"]
        if card is None or kind not in ("image", "glb"):
            raise ApiError(404, "no such card")
        if kind == "image" and card.image is not None:
            return web.Response(body=card.image, content_type=card.image_mime)
        if kind == "glb" and card.glb is not None:
            return web.Response(body=card.glb, content_type="model/gltf-binary")
        raise ApiError(404, "card has no " + kind)

    body_cap = max(settings.max_submit_bytes, settings.max_frame_bytes) + 256 * 1024
    app = web.Application(client_max_size=body_cap)
    app.router.add_get("/", index)
    app.router.add_get("/assets/{name:.+}", asset)
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/api/config", config)
    app.router.add_post("/api/submit", submit)
    app.router.add_post("/api/frame", frame)
    app.router.add_get("/api/feed", feed)
    app.router.add_get("/api/media/{id}/{kind}", media)

    if not settings.base_path:
        app.middlewares.append(guard)
        return app

    # Served under a prefix the proxy does NOT strip: mount, and send the bare
    # prefix to the trailing-slash form so the page's relative URLs resolve.
    # The guard wraps the outermost application only, so it runs once.
    outer = web.Application(client_max_size=body_cap)

    async def to_slash(request):
        raise web.HTTPPermanentRedirect(settings.base_path + "/")

    outer.router.add_get(settings.base_path, to_slash)
    outer.add_subapp(settings.base_path, app)
    outer.middlewares.append(guard)
    return outer


# ------------------------------------------------------------------ TLS
def make_self_signed(directory, names=()):
    """A throw-away certificate for a LAN deployment. Returns (cert, key) paths.

    Reused on restart while it has more than 30 days left, so a phone that
    accepted it once is not asked again every time the container restarts.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    d = pathlib.Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    cert_path, key_path = d / "cert.pem", d / "key.pem"
    now = datetime.datetime.now(datetime.timezone.utc)
    if cert_path.is_file() and key_path.is_file():
        try:
            old = x509.load_pem_x509_certificate(cert_path.read_bytes())
            if old.not_valid_after_utc - now > datetime.timedelta(days=30):
                return str(cert_path), str(key_path)
        except Exception:                                  # noqa: BLE001
            pass
    sans = []
    for n in list(names) or ["localhost", "127.0.0.1", socket.gethostname()]:
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(n)))
        except ValueError:
            sans.append(x509.DNSName(n))
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "visionist-phone")])
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(sans), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()))
    os.chmod(key_path, 0o600)
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    logging.warning("generated a self-signed certificate (sha256 %s); phones will "
                    "warn about it - use TLS_CERT/TLS_KEY or a proxy for a real one",
                    cert.fingerprint(hashes.SHA256()).hex())
    return str(cert_path), str(key_path)


def ssl_context(settings):
    """The server's SSLContext, or None for plain HTTP."""
    if not settings.tls:
        return None, None, None
    cert, key = settings.tls_cert, settings.tls_key
    if not cert:
        cert, key = make_self_signed(settings.tls_dir, settings.tls_san)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(cert, key)
    return ctx, cert, key


# --------------------------------------------------------------- the server
class WebServer:
    """Runs the aiohttp app on its own thread and event loop, so the gRPC side
    (threads) and the HTTP side (asyncio) share only the Store."""

    def __init__(self, store, settings):
        self.store, self.settings = store, settings
        self.port = None
        self._loop = None
        self._runner = None
        self._thread = None
        self._ready = threading.Event()
        self._error = None

    def start(self, timeout=15):
        self._thread = threading.Thread(target=self._run, name="phone-web", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("web server did not start")
        if self._error:
            raise self._error
        return self

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:                              # noqa: BLE001
            self._error = e
            self._ready.set()
            return
        self._loop.run_forever()

    async def _serve(self):
        s = self.settings
        app = make_app(self.store, s)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        ctx, cert, key = ssl_context(s)
        site = web.TCPSite(self._runner, s.web_host, s.web_port, ssl_context=ctx)
        await site.start()
        self.port = self._runner.addresses[0][1]
        if ctx is not None and s.tls_cert:
            self._loop.create_task(self._reload_certs(ctx, cert, key))
        logging.info("web server on %s:%s (%s)%s", s.web_host, self.port,
                     "https" if ctx else "http",
                     f" under {s.base_path}" if s.base_path else "")
        self._ready.set()

    async def _reload_certs(self, ctx, cert, key):
        """Pick up a renewed certificate (Let's Encrypt) without a restart."""
        def stamp():
            try:
                return os.stat(cert).st_mtime, os.stat(key).st_mtime
            except OSError:
                return None
        seen = stamp()
        while True:
            await asyncio.sleep(3600)
            now = stamp()
            if now and now != seen:
                try:
                    ctx.load_cert_chain(cert, key)
                    seen = now
                    logging.info("reloaded the TLS certificate")
                except Exception:                           # noqa: BLE001
                    logging.exception("could not reload the TLS certificate")

    def stop(self):
        if self._loop is None:
            return
        async def _down():
            if self._runner:
                await self._runner.cleanup()
        try:
            asyncio.run_coroutine_threadsafe(_down(), self._loop).result(10)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
