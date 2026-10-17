"""Environment-driven settings for the phone box, read once at startup.

Nothing here is baked into the image: how the box is exposed (behind a reverse
proxy, or straight on a public server) is decided at launch, like ``PORT`` and
the ``serve`` limits are.

Two deployments are supported, and they differ only in these variables:

**Behind a reverse proxy** (nginx, Caddy, Traefik, a cloud load balancer).
The proxy terminates TLS and the box speaks plain HTTP to it::

    PUBLIC_URL=https://lab.example.org/phone    # what the phone is sent to
    TRUST_PROXY=1                               # believe X-Forwarded-For (1 hop)
    ALLOWED_HOSTS=lab.example.org               # optional, recommended
    BASE_PATH=/phone                            # ONLY if the proxy does not
                                                # strip the prefix it routes on

**Directly on a public server.** The box terminates TLS itself::

    TLS_CERT=/certs/fullchain.pem  TLS_KEY=/certs/privkey.pem
    PUBLIC_URL=https://phone.example.org:9080
    ALLOWED_HOSTS=phone.example.org

(``TLS_SELF_SIGNED=1`` generates a throw-away certificate for a LAN. Phones
will warn about it.)  In both cases pairing is by an unguessable token, uploads
are size- and rate-limited, and nothing is fetched from a third party.

Every variable and its default:

======================  ========  ============================================
``WEB_PORT``            8062      in-container HTTP(S) port
``WEB_HOST``            0.0.0.0   bind address
``PUBLIC_URL``          (none)    external URL of the page root, prefix included
``BASE_PATH``           (none)    serve everything under this path prefix
``TRUST_PROXY``         0         proxy hops whose X-Forwarded-For is believed
``ALLOWED_HOSTS``       (any)     comma list; other Host headers get 421
``TLS_CERT``/``TLS_KEY``          PEM files; both set -> the box serves HTTPS
``TLS_SELF_SIGNED``     0         1 -> generate a certificate at startup
``TLS_SAN``             (auto)    comma list of DNS names / IPs for it
``TLS_DIR``             /tmp/phone-tls   where a generated certificate lives
``MAX_SUBMIT_MB``       24        bytes accepted per submission (all images)
``MAX_IMAGES``          12        images per submission, at most
``MAX_QUEUE``           200       submissions kept per session
``MAX_STORE_MB``        256       everything held in memory, all sessions
``MAX_SESSIONS``        32
``TTL_S``               3600      an unread submission or result expires
``RATE_PER_MIN``        60        submissions per client address per minute
``AUTH_FAIL_LIMIT``     10        wrong tokens per address per 5 min, then 429
``MAX_WAIT_S``          20        longest long-poll (keep under the proxy's
                                  read timeout)
``FEED_KEEP``           50        results kept per session for the phone
======================  ========  ============================================
"""

import logging
import os
import re
from dataclasses import dataclass

MB = 1024 * 1024

_BASE_PATH_RE = re.compile(r"^/[A-Za-z0-9._~\-/]*$")


def _int(env, name, default, lo=0, hi=None):
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(str(raw).strip())
    except ValueError:
        logging.warning("%s=%r is not an integer; using %r", name, raw, default)
        return default
    if value < lo or (hi is not None and value > hi):
        logging.warning("%s=%d is outside [%s, %s]; using %r", name, value, lo, hi, default)
        return default
    return value


def _flag(env, name, default=False):
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _list(env, name):
    return tuple(x.strip() for x in str(env.get(name) or "").split(",") if x.strip())


def normalize_base_path(raw):
    """``"phone/"`` -> ``"/phone"``; ``""`` and ``"/"`` -> ``""``."""
    raw = (raw or "").strip()
    if raw in ("", "/"):
        return ""
    if not raw.startswith("/"):
        raw = "/" + raw
    raw = raw.rstrip("/")
    if not _BASE_PATH_RE.match(raw) or "//" in raw or ".." in raw.split("/"):
        raise ValueError(f"BASE_PATH {raw!r} is not a plain path prefix")
    return raw


@dataclass(frozen=True)
class Settings:
    web_port: int = 8062
    web_host: str = "0.0.0.0"
    public_url: str = ""
    base_path: str = ""
    trust_proxy: int = 0
    allowed_hosts: tuple = ()
    tls_cert: str = ""
    tls_key: str = ""
    tls_self_signed: bool = False
    tls_san: tuple = ()
    tls_dir: str = "/tmp/phone-tls"
    max_submit_mb: int = 24
    max_images: int = 12
    max_queue: int = 200
    max_store_mb: int = 256
    max_sessions: int = 32
    ttl_s: int = 3600
    rate_per_min: int = 60
    auth_fail_limit: int = 10
    max_wait_s: int = 20
    feed_keep: int = 50
    #: the gRPC message cap (bytes, -1 unlimited): a submission must fit in one
    #: ``get`` reply or it could never be read.
    message_cap: int = -1

    @property
    def tls(self):
        return bool(self.tls_cert and self.tls_key) or self.tls_self_signed

    @property
    def max_submit_bytes(self):
        limit = self.max_submit_mb * MB
        if self.message_cap > 0:
            limit = min(limit, int(self.message_cap * 0.8))
        return limit

    @classmethod
    def from_env(cls, env=None, message_cap=None):
        env = os.environ if env is None else env
        if message_cap is None:
            try:
                import serve
                message_cap = serve.max_message_bytes()
            except Exception:                              # noqa: BLE001
                message_cap = -1
        public = str(env.get("PUBLIC_URL") or "").strip().rstrip("/")
        if public and not re.match(r"^https?://[^\s#?]+$", public):
            logging.warning("PUBLIC_URL=%r is not an http(s) URL; ignoring it", public)
            public = ""
        cert, key = str(env.get("TLS_CERT") or ""), str(env.get("TLS_KEY") or "")
        if bool(cert) != bool(key):
            logging.warning("TLS_CERT and TLS_KEY must both be set; serving plain HTTP")
            cert = key = ""
        return cls(
            web_port=_int(env, "WEB_PORT", 8062, 0, 65535),
            web_host=str(env.get("WEB_HOST") or "0.0.0.0"),
            public_url=public,
            base_path=normalize_base_path(env.get("BASE_PATH")),
            trust_proxy=_int(env, "TRUST_PROXY", 0, 0, 8),
            allowed_hosts=tuple(h.lower() for h in _list(env, "ALLOWED_HOSTS")),
            tls_cert=cert, tls_key=key,
            tls_self_signed=_flag(env, "TLS_SELF_SIGNED"),
            tls_san=_list(env, "TLS_SAN"),
            tls_dir=str(env.get("TLS_DIR") or "/tmp/phone-tls"),
            max_submit_mb=_int(env, "MAX_SUBMIT_MB", 24, 1),
            max_images=_int(env, "MAX_IMAGES", 12, 1, 64),
            max_queue=_int(env, "MAX_QUEUE", 200, 1),
            max_store_mb=_int(env, "MAX_STORE_MB", 256, 1),
            max_sessions=_int(env, "MAX_SESSIONS", 32, 1),
            ttl_s=_int(env, "TTL_S", 3600, 1),
            rate_per_min=_int(env, "RATE_PER_MIN", 60, 1),
            auth_fail_limit=_int(env, "AUTH_FAIL_LIMIT", 10, 1),
            max_wait_s=_int(env, "MAX_WAIT_S", 20, 0, 120),
            feed_keep=_int(env, "FEED_KEEP", 50, 1),
            message_cap=message_cap,
        )
