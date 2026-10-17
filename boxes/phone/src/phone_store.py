"""The phone box's memory: sessions, captures waiting to be read, results to show.

Everything lives in RAM and is bounded - by count per session, by bytes across
all sessions, and by age - so a phone (or a stuck consumer) can fill the box
only up to a stated limit, after which the OLDEST data is dropped and the
consumer is told how much it missed (``dropped``).

One lock guards it all. The gRPC threads and the web server's event loop both
use it; every critical section is a few list operations, never I/O.

A *session* is the unit a program pairs a phone with: its own config (the
labels, the capture mode), its own token, its own queue and its own results
feed. Two programs using two session ids never see each other's data.
"""

import collections
import hashlib
import hmac
import secrets
import threading
import time

from phone_settings import MB


#: A stream counts as running while a frame arrived this recently (seconds).
STREAM_ALIVE_S = 3.0


class StoreError(ValueError):
    """A request the store refuses (too large, too many sessions...)."""


# ---------------------------------------------------------------- sniffing
def sniff_image(head):
    """MIME type from the first bytes, or None. The box never decodes images -
    it stores and forwards bytes - so the magic number is the whole check."""
    head = bytes(head[:16])
    if head[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return None


def is_glb(head):
    return bytes(head[:4]) == b"glTF"


# ---------------------------------------------------------------- records
class Submission:
    __slots__ = ("seq", "t", "images", "mimes", "labels", "note", "sensors",
                 "device", "client_t", "nbytes")

    def __init__(self, seq, t, images, mimes, labels, note, sensors, device, client_t):
        self.seq, self.t = seq, t
        self.images, self.mimes = images, mimes
        self.labels, self.note, self.sensors = labels, note, sensors
        self.device, self.client_t = device, client_t
        self.nbytes = sum(len(i) for i in images)

    def meta(self, first_image):
        return {"seq": self.seq, "t": self.t, "client_t": self.client_t,
                "device": self.device, "labels": self.labels, "note": self.note,
                "sensors": self.sensors, "n_images": len(self.images),
                "first_image": first_image, "mime": self.mimes,
                "bytes": self.nbytes}


class Latest:
    """The newest thing the phone sent, whatever it was: a streamed frame or a
    submitted photo. One slot per session, overwritten, never queued - this is
    what ``latest`` reads, so a stream can run at any rate without filling the box.

    ``n`` counts every update (frames and photos together) and only increases;
    ``seq`` is the submission number when the source is a photo, else None.
    ``nbytes`` counts only bytes this slot owns: a photo's bytes are already
    accounted in the queue.
    """
    __slots__ = ("n", "t", "source", "seq", "images", "mimes", "labels", "note",
                 "sensors", "device", "client_t", "size", "depth", "depth_info",
                 "nbytes")

    def __init__(self, n, t, source, seq, images, mimes, labels, note, sensors,
                 device, client_t, size=None, depth=None, depth_info=None, owned=True):
        self.n, self.t, self.source, self.seq = n, t, source, seq
        self.images, self.mimes = images, mimes
        self.labels, self.note, self.sensors = labels, note, sensors
        self.device, self.client_t, self.size = device, client_t, size
        self.depth, self.depth_info = depth, depth_info
        self.nbytes = (sum(len(i) for i in images) if owned else 0) + len(depth or b"")

    def meta(self, now):
        return {"n": self.n, "t": self.t, "age_s": round(max(0.0, now - self.t), 3),
                "source": self.source, "seq": self.seq, "client_t": self.client_t,
                "device": self.device, "labels": self.labels, "note": self.note,
                "sensors": self.sensors, "size": self.size,
                "n_images": len(self.images), "mime": self.mimes,
                "bytes": sum(len(i) for i in self.images),
                "depth": self.depth_info}


class Card:
    __slots__ = ("id", "t", "title", "text", "ref_seq", "image", "image_mime",
                 "glb", "nbytes")

    def __init__(self, id, t, title, text, ref_seq, image, image_mime, glb):
        self.id, self.t, self.title, self.text, self.ref_seq = id, t, title, text, ref_seq
        self.image, self.image_mime, self.glb = image, image_mime, glb
        self.nbytes = len(image or b"") + len(glb or b"") + len(text or "") + len(title or "")

    def public(self):
        return {"id": self.id, "t": self.t, "title": self.title, "text": self.text,
                "ref_seq": self.ref_seq, "image": self.image is not None,
                "image_mime": self.image_mime, "glb": self.glb is not None}


class Session:
    def __init__(self, sid, token, config, now):
        self.id = sid
        self.token = token
        self.config = config
        self.config_version = 1
        self.created = now
        self.queue = collections.deque()          # Submissions, ascending seq
        self.next_seq = 1
        self.evicted = collections.deque(maxlen=10000)   # seqs dropped unread
        self.cards = collections.deque()
        self.next_card = 1
        self.latest = None                        # Latest | None
        self.next_n = 1
        self.last_frame_t = None                  # when a streamed frame last arrived
        self.devices = {}                         # id -> {name, last_seen}
        self.last_get = None                      # when the program last polled
        self.last_phone = None                    # when a phone last did anything


def _hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


class Store:
    def __init__(self, settings, clock=time.time):
        self.s = settings
        self._clock = clock
        self._cond = threading.Condition()
        self._sessions = {}
        self._by_token = {}
        self._bytes = 0

    def now(self):
        return self._clock()

    # --------------------------------------------------------- sessions
    def setup(self, sid, config, rotate=False):
        """Create the session or replace its config. Returns (session, created)."""
        with self._cond:
            sess = self._sessions.get(sid)
            created = sess is None
            if created:
                if len(self._sessions) >= self.s.max_sessions:
                    raise StoreError(f"at most {self.s.max_sessions} sessions "
                                     f"(MAX_SESSIONS); reset one first")
                sess = Session(sid, secrets.token_urlsafe(24), config, self._clock())
                self._sessions[sid] = sess
                self._by_token[_hash(sess.token)] = sess
            else:
                if sess.config != config:
                    sess.config = config
                    sess.config_version += 1
                if rotate:
                    self._by_token.pop(_hash(sess.token), None)
                    sess.token = secrets.token_urlsafe(24)
                    self._by_token[_hash(sess.token)] = sess
            self._cond.notify_all()
            return sess, created

    def session(self, sid):
        with self._cond:
            return self._sessions.get(sid)

    def find_by_token(self, token):
        if not token:
            return None
        with self._cond:
            sess = self._by_token.get(_hash(token))
            if sess is not None and hmac.compare_digest(sess.token, token):
                return sess
            return None

    def session_ids(self):
        with self._cond:
            return sorted(self._sessions)

    def reset(self, sid=None):
        """Drop one session, or all of them. Returns how many were removed."""
        with self._cond:
            gone = [sid] if sid is not None else list(self._sessions)
            n = 0
            for s in gone:
                sess = self._sessions.pop(s, None)
                if sess is None:
                    continue
                self._by_token.pop(_hash(sess.token), None)
                self._bytes -= sum(x.nbytes for x in sess.queue)
                self._bytes -= sum(c.nbytes for c in sess.cards)
                self._bytes -= sess.latest.nbytes if sess.latest else 0
                n += 1
            self._cond.notify_all()
            return n

    def touch_device(self, sess, device_id, name=None):
        now = self._clock()
        with self._cond:
            sess.last_phone = now
            if device_id:
                sess.devices[device_id] = {"name": name or "phone", "last_seen": now}
                while len(sess.devices) > 32:
                    oldest = min(sess.devices, key=lambda k: sess.devices[k]["last_seen"])
                    del sess.devices[oldest]

    # ------------------------------------------------------ submissions
    def add_submission(self, sess, images, mimes, labels, note, sensors, device, client_t):
        nbytes = sum(len(i) for i in images)
        if nbytes > self.s.max_submit_bytes:
            raise StoreError("submission too large")
        with self._cond:
            if self._sessions.get(sess.id) is not sess:
                raise StoreError("session no longer exists")
            sub = Submission(sess.next_seq, self._clock(), images, mimes, labels,
                             note, sensors, device, client_t)
            sess.next_seq += 1
            sess.queue.append(sub)
            self._bytes += sub.nbytes
            self._set_latest(sess, Latest(
                sess.next_n, sub.t, "photo", sub.seq, images, mimes, labels, note,
                sensors, device, client_t, owned=False))
            self._enforce_limits(sess)
            self._cond.notify_all()
            return sub.seq

    # ----------------------------------------------------------- latest
    def add_frame(self, sess, image, mime, labels, note, sensors, device, client_t,
                  size=None, depth=None, depth_info=None):
        """A streamed frame replaces the session's latest slot. Returns ``n``."""
        if len(image) + len(depth or b"") > self.s.max_frame_bytes:
            raise StoreError("frame too large")
        with self._cond:
            if self._sessions.get(sess.id) is not sess:
                raise StoreError("session no longer exists")
            now = self._clock()
            sess.last_frame_t = now
            n = self._set_latest(sess, Latest(
                sess.next_n, now, "stream", None, [image], [mime], labels, note,
                sensors, device, client_t, size, depth, depth_info))
            self._enforce_limits(sess)
            self._cond.notify_all()
            return n

    def fetch_latest(self, sess, after=0, wait_s=0.0, should_stop=None):
        """The newest Latest with ``n`` > ``after``, or None. ``wait_s`` > 0 blocks
        until one arrives. Returns ``(latest | None, info)``; ``info`` always says
        what the newest ``n`` is and whether frames are arriving right now."""
        deadline = self._clock() + max(0.0, wait_s)
        with self._cond:
            while True:
                if self._sessions.get(sess.id) is not sess:
                    raise StoreError("session was reset")
                cur = sess.latest
                fresh = cur is not None and cur.n > after
                remaining = deadline - self._clock()
                if fresh or remaining <= 0 or (should_stop and should_stop()):
                    break
                self._cond.wait(timeout=min(remaining, 0.5))
            sess.last_get = self._clock()
            now = self._clock()
            info = {"n": cur.n if cur else 0,
                    "streaming": sess.last_frame_t is not None
                    and now - sess.last_frame_t <= STREAM_ALIVE_S,
                    "has_latest": cur is not None}
            return (cur if fresh else None), info

    def fetch(self, sess, after=0, max_items=8, max_bytes=32 * MB, latest=False,
              pop=False, wait_s=0.0, should_stop=None):
        """Submissions with seq > after, oldest first.

        Returns (list, info). ``wait_s`` > 0 blocks until something arrives or
        the time is up. ``should_stop`` is polled so a cancelled RPC frees its
        worker. At least one submission is returned if any is available, even
        one over ``max_bytes`` (``add_submission`` guarantees it fits a reply).
        """
        deadline = self._clock() + max(0.0, wait_s)
        with self._cond:
            while True:
                if self._sessions.get(sess.id) is not sess:
                    raise StoreError("session was reset")
                avail = [s for s in sess.queue if s.seq > after]
                remaining = deadline - self._clock()
                if avail or remaining <= 0 or (should_stop and should_stop()):
                    break
                self._cond.wait(timeout=min(remaining, 0.5))
            sess.last_get = self._clock()
            if latest and avail:
                avail = avail[-1:]
            chosen, total = [], 0
            for s in avail:
                if chosen and (len(chosen) >= max_items or total + s.nbytes > max_bytes):
                    break
                chosen.append(s)
                total += s.nbytes
            pending = 0 if latest else len(avail) - len(chosen)
            dropped = sum(1 for e in sess.evicted if e > after)
            if pop and chosen:
                top = chosen[-1].seq
                gone = [s for s in sess.queue if s.seq <= top] if latest else chosen
                for s in gone:
                    self._bytes -= s.nbytes
                keep = {id(s) for s in gone}
                sess.queue = collections.deque(s for s in sess.queue if id(s) not in keep)
            info = {"seq": sess.next_seq - 1, "pending": pending, "dropped": dropped,
                    "next_after": chosen[-1].seq if chosen else after}
            return chosen, info

    def _set_latest(self, sess, new):
        if sess.latest is not None:
            self._bytes -= sess.latest.nbytes
        sess.latest = new
        sess.next_n += 1
        self._bytes += new.nbytes
        return new.n

    # ------------------------------------------------------------ cards
    def add_card(self, sess, title, text, ref_seq, image, image_mime, glb, replace=False):
        with self._cond:
            if self._sessions.get(sess.id) is not sess:
                raise StoreError("session no longer exists")
            if replace:
                self._bytes -= sum(c.nbytes for c in sess.cards)
                sess.cards.clear()
            card = Card(sess.next_card, self._clock(), title, text, ref_seq,
                        image, image_mime, glb)
            sess.next_card += 1
            sess.cards.append(card)
            self._bytes += card.nbytes
            while len(sess.cards) > self.s.feed_keep:
                self._bytes -= sess.cards.popleft().nbytes
            self._enforce_limits(sess)
            self._cond.notify_all()
            return card.id

    def cards_since(self, sess, after=0):
        with self._cond:
            return [c.public() for c in sess.cards if c.id > after]

    def card(self, sess, card_id):
        with self._cond:
            for c in sess.cards:
                if c.id == card_id:
                    return c
            return None

    # ----------------------------------------------------------- status
    def status(self, sess):
        now = self._clock()
        with self._cond:
            q = sess.queue
            return {
                "config_version": sess.config_version,
                "queue": {"count": len(q), "bytes": sum(s.nbytes for s in q),
                          "oldest_seq": q[0].seq if q else None,
                          "newest_seq": q[-1].seq if q else None},
                "next_seq": sess.next_seq,
                "latest": None if sess.latest is None else {
                    "n": sess.latest.n, "source": sess.latest.source,
                    "age_s": round(max(0.0, now - sess.latest.t), 1)},
                "streaming": sess.last_frame_t is not None
                and now - sess.last_frame_t <= STREAM_ALIVE_S,
                "evicted": len(sess.evicted),
                "cards": len(sess.cards),
                "devices": [{"id": k, "name": v["name"],
                             "last_seen_s": round(now - v["last_seen"], 1)}
                            for k, v in sess.devices.items()],
                "phone_last_seen_s": None if sess.last_phone is None
                else round(now - sess.last_phone, 1),
                "program_last_poll_s": None if sess.last_get is None
                else round(now - sess.last_get, 1),
                "store_bytes": self._bytes,
                "store_limit_bytes": self.s.max_store_mb * MB,
            }

    def program_age(self, sess):
        with self._cond:
            return None if sess.last_get is None else self._clock() - sess.last_get

    # ------------------------------------------------------ housekeeping
    def sweep(self):
        """Drop what has outlived ``TTL_S``. Returns how many items expired."""
        cutoff = self._clock() - self.s.ttl_s
        n = 0
        with self._cond:
            for sess in self._sessions.values():
                while sess.queue and sess.queue[0].t < cutoff:
                    self._evict_one(sess)
                    n += 1
                if sess.latest is not None and sess.latest.t < cutoff:
                    self._bytes -= sess.latest.nbytes
                    sess.latest = None
                    n += 1
                while sess.cards and sess.cards[0].t < cutoff:
                    self._bytes -= sess.cards.popleft().nbytes
                    n += 1
            if n:
                self._cond.notify_all()
        return n

    # ----------------------------------------------------------- internal
    def _evict_one(self, sess):
        sub = sess.queue.popleft()
        sess.evicted.append(sub.seq)
        self._bytes -= sub.nbytes

    def _enforce_limits(self, sess):
        while len(sess.queue) > self.s.max_queue:
            self._evict_one(sess)
        limit = self.s.max_store_mb * MB
        while self._bytes > limit:
            holders = [x for x in self._sessions.values() if x.queue]
            if holders:
                self._evict_one(min(holders, key=lambda x: x.queue[0].t))
                continue
            with_cards = [x for x in self._sessions.values() if x.cards]
            if not with_cards:
                break
            victim = min(with_cards, key=lambda x: x.cards[0].t)
            self._bytes -= victim.cards.popleft().nbytes
