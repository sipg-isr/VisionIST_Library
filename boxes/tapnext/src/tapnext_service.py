import concurrent.futures as futures
import grpc
import grpc_reflection.v1alpha.reflection as grpc_reflection
import logging
import os
import time
import json
import sys
import io
import threading
import tempfile

sys.path.append("./protos")
import pipeline_pb2 as tapnext_pb2
import pipeline_pb2_grpc as tapnext_pb2_grpc
from aux import wrap_value, unwrap_value

import numpy as np
import torch
import cv2


def build_observation_matrix(tracks_list):
    """
    Build Tomasi-Kanade observation matrix P from tracked points.

    Args:
        tracks_list: list of [num_points, 2] arrays (y, x coordinates) per frame

    Returns:
        P: observation matrix of shape (2 * num_frames, num_points)
           Row order: [x1..xN, y1..yN] for all frames concatenated
    """
    if not tracks_list:
        return None

    all_tracks = np.stack(tracks_list)  # [F, N, 2]
    F, N, _ = all_tracks.shape

    P = np.zeros((2 * F, N), dtype=np.float32)

    for f in range(F):
        y_coords = all_tracks[f, :, 0]  # Y coordinates
        x_coords = all_tracks[f, :, 1]  # X coordinates

        P[2*f, :] = x_coords
        P[2*f + 1, :] = y_coords

    return P


def default_model_factory(device):
    """
    Load the real TAPNext model. Kept as a factory (and imported lazily) so the
    session layer can be exercised in-process with a stub model (see test/).
    """
    from tapnet.tapnext.tapnext_torch import TAPNext
    from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint

    logging.info("Loading TAPNext model...")
    model = TAPNext(
        image_size=(256, 256),
        width=768,
        patch_size=(8, 8),
        num_heads=12,
        lru_width=768,
        depth=12,
    ).to(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    checkpoint_path = "/workspace/bootstapnext_ckpt.npz"
    if os.path.exists(checkpoint_path):
        restore_model_from_jax_checkpoint(model, checkpoint_path)
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logging.warning(f"Checkpoint not found at {checkpoint_path}")

    return model


_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_IDLE_TIMEOUT = 120  # seconds of global inactivity before the model parks on CPU
_DEFAULT_SESSION = "default"
# Time-of-day a given student session may sit idle before it (and its GPU state)
# is reaped. Default: 1800 s — reclaims per-session VRAM on the shared GPU
# while a classroom session (few-minute pauses) survives. Set 0 to keep
# sessions forever (e.g. overnight work that must persist).
_SESSION_TTL = float(os.getenv("TAPNEXT_SESSION_TTL", "1800"))

logging.basicConfig(
    format='[ %(levelname)s ] %(asctime)s (%(module)s) %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)


class Session:
    """
    All per-user tracking state. One Session per `session_id`.

    This is exactly the block of globals the service used to keep on `self`;
    moving it here (plus the per-session lock) is what makes the box
    multi-tenant: the model and device lifecycle stay global, everything a
    user can observe lives on their Session only.
    """

    __slots__ = (
        "tracking_state", "active_tracks", "next_track_id",
        "frame_counter", "initialized", "full_tracking_data",
        "accumulated_tracks", "accumulated_visibles", "last_used", "lock",
        # new_tracks (growing) state:
        "generations", "track_birth", "frame_history",
    )

    def __init__(self):
        self.tracking_state = None
        self.active_tracks = {}
        self.next_track_id = 0
        self.frame_counter = 0
        self.initialized = False
        self.full_tracking_data = []   # Tracks accumulated across requests (observation matrix)
        self.accumulated_tracks = []   # Tracks accumulated for responses (y, x coordinates)
        self.accumulated_visibles = []
        self.last_used = time.time()
        self.lock = threading.Lock()   # L2: serializes requests WITHIN this session
        # --- new_tracks (growing point set) ----------------------------------
        # One TAPNext "generation" per seed: {state, ids, pos, vis, streak}.
        # ids are GLOBAL track ids (column i == track id i, assigned in birth
        # order); track_birth[i] is the first frame of that point.
        self.generations = []
        self.track_birth = []
        # backfill: [(rgb256_float01, h, w)] oldest first — needed to re-run a
        # new generation over the session prefix for backfilled trajectories
        self.frame_history = []


def _sniff_video_ext(b: bytes) -> str:
    """Best-effort video container extension from magic bytes.

    Mirrors the yolo box: OpenCV's VideoCapture needs a container it can parse,
    but the extension only steers its backend selection, so ``.mp4`` is the
    safe fallback (covers ISO-BMFF: mp4/mov/m4v all share ``ftyp``).
    """
    if len(b) > 12 and b[4:8] == b"ftyp":
        return ".mp4"
    if b[:4] == b"RIFF" and len(b) > 12 and b[8:12] == b"AVI ":
        return ".avi"
    if b[:4] == b"\x1a\x45\xdf\xa3":          # EBML: webm/mkv
        return ".webm"
    return ".mp4"


class PipelineService(tapnext_pb2_grpc.PipelineServiceServicer):
    """
    Multi-session TAPNext box.

    Lock ordering (global, no exceptions — prevents deadlocks):
      L1  self._sessions_lock   — the sessions dict, session create/delete
      L2  Session.lock          — one per session, held for the full request
      L3  self._device_lock     — the model's CPU<->CUDA move only
    A lock may only be taken if no higher-numbered lock is held.
    """

    def __init__(self, model_factory=None, watchdog_interval=30.0, session_ttl=_SESSION_TTL):
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_factory = model_factory or default_model_factory
        self._last_request_time = time.time()

        self._sessions = {}                    # sid -> Session
        self._sessions_lock = threading.Lock() # L1
        self._device_lock = threading.Lock()   # L3
        self._session_ttl = session_ttl
        self._watchdog_interval = watchdog_interval
        self._load_event = threading.Event()

        self._loader_thread = threading.Thread(target=self._load_model_async, daemon=True)
        self._loader_thread.start()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

        logging.info("TAPNext service initialized (multi-session).")

    # ------------------------------------------------------------------ setup

    def _load_model_async(self):
        try:
            self._model = self._model_factory(self._device)
        except Exception as e:
            logging.exception(f"Failed to load TAPNext model: {e}")
        finally:
            self._load_event.set()

    def _get_session(self, sid):
        """Fetch-or-create the Session for `sid` (L1 only)."""
        with self._sessions_lock:
            sess = self._sessions.get(sid)
            if sess is None:
                sess = Session()
                self._sessions[sid] = sess
                logging.info(f"Session created: {sid} (active sessions: {len(self._sessions)})")
            sess.last_used = time.time()
            return sess

    def _reset_session(self, sess):
        """Reset one session's tracking state. Caller holds no lock (or L1)."""
        sess.tracking_state = None
        sess.active_tracks = {}
        sess.next_track_id = 0
        sess.frame_counter = 0
        sess.initialized = False
        sess.full_tracking_data = []
        sess.accumulated_tracks = []
        sess.accumulated_visibles = []
        sess.generations = []
        sess.track_birth = []
        sess.frame_history = []
        sess.last_used = time.time()
        logging.info("Tracking session state reset")

    # ------------------------------------------------------------- devicelife

    def _promote_to_gpu(self):
        """Move the shared model back to GPU. Call while holding L2 for this
        session (never take L1 here — that would invert the lock order)."""
        if self._device == "cpu" and torch.cuda.is_available() and self._model is not None:
            with self._device_lock:  # L3 under L2 — legal ordering
                logging.info("Request received: restoring model to GPU")
                self._model.to("cuda")
                self._device = "cuda"

    def _watchdog_loop(self):
        while True:
            time.sleep(self._watchdog_interval)
            try:
                self._reap_idle_sessions()
                self._park_model_if_idle()
            except Exception:
                logging.exception("Watchdog cycle failed")

    def _reap_idle_sessions(self):
        """Drop sessions idle beyond the TTL (frees their per-session state).
        A session with a request in flight is skipped this cycle (L2 busy)."""
        if not self._session_ttl or self._session_ttl <= 0:
            return
        now = time.time()
        reaped_any = False
        with self._sessions_lock:  # L1
            for sid, sess in list(self._sessions.items()):
                if now - sess.last_used <= self._session_ttl:
                    continue
                if not sess.lock.acquire(timeout=0.05):
                    continue  # in flight — leave it for the next cycle
                try:
                    del self._sessions[sid]
                    reaped_any = True
                    logging.info(
                        f"Reaped idle session {sid} (idle {now - sess.last_used:.0f}s)")
                finally:
                    sess.lock.release()  # L2 released before L1
        if reaped_any:
            # The dead sessions' GPU tensors (tracking_state etc.) are now
            # unreferenced; give the allocator blocks back to the driver.
            torch.cuda.empty_cache()

    def _park_model_if_idle(self):
        """Park the (expensive, shared) model on CPU after global idle.
        Session state tensors may remain on GPU; they are the small per-session
        cost and are reaped separately by TTL. Skipped while any session is
        computing (its L2 is held)."""
        if self._device != "cuda":
            return
        if time.time() - self._last_request_time <= _IDLE_TIMEOUT:
            return
        held = []
        with self._sessions_lock:  # L1
            for sess in self._sessions.values():
                if not sess.lock.acquire(timeout=0.05):
                    for s in held:
                        s.lock.release()
                    return  # someone is mid-inference; retry next cycle
                held.append(sess)  # L2 under L1 — legal ordering
            try:
                # The per-session tracking_state tensors live on CUDA and are
                # NOT released by the model move (plain session attributes).
                # Drop them: the session re-initializes on its next frame
                # (initialized goes False), so nothing observable is lost.
                for s in held:
                    s.tracking_state = None
                    s.initialized = False
                    # growing sessions: their generation states are CUDA tensors
                    # too. Drop everything — the session re-seeds generation 0
                    # on its next frame (a soft reset, same as the legacy path)
                    s.generations = []
                    s.track_birth = []
                    s.frame_history = []
                logging.info(
                    f"Parking model to CPU after _IDLE_TIMEOUT "
                    f"({len(held)} session(s) also drop their tracking state)")
                with self._device_lock:  # L3 under L2 under L1 — legal
                    self._model.to("cpu")
                    torch.cuda.empty_cache()
                    self._device = "cpu"
            finally:
                for s in held:
                    s.lock.release()

    # ----------------------------------------------------------------- public

    def Process(self, request, context):
        while not self._load_event.is_set():
            time.sleep(0.1)

        try:
            if not request.config_json:
                return tapnext_pb2.Envelope(
                    config_json=json.dumps({"tapnext": {"status": "error", "error": "No config JSON"}})
                )

            config = json.loads(request.config_json)
            tapnext_config = config.get("tapnext", {}) or {}
            sid = tapnext_config.get("session_id") or _DEFAULT_SESSION
            parameters = tapnext_config.get("parameters", {}) or {}

            command = tapnext_config.get("command")

            if command == "reset" or parameters.get("reset"):
                # Reset is scoped to THIS session — other sessions are untouched.
                sess = self._get_session(sid)
                with sess.lock:
                    self._reset_session(sess)
                return tapnext_pb2.Envelope(
                    config_json=json.dumps(
                        {"tapnext": {"status": "done", "action": "reset", "session": sid}})
                )

            if command == "list":
                # Operator view: active sessions only (state content is never
                # disclosed). Anyone who can reach the box can enumerate sids.
                with self._sessions_lock:
                    now = time.time()
                    listing = [
                        {
                            "session": kid,
                            "frames_processed": len(k.accumulated_tracks),
                            "num_tracks": len(k.active_tracks),
                            "idle_seconds": round(now - k.last_used, 1),
                        }
                        for kid, k in self._sessions.items()
                    ]
                return tapnext_pb2.Envelope(
                    config_json=json.dumps(
                        {"tapnext": {"status": "done", "action": "list", "sessions": listing}})
                )

            has_images = "images" in request.data
            has_video  = "video"  in request.data

            # `data.images` (a list of image bytes) and `data.video` (a single
            # video file) are mutually-exclusive inputs to the SAME per-frame
            # tracking loop: the video is decoded server-side into ordered
            # frames, mirroring the yolo box's `data.video` input.
            if has_images and has_video:
                return tapnext_pb2.Envelope(
                    config_json=json.dumps(
                        {"tapnext": {"status": "error",
                                     "error": "send either data.images or data.video, not both",
                                     "session": sid}})
                )
            if not has_images and not has_video:
                return tapnext_pb2.Envelope(
                    config_json=json.dumps(
                        {"tapnext": {"status": "empty_request", "session": sid}})
                )

            start_time = time.time()

            sess = self._get_session(sid)
            with sess.lock:  # L2: this session's requests are serialized here
                self._last_request_time = time.time()
                self._promote_to_gpu()

                growing = bool(parameters.get("new_tracks", False))

                def track_frame_in_session(frame_np):
                    """Feed one decoded frame into this session's tracker (None-safe)."""
                    if frame_np is None:
                        return
                    if growing:
                        tracks, visibles = self._track_frame_growing(frame_np, parameters, sess)
                    else:
                        tracks, visibles = self._track_frame(frame_np, parameters, sess)
                    if tracks is not None:
                        sess.accumulated_tracks.append(tracks)
                        sess.accumulated_visibles.append(visibles)
                        # Accumulate for full observation matrix across requests
                        sess.full_tracking_data.append((tracks.copy(), visibles.copy()))

                if has_images:
                    image_bytes_list = unwrap_value(request.data["images"])
                    if not isinstance(image_bytes_list, list) or len(image_bytes_list) == 0:
                        return tapnext_pb2.Envelope(
                            config_json=json.dumps(
                                {"tapnext": {"status": "error", "error": "No images in data",
                                             "session": sid}})
                        )
                    for img_bytes in image_bytes_list:
                        track_frame_in_session(self._decode_image(img_bytes))
                else:
                    video_bytes = unwrap_value(request.data["video"])
                    if not isinstance(video_bytes, (bytes, bytearray, memoryview)) or len(video_bytes) == 0:
                        return tapnext_pb2.Envelope(
                            config_json=json.dumps(
                                {"tapnext": {"status": "error", "error": "No video in data",
                                             "session": sid}})
                        )
                    frame_step = int(parameters.get("frame_step", 1) or 1)
                    max_frames = int(parameters.get("max_frames", 0) or 0)   # 0 -> no cap
                    for frame_np in self._decode_video(bytes(video_bytes), frame_step, max_frames):
                        track_frame_in_session(frame_np)

                growing_nmax = 0
                response_data = {}
                if sess.accumulated_tracks:
                    if growing:
                        # Rows have variable width (the point set grew over the
                        # session). Pad to a common width: not-yet-born columns
                        # are 0.0 + visibles=0 in `tracks`/`visibles`, NaN in
                        # `observation_matrix`. `visibles` is the validity mask;
                        # `data.birth_frames` (one per track id, column i ==
                        # track id i) + the config's `birth_hist` separate
                        # "not yet born" from "born but lost".
                        F = len(sess.accumulated_tracks)
                        growing_nmax = max(tf.shape[0] for tf in sess.accumulated_tracks)
                        t_arr = np.zeros((F, growing_nmax, 2), dtype=np.float32)
                        v_arr = np.zeros((F, growing_nmax), dtype=np.float32)
                        p_arr = np.full((2 * F, growing_nmax), np.nan, dtype=np.float32)
                        for f in range(F):
                            tf = sess.accumulated_tracks[f]
                            vf = sess.accumulated_visibles[f]
                            n = tf.shape[0]
                            t_arr[f, :n] = tf
                            v_arr[f, :n] = vf
                            p_arr[2 * f, :n] = tf[:, 1]
                            p_arr[2 * f + 1, :n] = tf[:, 0]
                        tracks_tensor = torch.from_numpy(t_arr)
                        visibles_tensor = torch.from_numpy(v_arr)
                        p_tensor = torch.from_numpy(p_arr)
                    else:
                        tracks_tensor = torch.stack([torch.from_numpy(t) for t in sess.accumulated_tracks])
                        visibles_tensor = torch.stack([torch.from_numpy(v) for v in sess.accumulated_visibles])
                        # Build the Tomasi-Kanade matrix (uniform-width rows only;
                        # the growing path above builds its padded equivalent)
                        P = build_observation_matrix([t for t, _ in sess.full_tracking_data])
                        p_tensor = torch.tensor(P) if P is not None else None

                    response_data["tracks"] = wrap_value(self._serialize_tensor(tracks_tensor))
                    response_data["visibles"] = wrap_value(self._serialize_tensor(visibles_tensor))

                    if p_tensor is not None:
                        response_data["observation_matrix"] = wrap_value(
                            self._serialize_tensor(p_tensor))

                    if growing:
                        # the per-track birth list lives in DATA (a typed
                        # float-list — rendered compact by client/webui), not
                        # in the config JSON: a few hundred raw ints there
                        # would flood any human-facing view; the config keeps
                        # only the compact birth_hist summary
                        response_data["birth_frames"] = wrap_value(
                            [float(b) for b in sess.track_birth])

                logging.info(
                    f"session={sid} frames={len(sess.accumulated_tracks)} "
                    f"runtime={time.time() - start_time:.3f}s "
                    f"active_sessions={self._count_sessions()}")

                section = {
                    "status": "done",
                    "session": sid,
                    "frames_processed": len(sess.accumulated_tracks),
                    "runtime": time.time() - start_time,
                    "num_points": growing_nmax if growing
                        else (sess.accumulated_tracks[0].shape[0] if sess.accumulated_tracks else 0),
                    # Declared payload encoding (generic visionist_client contract):
                    # all tensor responses are torch.save()-format bytes.
                    "encoding": {
                        "tracks": "torch",
                        "visibles": "torch",
                        "observation_matrix": "torch",
                    }
                }
                if growing:
                    section["new_tracks"] = True
                    section["num_generations"] = len(sess.generations)
                    # compact growth summary (frame -> how many points were
                    # seeded then); the full per-track list is data.birth_frames
                    born_frames = sorted(set(sess.track_birth))
                    section["birth_hist"] = {
                        str(int(f)): sess.track_birth.count(f) for f in born_frames
                    }

                return tapnext_pb2.Envelope(config_json=json.dumps({"tapnext": section}),
                                            data=response_data)

        except Exception as e:
            logging.exception(f"Error in Process: {e}")
            return tapnext_pb2.Envelope(
                config_json=json.dumps({"tapnext": {"status": "error", "error": str(e)}})
            )

    def _count_sessions(self):
        with self._sessions_lock:
            return len(self._sessions)

    # ------------------------------------------------------------- internals

    def _decode_image(self, img_bytes):
        try:
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            logging.error(f"Failed to decode image: {e}")
            return None

    def _decode_video(self, video_bytes, frame_step=1, max_frames=0):
        """Server-side decode of one video file (mp4/avi/webm/mov) into an
        ordered list of BGR frames — every `frame_step`-th frame, up to
        `max_frames` (0 = no cap). Same recipe as the yolo box: sniff the
        container from magic bytes, spill to a temp file, read with
        `cv2.VideoCapture`. BGR is what `_track_frame` expects (it converts
        BGR->RGB internally)."""
        suffix = _sniff_video_ext(video_bytes)
        tmp_path = None
        frames = []
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp_path = tmp.name
            tmp.write(video_bytes)
            tmp.close()
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                logging.error(
                    "cv2.VideoCapture could not open the video "
                    f"(sniffed container {suffix!r} — unsupported codec?)")
                return []
            decoded = 0
            step = max(1, int(frame_step))
            while (max_frames <= 0 or len(frames) < max_frames):
                ok, frame = cap.read()
                if not ok:
                    break
                if decoded % step == 0:
                    frames.append(frame)
                decoded += 1
            cap.release()
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        return frames

    def _track_frame_growing(self, frame_np, parameters, sess):
        """``new_tracks: true`` path: the point set *grows* across the session.

        Each seed is one TAPNext **generation** (its own ``tracking_state``, so
        surviving points keep their long-range LRU memory — no re-anchoring,
        unlike a merged re-init). When visible coverage thins out, empty grid
        cells get fresh points, starting new generations with fresh track ids.

        Per frame: step all generations → optionally retire fully-invisible
        ones (``retire_after_invisible``) → every ``add_interval`` frames re-seed
        empty grid cells (occupancy vs THIS frame's visible 256-space
        positions; capped by ``max_new_per_seed`` and ``max_total_points``).
        ``backfill`` seeds a generation over the whole stored session prefix
        (O(k) cost) so its recorded trajectory starts at frame 0; otherwise a
        new point simply does not exist before its birth frame.

        Returns (tracks (N, 2) in original resolution, ``(y, x)`` order; N ==
        ``len(sess.track_birth)`` — all columns including retired/not-yet-born,
        which are 0.0 there with ``visibles`` False), keeping column i ==
        track id i across the whole session.
        """
        if frame_np.ndim == 2:
            rgb = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = frame_np.shape[:2]
        rgb256 = np.ascontiguousarray(
            cv2.resize(rgb, (256, 256)).astype(np.float32) / 255.0)

        cell = int(parameters.get("cell_size", 20))
        min_dist = float(parameters.get("min_dist", 4.0))
        add_interval = max(1, int(parameters.get("add_interval", 5)))
        max_new = int(parameters.get("max_new_per_seed", 256))
        max_total = int(parameters.get("max_total_points", 4096))
        retire_after = int(parameters.get("retire_after_invisible", 0))
        backfill = bool(parameters.get("backfill", False))
        gsize = int(parameters.get("grid_size", 32))

        if backfill:
            # 256-space frames + that frame's original size (kept so backfilled
            # columns can be scaled back correctly per frame)
            sess.frame_history.append((rgb256, orig_h, orig_w))

        frame_t = torch.from_numpy(rgb256).unsqueeze(0).unsqueeze(0).to(self._device)
        scale_y, scale_x = orig_h / 256.0, orig_w / 256.0

        with torch.no_grad():
            use_amp = (self._device == "cuda")
            with torch.amp.autocast(self._device, dtype=torch.float16, enabled=use_amp):
                # --------------------------------------------------------- seed 0
                # First growing frame of the session (nothing has ever been
                # seeded here; note: NOT "no generations" — if they all retire
                # we must fall through to the normal capped seed logic below,
                # never a silent full-grid reseed that bypasses the budgets).
                if len(sess.track_birth) == 0:
                    xs = np.linspace(10.0, 246.0, gsize)
                    ys = np.linspace(10.0, 246.0, gsize)
                    gx, gy = np.meshgrid(xs, ys, indexing="xy")
                    cells = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
                    tracks, _, vl, state = self._model(
                        video=frame_t, query_points=self._grid_query_points(cells))
                    pos256 = tracks[0, 0].cpu().numpy().astype(np.float32)
                    # real TAPNext returns visible_logits with a trailing logit
                    # channel: (B, T, N, 1) — normalize to 1-D here
                    vis = (vl[0, 0] > 0).cpu().numpy().reshape(-1)
                    base = len(sess.track_birth)
                    sess.track_birth.extend([0] * pos256.shape[0])
                    sess.generations.append({
                        "state": state,
                        "ids": list(range(base, base + pos256.shape[0])),
                        "pos": pos256, "vis": vis, "streak": 0,
                    })
                else:
                    k = len(sess.accumulated_tracks)   # this frame's 0-based index

                    # ----------------------------------- retire dead (pre-step)
                    # Streaks were accumulated up to now; a generation that just
                    # crossed the threshold is dropped BEFORE this frame's step,
                    # so its previously recorded positions remain but it costs
                    # nothing more from here on.
                    if retire_after > 0:
                        dead = [g for g in sess.generations if g["streak"] >= retire_after]
                        sess.generations = [g for g in sess.generations if g not in dead]
                        for g in dead:
                            g["state"] = None   # let the LRU cache's tensors go
                            logging.info(
                                f"Retired generation {len(g['ids'])} pts after "
                                f"{retire_after} fully-invisible frames")

                    # ------------------------------------------------- step all
                    vis_pos256 = []   # 256-space, model output order (y, x)
                    for gen in sess.generations:
                        tr, _, vl, state = self._model(video=frame_t, state=gen["state"])
                        gen["state"] = state
                        gen["pos"] = tr[0, 0].cpu().numpy().astype(np.float32)
                        gen["vis"] = (vl[0, 0] > 0).cpu().numpy().reshape(-1)
                        if retire_after > 0:
                            gen["streak"] = 0 if gen["vis"].any() else gen["streak"] + 1
                        if gen["vis"].any():
                            vis_pos256.append(gen["pos"][gen["vis"]])

                    # ------------------------------------- re-seed empty cells
                    room = max_total - len(sess.track_birth)
                    if k > 0 and k % add_interval == 0 and room > 0:
                        # occupancy is measured in (x, y); the model's output
                        # order is (y, x) — flip before comparing to cells
                        existing = (np.concatenate(
                            [p[:, ::-1] for p in vis_pos256], axis=0)
                            if vis_pos256 else np.zeros((0, 2), dtype=np.float32))
                        new_pts = self._empty_cells(existing, cell, min_dist)
                        new_pts = new_pts[:min(max_new, room)]
                        if new_pts.shape[0] > 0:
                            n_hist = len(sess.frame_history)
                            do_back = (backfill and n_hist == k + 1 and n_hist > 1)
                            if do_back:
                                # New generation over the whole stored prefix —
                                # the new points get a recorded trajectory back
                                # to frame 0, written into the already-accumulated
                                # rows (they gain one column each).
                                prefix = np.stack(
                                    [fh[0] for fh in sess.frame_history]).astype(np.float32)
                                vid = torch.from_numpy(
                                    np.ascontiguousarray(prefix)).unsqueeze(0).to(self._device)
                                trf, _, vlf, state = self._model(
                                    video=vid,
                                    query_points=self._grid_query_points(new_pts, t=float(n_hist - 1)))
                                pos256 = trf[0, n_hist - 1].cpu().numpy().astype(np.float32)
                                vis = (vlf[0, n_hist - 1] > 0).cpu().numpy().reshape(-1)
                                for f in range(k):
                                    _, fh_h, fh_w = sess.frame_history[f]
                                    pos_f = trf[0, f].cpu().numpy().astype(np.float32)
                                    pos_f[:, 0] *= fh_h / 256.0
                                    pos_f[:, 1] *= fh_w / 256.0
                                    vis_f = (vlf[0, f] > 0).cpu().numpy().reshape(-1)
                                    # the new points were not born yet at these
                                    # earlier frames — one column appended per
                                    # existing column, i.e. append ROWS:
                                    sess.accumulated_tracks[f] = np.vstack(
                                        [sess.accumulated_tracks[f], pos_f])
                                    sess.accumulated_visibles[f] = np.concatenate(
                                        [sess.accumulated_visibles[f], vis_f])
                                    tp, tv = sess.full_tracking_data[f]
                                    sess.full_tracking_data[f] = (
                                        np.vstack([tp, pos_f]),
                                        np.concatenate([tv, vis_f]))
                            else:
                                tr, _, vl, state = self._model(
                                    video=frame_t, query_points=self._grid_query_points(new_pts))
                                pos256 = tr[0, 0].cpu().numpy().astype(np.float32)
                                vis = (vl[0, 0] > 0).cpu().numpy().reshape(-1)
                            base = len(sess.track_birth)
                            n_new = pos256.shape[0]
                            sess.track_birth.extend([k] * n_new)
                            sess.generations.append({
                                "state": state,
                                "ids": list(range(base, base + n_new)),
                                "pos": pos256, "vis": vis, "streak": 0,
                            })
                            logging.info(f"Re-seeded {n_new} new points "
                                         f"(generation {len(sess.generations)}) at frame {k}")

                # ---------------------------------- row in track-id (column) order
                n_total = len(sess.track_birth)
                pos_all = np.zeros((n_total, 2), dtype=np.float32)
                vis_all = np.zeros(n_total, dtype=bool)
                for gen in sess.generations:
                    ids = np.asarray(gen["ids"], dtype=np.int64)
                    pos_all[ids] = gen["pos"]
                    vis_all[ids] = gen["vis"]
                # 256-space (x, y) -> original resolution (y, x), as _track_frame
                pos_all[:, 0] *= scale_y
                pos_all[:, 1] *= scale_x
                return pos_all, vis_all

    def _grid_query_points(self, points_xy, t=0.0):
        """(n, 2) 256-space (x, y) points -> TAPNext query_points [1, n, 3] (t, x, y)."""
        n = points_xy.shape[0]
        q = np.empty((n, 3), dtype=np.float32)
        q[:, 0] = t
        q[:, 1] = points_xy[:, 0]
        q[:, 2] = points_xy[:, 1]
        return torch.from_numpy(q).unsqueeze(0).to(self._device)

    def _empty_cells(self, existing, cell_size, min_dist):
        """Centers of the 256x256 grid cells (step ``cell_size``) that hold no
        ``existing`` (x, y) point within ``min_dist`` — the candidates to seed."""
        ys = np.arange(cell_size // 2, 256, cell_size)
        xs = np.arange(cell_size // 2, 256, cell_size)
        gy, gx = np.meshgrid(ys, xs, indexing="ij")
        centers = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
        if existing.shape[0] == 0:
            return centers
        d2 = ((centers[:, None, :] - existing[None, :, :]) ** 2).sum(axis=2)
        return centers[d2.min(axis=1) >= min_dist * min_dist]

    def _track_frame(self, frame_np, parameters, sess):
        if frame_np.ndim == 2:
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2RGB)
        else:
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = frame_np.shape[:2]

        # 1. Resize frame to 256x256
        frame_resized = cv2.resize(frame_np, (256, 256))

        # 2. Prepare tensor: [B=1, T=1, H=256, W=256, C=3]
        frame_tensor = torch.from_numpy(frame_resized).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0).to(self._device)

        with torch.no_grad():
            use_amp = (self._device == "cuda")
            with torch.amp.autocast(self._device, dtype=torch.float16, enabled=use_amp):
                if not sess.initialized:
                    grid_size = parameters.get("grid_size", 32)

                    # FIX: Correct coordinate ordering [t, x, y] for PyTorch grid_sample
                    x_coords = np.linspace(10.0, 246.0, grid_size)
                    y_coords = np.linspace(10.0, 246.0, grid_size)
                    xx, yy = np.meshgrid(x_coords, y_coords, indexing='xy')

                    query_points = []
                    for x, y in zip(xx.flatten(), yy.flatten()):
                        query_points.append([0.0, float(x), float(y)])

                    query_points_tensor = torch.tensor(
                        query_points, dtype=torch.float32
                    ).unsqueeze(0).to(self._device) # [1, N, 3]

                    # TAPNext initialization
                    tracks, track_logits, visible_logits, sess.tracking_state = self._model(
                        video=frame_tensor,
                        query_points=query_points_tensor
                    )

                    num_feats = tracks.shape[2]
                    for i in range(num_feats):
                        sess.active_tracks[i] = sess.next_track_id
                        sess.next_track_id += 1

                    sess.initialized = True
                else:
                    # Sequential step inference using saved tracking state
                    tracks, track_logits, visible_logits, sess.tracking_state = self._model(
                        video=frame_tensor,
                        state=sess.tracking_state
                    )

                sess.frame_counter += 1

                tracks_np = tracks.cpu().numpy()[0, 0].copy() # [N, 2] -> (x, y)
                visibles_np = (visible_logits.cpu().numpy()[0, 0] > 0)

                # Flip column order to (y, x), scaled back to original resolution
                tracks_yx = tracks_np # Index 0 is Y, Index 1 is X
                scale_y = orig_h / 256.0
                scale_x = orig_w / 256.0
                tracks_yx[:, 0] *= scale_y  # Y coordinate (scaled to orig_h)
                tracks_yx[:, 1] *= scale_x  # X coordinate (scaled to orig_w)

                return tracks_yx, visibles_np

    def _serialize_tensor(self, tensor):
        buf = io.BytesIO()
        torch.save(tensor.cpu(), buf, pickle_protocol=4)
        return buf.getvalue()


def get_port():
    try:
        port = int(os.getenv('PORT', _PORT_DEFAULT))
        if port <= 0:
            logging.error('Port must be positive')
            return None
        return port
    except ValueError:
        logging.exception('Invalid port value')
        return None


def run_server(server):
    port = get_port()
    if not port:
        return

    target = f'[::]:{port}'
    server.add_insecure_port(target)
    server.start()
    logging.info(f'Server started at {target}')

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    server = grpc.server(
        futures.ThreadPoolExecutor(),
        options=[
            ('grpc.max_send_message_length', -1),
            ('grpc.max_receive_message_length', -1),
        ]
    )

    tapnext_pb2_grpc.add_PipelineServiceServicer_to_server(PipelineService(), server)

    service_names = (
        tapnext_pb2.DESCRIPTOR.services_by_name['PipelineService'].full_name,
        grpc_reflection.SERVICE_NAME
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)
