"""YOLO box — object detection AND tracking on images and videos
(ultralytics BoT-SORT-class tracker, every call).

A standard shared-envelope box with **per-session tracker state** (the same
multi-session contract as ``tapnext`` — see its service for the pattern):

* ``data["images"]`` — list of image bytes (JPEG/PNG/…); one detection
  record per image.
* ``data["video"]``  — one video file (mp4/avi/webm/mov); the box decodes
  it server-side (OpenCV/ffmpeg), samples ``frame_step``, and returns one
  detection record per sampled frame (with the original frame index).
* **Tracking is always on**: every call runs the model in ``track`` mode, so
  detections carry a per-session ``track_id`` (the tracker state is a
  byproduct of detection, not a separate mode).
* ``session_id`` (section top level; omitted -> shared ``default`` session)
  keys the tracker state: same session = stable IDs across calls; different
  sessions = independent ID sequences.  Server-side sessions are reaped
  after ``YOLO_SESSION_TTL`` idle (default 1800 s; 0 keeps them forever).
* ``command: reset`` — clears **this session's** tracker state (other
  sessions' live tracks untouched); the next call adopts a fresh tracker,
  and ANY fresh tracker starts its numbering from the base (see
  ``_attach_tracker``) — so "reset" always means "start over", exactly
  like a brand-new session (new key / regenerate / page reload): track
  ids are per-session namespaces and a new session can never collide
  with an old one, so both start at 1.
* ``command: list`` — operator view of active sessions (id/opaque only).

One YOLO model is shared by all sessions; the ultralytics *tracker* object
(owned by the session) is attached to the shared predictor per call — so a
session switch costs nothing but the small tracker state.

GPU lifecycle (fleet convention): the model loads on CPU at startup, moves
to CUDA in place on the first request (unless ``parameters.device`` says
otherwise), and a watchdog thread moves it back to CPU after ``_IDLE_TIMEOUT``
seconds of inactivity. Weights are **not** baked into the image: the default
checkpoint (env ``YOLO_WEIGHTS``, default ``yolov8n.pt``) is downloaded/loaded
when the box starts, and ``parameters.weights`` switches the active
checkpoint per call (fresh ones download on first use, into the container
workspace).

Payloads: ``detections`` is JSON (declared via the ``encoding`` contract
key so ``visionist_client`` decodes it); ``annotated`` is a list of JPEG bytes
(declared ``identity``); a video input additionally yields
``annotated_video`` — the annotated frames re-encoded as one MP4 (H.264
when PyAV is available so the browser's ``<video>`` can play it, mp4v
fallback otherwise).
"""

import concurrent.futures as futures
import io
import json
import logging
import os
import sys
import tempfile
import threading
import time
import zipfile

sys.path.append("./protos")
import pipeline_pb2  # noqa: E402
import pipeline_pb2_grpc  # noqa: E402
from aux import wrap_value, unwrap_value
import serve  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

_PORT_ENV_VAR = 'PORT'
_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_IDLE_TIMEOUT = 60  # seconds

_DEFAULT_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov8n.pt")  # runtime download
_DEFAULT_SESSION = "default"
_SESSION_TTL = int(os.getenv("YOLO_SESSION_TTL", "1800") or 1800)  # 0 = keep forever
_DEFAULTS = {
    "conf": 0.25,
    "iou": 0.70,
    "imgsz": 640,
    "max_det": 300,
    "save_annotated": True,
    "frame_step": 1,
    "max_frames": 1024,
}


class _Session:
    """Per-session tracker state (the tapnext multi-session pattern).

    The *tracker object* is owned by the session; it is attached to the
    shared predictor only for the duration of a call (see
    ``_track_frames``), so one YOLO model serves every session."""

    __slots__ = ("tracker", "last_used", "created", "frames")

    def __init__(self):
        self.tracker = None
        self.created = time.time()
        self.last_used = self.created
        self.frames = 0


def _error(message, extra=None):
    section = {"status": "error", "error": message}
    if extra:
        section.update(extra)
    return pipeline_pb2.Envelope(config_json=json.dumps({"yolo": section}))


def _sniff_video_ext(b: bytes) -> str:
    """Best-effort video container extension from magic bytes.

    OpenCV's VideoCapture needs a container it can parse; the extension only
    steers its backend selection, so mis-guessing is low-risk (``.mp4`` is
    the fallback because it covers ISO-BMFF: mp4/mov/m4v all share ``ftyp``).
    """
    if len(b) > 12 and b[4:8] == b"ftyp":
        return ".mp4"
    if b[:4] == b"RIFF" and len(b) > 12 and b[8:12] == b"AVI ":
        return ".avi"
    if b[:4] == b"\x1a\x45\xdf\xa3":          # EBML: webm/mkv
        return ".webm"
    return ".mp4"


class PipelineService(pipeline_pb2_grpc.PipelineServiceServicer):

    def __init__(self):
        import torch  # deferred: ultralytics needs it, the box should not
        from ultralytics import YOLO

        self._torch = torch
        # Weights are NOT baked into the image (keeps the build deterministic
        # and offline-friendly): the default checkpoint downloads at box
        # startup into the container workspace (CWD, writable by the runner
        # user), like clip's startup ViT download. Always loaded to CPU
        # first; moved to GPU lazily on request (below).
        self._models = {}
        self._default_weights = _DEFAULT_WEIGHTS
        self._current_weights = _DEFAULT_WEIGHTS
        self._model = YOLO(_DEFAULT_WEIGHTS)
        self._models[_DEFAULT_WEIGHTS] = self._model
        self._device = "cpu"
        logging.info("YOLO model (%s) loaded on CPU", _DEFAULT_WEIGHTS)

        self._last_request_time = time.time()
        self._lock = threading.Lock()

        # Multi-session tracker state (see module docstring).
        self._sessions = {}
        self._sessions_lock = threading.Lock()
        self._session_ttl = _SESSION_TTL
        # Serializes "attach session tracker -> track -> adopt" so concurrent
        # calls from different sessions can't cross-wire predictors.
        self._track_lock = threading.Lock()
        # predictor ids whose TRACKTRACK setup hook was attached (idempotence
        # guard — calling it twice would double-wrap the predictor).
        self._tracker_setups = set()

        # Background thread to monitor idle time.
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    # ------------------------------------------------------------- weights
    def _activate_weights(self, weights):
        """Switch the active checkpoint (under the lock).

        The box stays **stateless**: ``weights`` is resolved *per call*
        (explicit ``parameters.weights``, else the startup default) — the
        cache only avoids re-loading the last-used checkpoints, it never
        leaks state between calls. Inactive cached models are parked on CPU
        (a GPU-resident model is moved down before the switch) so only the
        *active* model can hold VRAM — the watchdog lifecycle stays 1:1
        with the active model. Unknown checkpoints are fetched by
        ultralytics on first use (the download lands in the container
        workspace).
        """
        with self._lock:
            if weights == self._current_weights:
                return
            if self._device.startswith("cuda"):
                logging.info("Switching weights: parking %s on CPU", self._current_weights)
                self._model.to("cpu")
                self._torch.cuda.empty_cache()
                self._device = "cpu"
            if weights not in self._models:
                logging.info("Loading checkpoint %s (first use may download)", weights)
                from ultralytics import YOLO
                self._models[weights] = YOLO(weights)
            self._model = self._models[weights]
            self._current_weights = weights
            logging.info("Active checkpoint: %s (CPU)", weights)

    # ------------------------------------------------------------------ GPU
    def _watchdog_loop(self):
        while True:
            time.sleep(10)  # check every 10s
            idle_time = time.time() - self._last_request_time
            if not (idle_time > _IDLE_TIMEOUT and self._device.startswith("cuda")):
                continue
            with self._sessions_lock:
                # Global idle > _IDLE_TIMEOUT implies every session is idle
                # too (both clocks advance on each request), so nobody is
                # mid-track and dropping the live trackers is safe.  The next
                # call adopts a FRESH tracker, and per this box's contract a
                # fresh tracker restarts its numbering from the base (_attach
                # _tracker) — exactly the "reset == start over" semantics.
                for s in self._sessions.values():
                    s.tracker = None
            self._tracker_setups.clear()
            with self._lock:
                if not self._device.startswith("cuda"):
                    return
                logging.info("Idle timeout reached: moving model back to CPU")
                self._model.to("cpu")
                self._device = "cpu"
                # Also drop the cached predictor's tracker slot: it is a plain
                # attribute holding CUDA buffers, unreachable by .to()/
                # empty_cache().
                try:
                    predictor = self._model.predictor
                    if predictor is not None and hasattr(predictor, "trackers"):
                        del predictor.trackers
                except Exception:
                    logging.exception("failed to drop cached predictor trackers")
                self._torch.cuda.empty_cache()

    def _place_model(self, parameters):
        """Move the model (under the lock): explicit ``parameters.device``
        wins; otherwise CUDA if visible. Returns the resulting device."""
        target = str(parameters.get("device") or "").strip().lower()
        if not target and self._torch.cuda.is_available():
            target = "cuda"
        if target and self._device != target:
            logging.info("Moving model to %s", target)
            with self._lock:
                self._model.to(target)
                self._device = target
        return self._device

    # ---------------------------------------------------------------- sessions
    def _reap_idle_sessions(self):
        """Drop sessions idle beyond the TTL (frees their tracker state)."""
        if not self._session_ttl:
            return
        now = time.time()
        with self._sessions_lock:
            stale = [(sid, now - k.last_used) for sid, k in self._sessions.items()
                     if now - k.last_used > self._session_ttl]
            for sid, idle in stale:
                del self._sessions[sid]
                logging.info("Reaped idle yolo session %s (idle %.0fs)", sid, idle)

    def _get_session(self, sid):
        """Fetch/create a session (tapnext semantics), bumping its clock."""
        self._reap_idle_sessions()
        with self._sessions_lock:
            sess = self._sessions.get(sid)
            if sess is None:
                sess = _Session()
                self._sessions[sid] = sess
                logging.info("YOLO session created: %s (active: %d)",
                             sid, len(self._sessions))
            sess.last_used = time.time()
            return sess

    def _rewind_tracker_counter(self):
        """Rewind the ultralytics id counter to its base (class-level)."""
        from ultralytics.trackers.track_tracker import TTSTrack
        TTSTrack.reset_id()

    def _reset_session(self, sess):
        """Clear ONE session's tracking state (tapnext semantics).

        "Reset" means *start over*: the next call adopts a fresh tracker,
        and ANY fresh tracker starts its numbering from the base (see
        ``_attach_tracker``) — so the caller's next objects get ids 1, 2, …
        regardless of what other sessions have done.  Track ids are
        per-session namespaces: if another live session later assigns a
        number this one used, that is an expected cross-session overlap,
        not a bug — ids carry no meaning outside their own session (the
        same reason tapnext's per-session counters restart on every reset).
        """
        sess.tracker = None             # next call adopts a fresh tracker
        sess.frames = 0

    def _attach_tracker(self, predictor, sess):
        """Give ``predictor`` the session's tracker for the upcoming track()
        call (new session: none, so ultralytics creates one we then adopt)."""
        if sess.tracker is None:
            # A FRESH tracker for this session (new key / regenerate / reload
            # / right after reset): rewind the shared id counter before it
            # is adopted, so its first objects are numbered from the base.
            # Ultralytics' counter is class-level (shared by all tracker
            # instances); without this, a new session would continue where
            # all other sessions left off — the "new session but high ids"
            # surprise.  (Box default tracker: TRACKTRACK -> TTSTrack/
            # BaseTrack counter; the box does not expose other tracker types.)
            try:
                self._rewind_tracker_counter()
            except Exception:
                logging.exception("id-counter rewind failed")
            if predictor is not None and hasattr(predictor, "trackers"):
                try:
                    del predictor.trackers
                except AttributeError:
                    pass
            return
        if predictor is not None:
            predictor.trackers = [sess.tracker]
            # Replicate ultralytics' fresh-creation setup once per predictor
            # (it is skipped when on_predict_start early-returns on reuse).
            if id(predictor) not in self._tracker_setups:
                try:
                    setter = getattr(type(sess.tracker), "setup_predictor", None)
                    if setter is not None:
                        setter(predictor)
                    self._tracker_setups.add(id(predictor))
                except Exception:
                    logging.exception("tracker setup_predictor failed")

    def _track_frames(self, frames, spec, sess, device):
        """One tracking pass over ``frames`` (in order; per-frame Results).

        Ultralytics forces batch=1 in track mode, so every frame is one
        sequential forward + one tracker.update — the tracker state (owned
        by ``sess``) carries the track ids across frames AND across calls.
        """
        with self._track_lock:
            predictor = self._model.predictor
            self._attach_tracker(predictor, sess)
            if os.getenv("YOLO_DBG"):
                logging.info("DBG-ATTACH pct=%s sess=%s tracker=%s n_ses=%d",
                             id(predictor), sess, id(sess.tracker), len(self._sessions))

            results = self._model.track(
                frames,
                persist=True,          # reuse the attached tracker, never re-create
                conf=spec["conf"],
                iou=spec["iou"],
                imgsz=spec["imgsz"],
                max_det=spec["max_det"],
                classes=spec["classes"],
                device=device,
                verbose=False,
            )

            # Adopt the tracker this call actually used: a fresh one for a
            # new session, or the surviving one if the model swapped a
            # predictor under us (first GPU call / weights switch).
            live = self._model.predictor
            live_trackers = getattr(live, "trackers", None) if live is not None else None
            if live_trackers:
                sess.tracker = live_trackers[0]
            if os.getenv("YOLO_DBG"):
                logging.info("DBG-ADOPT live=%s live_is_same=%s live_trackers=%s sess_trk=%s",
                             id(live), live is predictor,
                             [id(t) for t in (live_trackers or [])], id(sess.tracker))
            return results

    # ---------------------------------------------------------------- params
    @staticmethod
    def _num(parameters, key, cast=float):
        if key not in parameters:
            return cast(_DEFAULTS[key])
        try:
            v = cast(parameters[key])
        except (TypeError, ValueError) as e:
            raise ValueError(f"parameters.{key} must be a number, got {parameters[key]!r}") from e
        if v < 0:
            raise ValueError(f"parameters.{key} must be >= 0, got {v}")
        return v

    @staticmethod
    def _boolish(v, key):
        """True/False for JSON booleans and the string forms the webui
        select produces (``"true"`` / ``"false"``)."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        if isinstance(v, (int, float)):
            return bool(v)
        raise ValueError(f"parameters.{key} must be true/false, got {v!r}")

    def _parse_parameters(self, parameters):
        """Validate + coerce user parameters into a clean inference spec."""
        classes = parameters.get("classes")
        if classes is not None:
            if not isinstance(classes, (list, tuple)) or not all(
                    isinstance(c, (int, float, str)) for c in classes):
                raise ValueError("parameters.classes must be a list of class ids")
            try:
                classes = [int(float(c)) for c in classes]
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"parameters.classes must be a list of (stringable) ints, got {classes!r}") from e

        return {
            "conf": min(1.0, self._num(parameters, "conf")),
            "iou": min(1.0, self._num(parameters, "iou")),
            "imgsz": int(self._num(parameters, "imgsz", int)),
            "max_det": int(self._num(parameters, "max_det", int)),
            "classes": classes,
            "save_annotated": self._boolish(
                parameters.get("save_annotated", _DEFAULTS["save_annotated"]), "save_annotated"),
            "frame_step": max(1, int(self._num(parameters, "frame_step", int))),
            "max_frames": max(1, int(self._num(parameters, "max_frames", int))),
        }

    # ---------------------------------------------------------------- inputs
    @staticmethod
    def _decode_images(image_bytes_list):
        frames = []
        for i, raw in enumerate(image_bytes_list):
            arr = np.frombuffer(bytes(raw), dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"could not decode image #{i + 1} of "
                                 f"{len(image_bytes_list)} (not a supported raster?)")
            frames.append(frame)
        return frames

    def _decode_video(self, video_bytes, frame_step, max_frames):
        """Server-side decode: (sampled BGR frames, their original indices,
        total frame count, source fps).  The total comes from the container
        metadata (``CAP_PROP_FRAME_COUNT``) when the codec provides it, else
        falls back to the number of frames actually decoded before the
        ``max_frames`` cap."""
        suffix = _sniff_video_ext(video_bytes)
        tmp_path = None
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp_path = tmp.name
            tmp.write(video_bytes)
            tmp.close()

            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                raise ValueError(f"cv2.VideoCapture could not open the video "
                                 f"(sniffed container {suffix!r} — unsupported codec?)")

            try:
                meta_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            except (TypeError, ValueError):
                meta_total = 0
            try:
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            except (TypeError, ValueError):
                fps = 0.0
            if not (fps > 0) or fps != fps:      # NaN guard
                fps = 30.0

            frames, indices = [], []
            decoded = 0
            while len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if decoded % frame_step == 0:
                    frames.append(frame)
                    indices.append(decoded)
                decoded += 1
            cap.release()
            return frames, indices, (meta_total if meta_total > 0 else decoded), fps
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def _build_annotated_zip(
        annotated: list,     # list of JPEG bytes, in frame order
        detections_json: str,
        manifest: dict,
    ) -> bytes:
        """Bundle annotated frames + detections + manifest into a single ZIP.

        Layout:
          frame_0001.jpg, frame_0002.jpg, …   (zero-padded, ordered)
          detections.json                      (same payload as data.detections)
          manifest.json                        (model, params, runtime, etc.)
        """
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            width = max(len(str(len(annotated))), 1)
            for idx, img_bytes in enumerate(annotated, start=1):
                name = f"frame_{str(idx).zfill(width)}.jpg"
                z.writestr(name, bytes(img_bytes))
            z.writestr("detections.json", detections_json)
            z.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        return buf.getvalue()

    @staticmethod
    def _encode_mp4(frames, fps):
        """Encode BGR frames into one MP4, preferring the **browser-playable
        H.264** (PyAV bundles the FFmpeg libraries incl. libx264).  Degrades
        to ``mp4v`` (MPEG-4 Part 2, plays in VLC but NOT in HTML5 <video>)
        when PyAV is missing or the encode fails.  Returns ``(bytes, codec)``
        or ``(None, None)"."""
        if not frames:
            return None, None
        h, w = frames[0].shape[:2]
        # H.264 needs even dimensions — pad by one black pixel when odd.
        if w % 2 or h % 2:
            w2, h2 = w + w % 2, h + h % 2
            frames = [cv2.copyMakeBorder(f, 0, h2 - h, 0, w2 - w,
                                         cv2.BORDER_CONSTANT, value=0)
                      for f in frames]
        else:
            w2, h2 = w, h

        # --- preferred: H.264 via PyAV (browser <video> support) ----------
        try:
            import av
        except ImportError:
            av = None
        if av is not None:
            tmp_path = None
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                tmp_path = tmp.name
                tmp.close()
                rate = max(1, int(round(fps))) if fps and fps > 0 else 30
                container = av.open(tmp_path, mode="w")
                # Speed knobs: an annotated preview should not take as long to
                # encode as it took to infer. ``preset=veryfast`` +
                # ``tune=zerolatency`` trades ~3-5% quality (irrelevant for a
                # preview of annotated frames) for ~3x encode speed, and in
                # practice produces SMALLER files than the default ``medium``
                # preset because it skips lookahead/GOP buffering.  ``crf=23``
                # is the x264 default and fine for 720p-1080p previews.
                stream = container.add_stream("libx264", rate=rate, options={
                    "preset": "veryfast",
                    "tune": "zerolatency",
                    "crf": "23",
                })
                stream.width, stream.height, stream.pix_fmt = w2, h2, "yuv420p"
                for frame in frames:
                    vframe = av.VideoFrame.from_ndarray(frame, format="bgr24")
                    for pkt in stream.encode(vframe):
                        container.mux(pkt)
                for pkt in stream.encode(None):      # flush
                    container.mux(pkt)
                container.close()
                with open(tmp_path, "rb") as f:
                    data = f.read()
                return (data, "h264") if data else (None, None)
            except (OSError, ValueError) as e:
                logging.warning("h264 encode failed (%s); falling back to mp4v", e)
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        # --- fallback: mp4v via OpenCV (non-browser codecs are fine here) --
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp_path = tmp.name
            tmp.close()
            writer = cv2.VideoWriter(
                tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps if fps and fps > 0 else 30.0, (w2, h2))
            if not writer.isOpened():
                return None, None
            for frame in frames:
                writer.write(frame)
            writer.release()
            with open(tmp_path, "rb") as f:
                data = f.read()
            return (data, "mp4v") if data else (None, None)
        except (OSError, cv2.error):
            logging.exception("annotated_video encoding failed")
            return None, None
        finally:
            try:
                os.unlink(tmp_path)
            except (OSError, NameError):
                pass

    # -------------------------------------------------------------- inference
    def _detect_frame(self, r, frame, frame_index):
        """One frame's Result -> a plain-JSON detection record.

        The box always runs in track mode, so ``boxes`` carries ``.id``
        (per-session tracker ids); a frame with zero boxes has an empty
        box list and ``track_id: null``."""
        xyxy = labels = class_ids = scores = track_ids = []
        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes):
            names = self._model.names
            xyxy = np.round(boxes.xyxy.cpu().numpy(), 2).tolist()
            class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
            labels = [names[c] for c in class_ids]
            scores = np.round(boxes.conf.cpu().numpy(), 4).tolist()
            if getattr(boxes, "id", None) is not None:
                track_ids = [int(t) for t in boxes.id.cpu().numpy().tolist()]
        h, w = frame.shape[:2]
        return {
            "frame_index": frame_index,   # original index in the source (video frame nº / list pos)
            "width": int(w),
            "height": int(h),
            "boxes": xyxy,                # [x1, y1, x2, y2] in input-frame pixels
            "class_ids": class_ids,
            "labels": labels,
            "scores": scores,
            "track_id": track_ids if track_ids else None,   # per-box session tracker id
        }

    # ---------------------------------------------------------------- Process
    def Process(self, request, context):
        start_time = time.time()
        self._last_request_time = time.time()  # keep the idle watchdog honest

        try:
            if not request.config_json:
                return _error("No config JSON")

            config = json.loads(request.config_json)
            yolo_config = config.get("yolo", {})
            if not isinstance(yolo_config, dict):
                return _error("config section 'yolo' must be an object")
            parameters = yolo_config.get("parameters", {}) or {}
            if not isinstance(parameters, dict):
                return _error("parameters must be an object")

            # Multi-session (tapnext contract): session_id at the section
            # top level; omitted -> the shared "default" session.
            sid = str(yolo_config.get("session_id") or _DEFAULT_SESSION)

            # Reset is scoped to THIS session — other sessions untouched.
            if yolo_config.get("command") == "reset" or parameters.get("reset"):
                sess = self._get_session(sid)
                self._reset_session(sess)
                return pipeline_pb2.Envelope(
                    config_json=json.dumps(
                        {"yolo": {"status": "done", "action": "reset", "session": sid}}))

            if yolo_config.get("command") == "list":
                # Operator view: active sessions only (no state content).
                with self._sessions_lock:
                    now = time.time()
                    listing = [
                        {
                            "session": k,
                            "frames_processed": s.frames,
                            "has_tracker": s.tracker is not None,
                            "idle_seconds": round(now - s.last_used, 1),
                        }
                        for k, s in self._sessions.items()
                    ]
                return pipeline_pb2.Envelope(
                    config_json=json.dumps(
                        {"yolo": {"status": "done", "action": "list",
                                  "sessions": listing}}))

            try:
                spec = self._parse_parameters(parameters)
            except ValueError as e:
                return _error(str(e))

            image_bytes_list = unwrap_value(request.data["images"]) if "images" in request.data else None
            if image_bytes_list is not None and not isinstance(image_bytes_list, list):
                return _error("data.images must be a list of image bytes")
            video_bytes = unwrap_value(request.data["video"]) if "video" in request.data else None

            if image_bytes_list and video_bytes:
                return _error("send either data.images or data.video, not both")
            if not image_bytes_list and not video_bytes:
                return pipeline_pb2.Envelope(
                    config_json=json.dumps(
                        {"yolo": {"status": "empty_request", "session": sid}}))

            # Weights: resolved per call — explicit parameters.weights or
            # the startup default.  A weights switch swaps the YOLO model;
            # session trackers attach to whichever predictor serves the
            # next call (they outlive the model instance).  Fresh
            # checkpoints download on first use.
            weights = str(parameters.get("weights") or "").strip() \
                or self._default_weights
            try:
                self._activate_weights(weights)
            except Exception as e:
                logging.exception("failed to load weights %r", weights)
                return _error(f"failed to load weights {weights!r}: {e}")

            # GPU lifecycle: CPU at startup -> lazy move on request -> idle
            # watchdog fallback (see _place_model / _watchdog_loop).
            device = self._place_model(parameters)

            source = "images"
            extra_status = {}
            fps = 0.0
            if video_bytes:
                frames, frame_indices, total_frames, fps = self._decode_video(
                    video_bytes, spec["frame_step"], spec["max_frames"])
                source = "video"
                extra_status = {
                    "frames_in_video": total_frames,
                    "frame_step": spec["frame_step"],
                }
                if not frames:
                    return _error("no frames could be decoded from data.video")
            else:
                frames = self._decode_images(image_bytes_list)
                frame_indices = list(range(len(frames)))

            # Tracking pass: every call runs track mode (track ids are a
            # byproduct of detection, not a separate mode).  Ultralytics does
            # the frames sequentially (it forces batch=1 in track mode) and
            # advances the session's tracker over them, so VRAM stays bounded
            # per frame and ids stay stable across calls in this session.
            sess = self._get_session(sid)
            results = self._track_frames(frames, spec, sess, device)
            if not isinstance(results, list) or len(results) != len(frames):
                n = len(results) if isinstance(results, list) else type(results).__name__
                return _error(f"tracking returned {n} results for "
                              f"{len(frames)} frames (internal error)")

            detections = [self._detect_frame(results[i], frames[i], frame_indices[i])
                          for i in range(len(frames))]
            num_detections = sum(len(d["boxes"]) for d in detections)
            num_tracks = len({t for d in detections for t in (d["track_id"] or [])})
            sess.frames += len(frames)

            response_data = {
                "detections": wrap_value(json.dumps(detections).encode("utf-8")),
            }
            encoding = {"detections": "json"}

            if spec["save_annotated"]:
                annotated_bgr, annotated = [], []
                for r in results:
                    # ``r.plot()`` is BGR: ultralytics draws its annotations
                    # on the *original* frame, which we decoded BGR with
                    # cv2. Keep it BGR — ``cv2.imencode`` and the mp4v
                    # ``VideoWriter`` both expect BGR, and the H.264 path
                    # does its own BGR->RGB conversion before x264.  Converting
                    # *here* (as before) double-swapped channels and put
                    # red/blue the wrong way round in the JPEGs and video.
                    bgr_frame = np.ascontiguousarray(r.plot())
                    annotated_bgr.append(bgr_frame)
                    ok, buf = cv2.imencode(".jpg", bgr_frame,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    if ok:
                        annotated.append(buf.tobytes())
                response_data["annotated"] = wrap_value(annotated)
                encoding["annotated"] = "identity"
                if source == "video":
                    # A video in -> a real video out: the same annotated
                    # frames re-encoded as one mp4 (source fps).  H.264 when
                    # PyAV is available (browser <video>), else mp4v fallback.
                    out_video, out_codec = self._encode_mp4(annotated_bgr, fps)
                    if out_video:
                        response_data["annotated_video"] = wrap_value(out_video)
                        encoding["annotated_video"] = "identity"
                        extra_status["annotated_video_codec"] = out_codec

                # Single downloadable ZIP: all annotated frames + detections JSON + manifest.
                # Lets the user grab everything in one click without iterating.
                manifest = {
                    "box": "yolo",
                    "weights": self._current_weights,
                    "source": source,
                    "session_id": sid,
                    "tracked": True,
                    "num_frames": len(frames),
                    "num_detections": num_detections,
                    "num_tracks": num_tracks,
                    "runtime_seconds": round(time.time() - start_time, 3),
                    "parameters": {k: v for k, v in parameters.items()
                                   if k not in ("images", "video")},
                    "frame_indices": frame_indices,
                }
                detections_json_bytes = response_data["detections"].b  # raw JSON bytes
                try:
                    zip_bytes = self._build_annotated_zip(
                        annotated, detections_json_bytes.decode("utf-8"), manifest)
                    response_data["annotated_zip"] = wrap_value(zip_bytes)
                    encoding["annotated_zip"] = "identity"
                except Exception as e:
                    logging.warning("annotated_zip build failed: %s", e)

            return pipeline_pb2.Envelope(
                config_json=json.dumps({
                    "yolo": {
                        "status": "done",
                        "session": sid,
                        "weights": self._current_weights,
                        "runtime": time.time() - start_time,
                        "source": source,
                        "num_frames": len(frames),
                        "frames_sampled": len(frames),
                        "num_detections": num_detections,
                        "tracked": True,
                        "num_tracks": num_tracks,
                        # Declared payload encoding (generic visionist_client
                        # contract): JSON detection records, raw JPEG bytes.
                        "encoding": encoding,
                        **extra_status,
                    }
                }),
                data=response_data,
            )

        except Exception as e:
            logging.exception(f"Error in Process: {e}")
            return _error(str(e))


def get_port():
    """Parse the port where the server should listen.

    Exits the program if the environment variable is not a positive int.

    Returns:
        The port where the server should listen, or None if an error occurred.
    """
    try:
        server_port = int(os.getenv(_PORT_ENV_VAR, _PORT_DEFAULT))
        if server_port <= 0:
            logging.error('Port should be greater than 0')
            return None
        return server_port
    except ValueError:
        logging.exception('Unable to parse port')
        return None


def run_server(server):
    """Run the given server on the port defined by the environment variables
    or the default port if it is not defined."""
    port = get_port()
    if not port:
        return

    target = f'[::]:{port}'
    server.add_insecure_port(target)
    server.start()
    logging.info(f'''Server started at {target}''')
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    import grpc
    import grpc_reflection.v1alpha.reflection as grpc_reflection

    logging.basicConfig(
        format='[ %(levelname)s ] %(asctime)s (%(module)s) %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=serve.max_workers()),
        options=serve.server_options(),
        maximum_concurrent_rpcs=serve.max_concurrent_rpcs())

    pipeline_pb2_grpc.add_PipelineServiceServicer_to_server(PipelineService(), server)

    service_names = (
        pipeline_pb2.DESCRIPTOR.services_by_name['PipelineService'].full_name,
        grpc_reflection.SERVICE_NAME
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)
