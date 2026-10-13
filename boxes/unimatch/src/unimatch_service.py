"""unimatch box — dense optical flow / stereo disparity / metric depth.

A standard shared-envelope box (one RPC: ``Process``) wrapping the UniMatch
model family (https://github.com/autonomousvision/unimatch, TPAMI'23): one
CNN-Transformer network used for three dense-estimation tasks.

Commands (``config_json`` section ``{"unimatch": {"command", "parameters"}}``):

* **``flow``** (default when no command is given) — exactly 2 images,
  forward optical flow from image 1 to image 2 -> ``flow`` [H, W, 2]
  float32.  Optional (``parameters``): ``pred_bidir_flow`` -> also
  ``bwd_flow``; ``fwd_bwd_check`` -> also ``occ_fwd``/``occ_bwd``
  binary occlusion masks (their alpha/beta thresholds are
  ``occ_alpha``/``occ_beta``).
* **``stereo``** — exactly 2 rectified images (left, right) ->
  ``disparity`` [H, W] float32, in pixels of the left image.
* **``depth``** — 1 image (monocular, identity pose) or a reference/target
  pair -> metric ``depth`` [H, W] float32.  Needs ``parameters.intrinsics``
  (3x3, 4x4, or ``[fx, fy, cx, cy]`` at the sent image resolution — it is
  scaled to the model's padded input automatically) and either the relative
  pose ``parameters.pose`` (4x4, world-referenced, ref -> target, following
  the repo demos: ``inv(pose_tgt) @ pose_ref``) or both
  ``parameters.pose_ref``/``parameters.pose_tgt`` (4x4 world-referenced,
  the box computes the relative pose).  Depth range/candidates:
  ``min_depth``/``max_depth`` (m, default 0.5/10),
  ``num_depth_candidates`` (default 64).
* **``reset``** — stateless box: standard no-op, acknowledged for the
  shared contract.

Shared ``parameters``: ``device`` (default: CUDA if visible, else CPU),
``model`` (checkpoint file name in ``PRETRAINED_DIR``, a local path, or an
http(s) URL — downloaded once), ``padding_factor`` (default 16),
``inference_size`` (``[h, w]``, optional; default is the nearest multiple
of ``padding_factor``), plus the architecture/forward knobs of the
checkpoint family (``num_scales``, ``attn_type``, ``attn_splits_list``,
``corr_radius_list``, ``prop_radius_list``, ``reg_refine``,
``num_reg_refine``, ...).  Defaults match the scale1 zoo models (lengths of
the per-scale lists must equal ``num_scales``).

Weights: one checkpoint per task is expected under ``PRETRAINED_DIR``
(env, default ``./pretrained``).  The image ships the flow model
(``gmflow-scale1-mixdata-train320x576-4c3a6e9a.pth``); ``stereo``/``depth``
answer with a clear status naming the expected default file when it is not
present yet.

Contract (see docs/gRPC_Services_Reference.md):

* response ``config_json`` is namespaced: ``{"unimatch": {"status": …}}``
  with ``status`` in ``done | empty_request | error``; ``encoding``
  declares every array output as ``numpy`` (``.npy`` blobs — decoded to
  ``np.ndarray`` by ``visionist_client``).

GPU lifecycle (fleet convention): the model lives on ``parameters.device``
(default: CUDA if visible, else CPU); a watchdog thread parks it on CPU
after ``_IDLE_TIMEOUT`` seconds of inactivity and reclaims VRAM.
"""

import concurrent.futures as futures
import io
import json
import logging
import os
import sys
import threading
import time

_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_PORT_ENV_VAR = 'PORT'
_IDLE_TIMEOUT = 60  # seconds: park the model on CPU after this much idle time

# Where the unimatch python package lives (cloned at build time).
_UNIMATCH_ROOT = os.getenv("UNIMATCH_ROOT", "./unimatch")
# Where checkpoints are looked up / downloaded.
_PRETRAINED_DIR = os.getenv("PRETRAINED_DIR", "./pretrained")
sys.path.insert(0, _UNIMATCH_ROOT)

sys.path.append("./protos")
import pipeline_pb2 as unimatch_pb2  # noqa: E402
import pipeline_pb2_grpc as unimatch_pb2_grpc  # noqa: E402
from aux import wrap_value, unwrap_value
import serve  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# Default checkpoint per task (file names, from MODEL_ZOO.md).
# Only the flow one is baked into the image by default.
_DEFAULT_WEIGHTS = {
    "flow":   "gmflow-scale1-mixdata-train320x576-4c3a6e9a.pth",
    "stereo": "gmstereo-scale1-sceneflow-124a438f.pth",
    "depth":  "gmdepth-scale1-scannet-d3d1efb5.pth",
}

# UniMatch architecture defaults — the scale1 zoo variants.
# `num_scales`/`reg_refine` must agree with the loaded checkpoint
# (see the box README table of supported `parameters`).
_ARCH_DEFAULTS = {
    "num_scales": 1,
    "feature_channels": 128,
    "upsample_factor": 8,
    "num_head": 1,
    "ffn_dim_expansion": 4,
    "num_transformer_layers": 6,
    "reg_refine": False,
}
_FORWARD_DEFAULTS = {
    "attn_type": "swin",
    "attn_splits_list": [2],
    "corr_radius_list": [-1],
    "prop_radius_list": [-1],
    "num_reg_refine": 1,
}
_DEFAULT_PADDING_FACTOR = 16

# stereo/depth normalize the pair themselves ((x/255 - mean)/std);
# flow normalizes internally.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Depth candidates (inverse-depth range), their evaluate_depth defaults.
_DEPTH_MIN = 0.5
_DEPTH_MAX = 10.0
_DEPTH_CANDIDATES = 64


def torch_device_default():
    """CUDA if visible, else CPU."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# Automatic input cap: the global-attention stages are O((HW)^2), so a box
# must not accept arbitrarily large inputs on a shared GPU (a 2840x1984 stereo
# pair tried to allocate ~29 GB for one matmul buffer and OOM'd).  Requests
# larger than this are downscaled automatically AND reported in the response
# config (`auto_resized`); an explicit `parameters.inference_size` always wins.
_MAX_INPUT_AREA = int(os.getenv("UNIMATCH_MAX_INPUT_AREA", "1600000"))  # ~1.6 MP


def _done(extra, data=None, encoding=None):
    """Standard success envelope: namespaced status + declared encoding."""
    section = {"status": "done", **extra}
    if encoding:
        section["encoding"] = encoding
    return unimatch_pb2.Envelope(config_json=json.dumps({"unimatch": section}),
                                 data=data or {})


def _empty_request():
    return unimatch_pb2.Envelope(
        config_json=json.dumps({"unimatch": {"status": "empty_request"}}))


def _error(message):
    return unimatch_pb2.Envelope(
        config_json=json.dumps(
            {"unimatch": {"status": "error", "error": message}}))


def np_to_bytes(arr: np.ndarray) -> bytes:
    """Serialize with np.save — dtype and shape are kept in the blob, so
    the client's ``numpy`` codec (``np.load``) restores them exactly."""
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr))
    return buf.getvalue()


def _num(parameters, key, default, cast, minimum=None):
    v = parameters.get(key, default)
    try:
        v = cast(v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"parameters.{key} must be a number, got {v!r}") from e
    if minimum is not None and v < minimum:
        raise ValueError(f"parameters.{key} must be >= {minimum}, got {v}")
    return v


def _int_list(parameters, key, default, minimum=None):
    v = parameters.get(key, default)
    if not isinstance(v, (list, tuple)) or not v:
        raise ValueError(f"parameters.{key} must be a non-empty list of ints")
    out = []
    for x in v:
        try:
            ix = int(x)
        except (TypeError, ValueError) as e:
            raise ValueError(f"parameters.{key} must be a list of ints") from e
        if minimum is not None and ix < minimum:
            raise ValueError(f"parameters.{key} values must be >= {minimum}")
        out.append(ix)
    return out


def _load_intrinsics(parameters):
    """``parameters.intrinsics`` -> np.float64 [3, 3] (or None).

    Accepted: 3x3 matrix, 4x4 matrix, or [fx, fy, cx, cy]."""
    if "intrinsics" not in parameters or parameters["intrinsics"] is None:
        return None
    try:
        m = np.asarray(parameters["intrinsics"], dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"parameters.intrinsics is malformed: {e}") from e
    if m.shape == (3, 3):
        return m
    if m.shape == (4, 4):
        return m[:3, :3]
    m = m.reshape(-1)
    if m.size == 4:
        fx, fy, cx, cy = m
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    raise ValueError("parameters.intrinsics must be 3x3, 4x4, or "
                     f"[fx, fy, cx, cy] (got shape {m.shape})")


def _load_pose4(parameters, key):
    """A 4x4 matrix parameter -> np.float64 [4, 4] (or None if absent)."""
    if key not in parameters or parameters[key] is None:
        return None
    m = np.asarray(parameters[key], dtype=np.float64).reshape(-1)
    if m.size != 16:
        raise ValueError(f"parameters.{key} must be a 4x4 matrix "
                         f"(16 values; got {m.size})")
    return m.reshape(4, 4)


# ---------------------------------------------
# Service Definition
# ---------------------------------------------
class PipelineService(unimatch_pb2_grpc.PipelineServiceServicer):
    def __init__(self):
        self._models = {}           # (weight_path, task) -> UniMatch
        self._device = "cpu"        # where loaded models currently live
        self._models_lock = threading.Lock()
        self._last_request_time = time.time()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop,
                                                 daemon=True)
        self._watchdog_thread.start()
        logging.info(f"unimatch box ready (unimatch root: {_UNIMATCH_ROOT}, "
                     f"pretrained dir: {_PRETRAINED_DIR})")

    # ------------------------------------------------------------- device
    def _watchdog_loop(self):
        """Park idle models on CPU and release allocator cache.

        Reads the *actual* device of the loaded models (never a possibly-
        stale flag) and is wrapped so a single bad iteration can never kill
        the daemon thread (a CUDA error here used to silently end it)."""
        import torch
        while True:
            time.sleep(10)
            try:
                with self._models_lock:
                    if not self._models:
                        continue
                    if time.time() - self._last_request_time <= _IDLE_TIMEOUT:
                        continue
                    on_cuda = any(
                        next(m.parameters()).device.type == "cuda"
                        for m in self._models.values())
                    if on_cuda:
                        logging.info("Idle timeout reached: moving models to CPU")
                        for m in self._models.values():
                            try:
                                m.to("cpu")
                            except Exception:
                                logging.exception("park-to-CPU move failed")
                    self._device = "cpu"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                logging.exception("watchdog iteration failed")

    # ------------------------------------------------------------ weights
    def _resolve_weight(self, task, model_param):
        """Locate (or download once) the checkpoint for ``task``.

        ``model_param``: a file name in ``PRETRAINED_DIR``, an existing
        local path, or an http(s) URL downloaded into it."""
        import urllib.request

        name = model_param or _DEFAULT_WEIGHTS[task]
        if str(name).startswith(("http://", "https://")):
            os.makedirs(_PRETRAINED_DIR, exist_ok=True)
            target = os.path.join(_PRETRAINED_DIR,
                                  os.path.basename(str(name).split("?")[0]))
            if not os.path.exists(target):
                logging.info(f"Downloading weights {name} -> {target}")
                urllib.request.urlretrieve(name, target)
            return target

        path = str(name) if os.path.isabs(str(name)) or os.sep in str(name) \
            else os.path.join(_PRETRAINED_DIR, str(name))
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no checkpoint for task '{task}': expected "
                f"'{_DEFAULT_WEIGHTS[task]}' under {os.path.abspath(_PRETRAINED_DIR)} "
                f"or parameters.model. Pass parameters.model with a file "
                f"name in the pretrained dir, an existing local path, or a "
                f"checkpoint URL (table in the box README).")
        return path

    def _get_model(self, weight_path, task, num_scales, reg_refine, device):
        """Load (once) the UniMatch model for ``weight_path`` on ``device``.

        The architecture (``num_scales``/``reg_refine``) must match the
        checkpoint family; ``task`` is part of the key because the optional
        refinement head differs per task."""
        import torch

        key = (weight_path, num_scales, bool(reg_refine), task)
        with self._models_lock:
            self._last_request_time = time.time()
            model = self._models.get(key)
            if model is None:
                from unimatch.unimatch import UniMatch
                logging.info(f"Loading UniMatch weights: {weight_path} "
                             f"(task={task})")

                try:  # torch >= 2.6 defaults to weights_only=True
                    checkpoint = torch.load(weight_path, map_location="cpu",
                                            weights_only=True)
                except Exception:
                    checkpoint = torch.load(weight_path, map_location="cpu")

                state = checkpoint.get("model", checkpoint) \
                    if isinstance(checkpoint, dict) else checkpoint
                model = UniMatch(
                    num_scales=num_scales,
                    feature_channels=_ARCH_DEFAULTS["feature_channels"],
                    upsample_factor=_ARCH_DEFAULTS["upsample_factor"],
                    num_head=_ARCH_DEFAULTS["num_head"],
                    ffn_dim_expansion=_ARCH_DEFAULTS["ffn_dim_expansion"],
                    num_transformer_layers=_ARCH_DEFAULTS[
                        "num_transformer_layers"],
                    reg_refine=bool(reg_refine),
                    task=task,
                ).eval()
                model.load_state_dict(state, strict=True)
                self._models[key] = model
                logging.info(f"UniMatch loaded: {len(state)} tensors")

            current = next(model.parameters()).device
            want = str(device).split(":")[0]
            if current.type != want:
                logging.info(f"Moving UniMatch model to {device}")
                model.to(device)
            self._device = want   # keep the flag honest (watchdog + diagnostics)
            return model

    # ------------------------------------------------------------ inputs
    @staticmethod
    def _decode_images(image_bytes_list):
        frames = []
        for i, raw in enumerate(image_bytes_list):
            arr = np.frombuffer(bytes(raw), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"could not decode image #{i + 1} of "
                                 f"{len(image_bytes_list)} "
                                 f"(not a supported raster?)")
            frames.append(img)
        return frames

    @staticmethod
    def _bgr_to_tensor(img_bgr, device, normalized):
        """BGR uint8 -> float32 [1, 3, H, W] on ``device``.

        ``normalized``: stereo/depth contract — (x/255 - mean)/std, done by
        the caller (the model does not normalize for these tasks)."""
        import torch
        rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        t = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        if normalized:
            mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
            std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
            t = (t / 255. - mean) / std
        return t.to(device)

    @staticmethod
    def _prepare_pair(img0_bgr, img1_bgr, device, parameters,
                      normalized, transpose):
        """Tensors for the model + the geometry bookkeeping.

        Mirrors the repo's inference scripts: ``transpose`` (flow only) puts
        portrait input to landscape on the model side; then both images are
        resized to ``inference_size`` or the nearest multiple of
        ``padding_factor``.

        Returns (t0, t1, model_hw, ori_hw, transposed): ``ori_hw`` is the
        size to restore predictions to (in the model's orientation)."""
        import torch.nn.functional as F

        padding_factor = int(parameters.get("padding_factor",
                                            _DEFAULT_PADDING_FACTOR))
        t0 = PipelineService._bgr_to_tensor(img0_bgr, device, normalized)
        t1 = PipelineService._bgr_to_tensor(img1_bgr, device, normalized)

        transposed = False
        if transpose and t0.size(-2) > t0.size(-1):  # H > W: expect W > H
            t0 = torch.transpose(t0, -2, -1)
            t1 = torch.transpose(t1, -2, -1)
            transposed = True

        ori_hw = (t0.size(-2), t0.size(-1))

        auto_resized = False
        fixed = parameters.get("inference_size")
        if fixed is not None:
            if not (isinstance(fixed, (list, tuple)) and len(fixed) == 2):
                raise ValueError("parameters.inference_size must be "
                                 "[height, width]")
            target = (int(fixed[0]), int(fixed[1]))
        else:
            target = (int(np.ceil(ori_hw[0] / padding_factor)) *
                      padding_factor,
                      int(np.ceil(ori_hw[1] / padding_factor)) *
                      padding_factor)
            # global attention is O((HW)^2): cap the input unless the caller
            # overrode the size explicitly (an explicit size always wins)
            area = int(parameters.get("max_input_area", _MAX_INPUT_AREA))
            if target[0] * target[1] > area:
                s = (area / (target[0] * target[1])) ** 0.5
                target = (max(16, int(target[0] * s // padding_factor *
                                        padding_factor)),
                          max(16, int(target[1] * s // padding_factor *
                                        padding_factor)))
                auto_resized = True

        if target != ori_hw:
            t0 = F.interpolate(t0, size=target, mode="bilinear",
                               align_corners=True)
            t1 = F.interpolate(t1, size=target, mode="bilinear",
                               align_corners=True)
        return t0, t1, target, ori_hw, transposed, auto_resized

    @staticmethod
    def _restore_flow_t(flow_pr, model_hw, ori_hw, transposed):
        """Flow at model-input resolution [N, 2, Hm, Wm] -> [N, 2, H0, W0]
        with the repo's exact undo (bilinear resize + per-axis scale)."""
        import torch.nn.functional as F

        flow_pr = F.interpolate(flow_pr, size=list(ori_hw),
                                mode="bilinear", align_corners=True)
        flow_pr[:, 0] = flow_pr[:, 0] * ori_hw[1] / model_hw[1]
        flow_pr[:, 1] = flow_pr[:, 1] * ori_hw[0] / model_hw[0]
        if transposed:
            flow_pr = torch.transpose(flow_pr, -2, -1)
        return flow_pr

    # ------------------------------------------------------- command: flow
    def _run_flow(self, imgs, parameters, device, model):
        import torch

        pred_bidir = bool(parameters.get("pred_bidir_flow", False))
        fwd_bwd_check = bool(parameters.get("fwd_bwd_check", False))
        if fwd_bwd_check and not pred_bidir:
            raise ValueError("parameters.fwd_bwd_check needs "
                             "parameters.pred_bidir_flow=true")

        t0, t1, model_hw, ori_hw, transposed, auto_resized = \
            PipelineService._prepare_pair(
                imgs[0], imgs[1], device, parameters, normalized=False,
                transpose=True)

        forward = self._forward(model, t0, t1, parameters, task="flow")
        flow_pr = forward["flow_preds"][-1]  # [1,2,Hm,Wm] or [2,2,Hm,Wm]

        if fwd_bwd_check:
            import torch.nn.functional as F
            from unimatch.geometry import forward_backward_consistency_check
            occ_alpha = _num(parameters, "occ_alpha", 0.01, float)
            occ_beta = _num(parameters, "occ_beta", 0.5, float)
            with torch.no_grad():
                occ_fwd, occ_bwd = forward_backward_consistency_check(
                    flow_pr[:1], flow_pr[1:], alpha=occ_alpha, beta=occ_beta)
            occ_fwd = F.interpolate(occ_fwd.unsqueeze(1), size=list(ori_hw),
                                    mode="nearest").squeeze(1)
            occ_bwd = F.interpolate(occ_bwd.unsqueeze(1), size=list(ori_hw),
                                    mode="nearest").squeeze(1)
            if transposed:
                occ_fwd = torch.transpose(occ_fwd, -2, -1)
                occ_bwd = torch.transpose(occ_bwd, -2, -1)

        flow_pr = self._restore_flow_t(flow_pr, model_hw, ori_hw, transposed)

        data = {"flow": flow_pr[0].permute(1, 2, 0).cpu().numpy()
                .astype(np.float32)}
        if pred_bidir:
            data["bwd_flow"] = flow_pr[1].permute(1, 2, 0).cpu().numpy() \
                .astype(np.float32)
        if fwd_bwd_check:
            data["occ_fwd"] = occ_fwd[0].cpu().numpy().astype(np.float32)
            data["occ_bwd"] = occ_bwd[0].cpu().numpy().astype(np.float32)
        if auto_resized:
            data["_auto_resized"] = True
        return data

    # ------------------------------------------------------ command: stereo
    def _run_stereo(self, imgs, parameters, device, model):
        """images: (left, right) -> left-view disparity [H, W]."""
        t0, t1, model_hw, ori_hw, _, auto_resized = \
            PipelineService._prepare_pair(
                imgs[0], imgs[1], device, parameters, normalized=True,
                transpose=False)

        forward = self._forward(model, t0, t1, parameters, task="stereo")
        disp = forward["flow_preds"][-1]  # [1, Hm, Wm] at model input size

        import torch.nn.functional as F
        disp = F.interpolate(disp.unsqueeze(1), size=list(ori_hw),
                             mode="bilinear", align_corners=True).squeeze(1)
        disp = disp * (ori_hw[1] / model_hw[1])
        out = {"disparity": disp[0].cpu().numpy().astype(np.float32)}
        if auto_resized:
            out["_auto_resized"] = True
        return out

    # ------------------------------------------------------- command: depth
    def _run_depth(self, imgs, parameters, device, model):
        intr = _load_intrinsics(parameters)
        if intr is None:
            raise ValueError("command 'depth' needs parameters.intrinsics "
                             "(3x3, 4x4, or [fx, fy, cx, cy] at the sent "
                             "image resolution)")
        return self._run_pair_depth(imgs, intr, parameters, device, model)

    def _run_pair_depth(self, imgs, intr, parameters, device, model):
        import torch
        import torch.nn.functional as F

        monocular = len(imgs) == 1
        if monocular:
            imgs = [imgs[0], imgs[0]]

        pose_ref = _load_pose4(parameters, "pose_ref")
        pose_tgt = _load_pose4(parameters, "pose_tgt")
        pose_rel = _load_pose4(parameters, "pose")
        if pose_rel is None:
            if pose_ref is not None and pose_tgt is not None:
                pose_rel = np.linalg.inv(pose_tgt) @ pose_ref
            else:
                pose_rel = np.eye(4)

        t0, t1, model_hw, ori_hw, _, auto_resized = \
            PipelineService._prepare_pair(
                imgs[0], imgs[1], device, parameters, normalized=True,
                transpose=False)

        # intrinsics at the model's actual input size (image was resampled
        # by the prepare step)
        sw = model_hw[1] / ori_hw[1]
        sh = model_hw[0] / ori_hw[0]
        intr_in = intr.copy()
        intr_in[0, 0] *= sw
        intr_in[2, 0] *= sw
        intr_in[1, 1] *= sh
        intr_in[2, 1] *= sh

        forward = self._forward(
            model, t0, t1, parameters, task="depth",
            extra={
                "intrinsics": torch.from_numpy(
                    intr_in.astype(np.float32)).unsqueeze(0).to(device),
                "pose": torch.from_numpy(
                    pose_rel.astype(np.float32)).unsqueeze(0).to(device),
            },
        )
        depth = forward["flow_preds"][-1]  # [1, Hm, Wm] metric depth

        depth = F.interpolate(depth.unsqueeze(1), size=list(ori_hw),
                              mode="bilinear", align_corners=True).squeeze(1)
        data = {"depth": depth[0].cpu().numpy().astype(np.float32)}
        if monocular:
            data["_monocular"] = np.array([1])
        if auto_resized:
            data["_auto_resized"] = True
        return data

    # -------------------------------------------------------------- common
    def _forward(self, model, img0, img1, parameters, task, extra=None):
        """One model forward with the per-scale lists validated.

        For the scale1 default zoo models these are single-entry lists;
        for scale2 checkpoints pass 2-element lists (see README)."""
        import torch

        num_scales = int(_num(parameters, "num_scales",
                              _ARCH_DEFAULTS["num_scales"], int, minimum=1))
        reg_refine = bool(parameters.get("reg_refine", False))

        splits = _int_list(parameters, "attn_splits_list",
                           _FORWARD_DEFAULTS["attn_splits_list"])
        corr = _int_list(parameters, "corr_radius_list",
                         _FORWARD_DEFAULTS["corr_radius_list"], minimum=-1) \
            if task != "depth" else None
        prop = _int_list(parameters, "prop_radius_list",
                         _FORWARD_DEFAULTS["prop_radius_list"])
        if len(splits) != num_scales or \
           (corr is not None and len(corr) != num_scales) or \
           len(prop) != num_scales:
            raise ValueError(
                f"per-scale lists: attn_splits_list/prop_radius_list"
                f"({'/corr_radius_list' if task != 'depth' else ''}) must all "
                f"have length num_scales ({num_scales}); got "
                f"{len(splits)}/{len(prop)}"
                f"({len(corr) if corr else '-'})")
        if reg_refine and num_scales > 2:
            raise ValueError("num_scales must be <= 2")

        args = {
            "attn_type": parameters.get("attn_type", _FORWARD_DEFAULTS["attn_type"]),
            "attn_splits_list": splits,
            "prop_radius_list": prop,
            "num_reg_refine": int(parameters.get(
                "num_reg_refine", _FORWARD_DEFAULTS["num_reg_refine"])),
            "task": task,
        }
        if corr is not None:
            args["corr_radius_list"] = corr
        if task == "flow":
            args["pred_bidir_flow"] = bool(
                parameters.get("pred_bidir_flow", False))
        if task == "depth":
            args.update({
                "intrinsics": extra["intrinsics"],
                "pose": extra["pose"],
                "min_depth": 1. / _num(parameters, "min_depth", _DEPTH_MIN,
                                       float, minimum=1e-4),
                "max_depth": 1. / _num(parameters, "max_depth", _DEPTH_MAX,
                                       float, minimum=1e-4),
                "num_depth_candidates": int(_num(parameters, "num_depth_candidates",
                                                 _DEPTH_CANDIDATES, int,
                                                 minimum=4)),
                "pred_bidir_depth": bool(
                    parameters.get("pred_bidir_depth", False)),
            })
        with torch.no_grad():
            return model(img0, img1, **args)

    # -------------------------------------------------------------- Process
    def Process(self, request, context):
        start_time = time.time()
        self._last_request_time = time.time()  # keep the idle watchdog honest

        try:
            if not request.config_json:
                return _error("No config JSON")

            config = json.loads(request.config_json)
            if not isinstance(config, dict) or \
                    not isinstance(config.get("unimatch"), dict):
                return _error("config section 'unimatch' missing or "
                              "not an object")
            section = config["unimatch"]
            command = str(section.get("command") or "flow").lower()
            parameters = section.get("parameters", {})
            if not isinstance(parameters, dict):
                return _error("parameters must be an object")

            if command == "reset":
                return _done({"action": "reset"})

            if command not in ("flow", "stereo", "depth"):
                return _error(f"unknown command {command!r} (expected one of: "
                              "flow | stereo | depth | reset)")

            images = unwrap_value(request.data["images"]) \
                if "images" in request.data else None
            if isinstance(images, (bytes, bytearray)):
                images = [images]
            if not images:
                return _empty_request()
            if command in ("flow", "stereo"):
                want = 2
            else:
                want = None
            if want and len(images) != want:
                return _error(f"command {command!r} takes exactly 2 images, "
                              f"got {len(images)}")
            if command == "depth" and len(images) not in (1, 2):
                return _error("command 'depth' takes 1 (monocular) or 2 "
                              f"images (reference, target); got {len(images)}")

            device = torch_device_default()
            req_device = str(parameters.get("device") or "").strip().lower()
            if req_device:
                device = req_device

            imgs = self._decode_images(images)

            weight_path = self._resolve_weight(command,
                                               parameters.get("model"))
            num_scales = int(_num(parameters, "num_scales",
                                  _ARCH_DEFAULTS["num_scales"], int,
                                  minimum=1))
            reg_refine = bool(parameters.get("reg_refine", False))
            model = self._get_model(weight_path, command, num_scales,
                                    reg_refine, device)

            if command == "flow":
                data = self._run_flow(imgs, parameters, device, model)
            elif command == "stereo":
                data = self._run_stereo(imgs, parameters, device, model)
            else:
                data = self._run_depth(imgs, parameters, device, model)

            monocular = data.pop("_monocular", None) is not None
            auto_resized = data.pop("_auto_resized", None) is not None

            data = {key: wrap_value(np_to_bytes(arr)) for key, arr in data.items()}
            return _done(
                {
                    "command": command,
                    "model": os.path.basename(weight_path),
                    "num_images": len(images),
                    **({"monocular": True}
                       if (command == "depth" and monocular) else {}),
                    **({"auto_resized": True} if auto_resized else {}),
                    "device": device,
                    "runtime": time.time() - start_time,
                },
                data=data, encoding={key: "numpy" for key in data})

        except FileNotFoundError as e:
            return _error(str(e))
        except Exception as e:
            logging.exception(f"Error in Process: {e}")
            return _error(str(e))


# ---------------------------------------------
# Server setup
# ---------------------------------------------
def get_port():
    try:
        port = int(os.getenv(_PORT_ENV_VAR, _PORT_DEFAULT))
        if port <= 0:
            logging.error(f"Port should be greater than 0")
            return None
        return port
    except ValueError:
        logging.exception(f"Cannot parse port {_PORT_ENV_VAR}")
        return None


def run_server(server):
    port = get_port()
    if not port:
        return

    target = f'[::]:{port}'
    server.add_insecure_port(target)
    server.start()
    logging.info(f"Server started at {target}")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    import grpc
    import grpc_reflection.v1alpha.reflection as grpc_reflection

    logging.basicConfig(
        format="[ %(levelname)s ] %(asctime)s (%(module)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=serve.max_workers()),
        options=serve.server_options(),
        maximum_concurrent_rpcs=serve.max_concurrent_rpcs())

    unimatch_pb2_grpc.add_PipelineServiceServicer_to_server(
        PipelineService(), server)

    service_names = (
        unimatch_pb2.DESCRIPTOR.services_by_name["PipelineService"].full_name,
        grpc_reflection.SERVICE_NAME,
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)
