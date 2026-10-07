"""SfM box — depth-augmented projective structure-from-motion over the shared
envelope, with MISSING DATA.

Feature tracks + monocular depth -> per-frame camera poses ``[R|t]``, the
3D point cloud and the per-frame affine of the monocular depth
(``Z = d*lambda + o``). Tracks need not be complete. Two solvers
(``parameters.solver``):

  * ``completion`` (default) -- ``projective_reconstruction.py`` (ported from
    ``~/trackers/src/projective_reconstruction.py``): unobserved entries are
    filled by rank-4 matrix completion, RANSAC rejects outlier tracks, then
    factorization + metric upgrade.
  * ``pairs`` -- ``pairs_reconstruction.py``: every consecutive 2-frame
    pair reconstructed with the same pipeline, chained through the shared
    cameras into a warm start, then ONE global pass of the completion
    ping-pong over all frames (rigid factors, ridge on the offsets).
  * ``linear`` -- ``linear_reconstruction.py``: rotations from 2-frame
    solves, an exact linear solve for (t, X), then joint Levenberg-Marquardt
    over (d, o, R, t, X) on a scale-invariant reprojection-style residual,
    Huber IRLS. No completion, no offset anchor.

Two input modes (every array is an ``np.save`` (``.npy``) blob; a
``torch.save`` tensor is accepted too):

  A. ``tracks`` (2F, P) pixel coords, rows ``[u0, v0, u1, v1, ...]``
     + ``depths`` (F, H, W) monocular depth maps
     + ``intrinsics`` (3, 3) shared or (F, 3, 3) per frame
     -> ``build_depth_weighted_matrix`` builds the rays and samples the depths.
  B. ``W_mat`` (3F, P) homogeneous rays ``[x, y, 1]`` per frame
     + ``lambda_mat`` (F, P) monocular depth at each track
     -> straight into the solver.

An entry (frame f, point p) is MISSING when its track coordinate is NaN, it
falls outside the depth map, or its depth sample is non-finite (e.g. invalid
MoGe pixels). The solvers drop under-observed frames/points (and, for
``completion``, RANSAC outliers); ``frame_ids`` / ``point_ids`` map the surviving cameras and
points back to the input rows/columns.

Every array output is an ``np.save`` blob declared ``numpy`` — the client's
``numpy`` codec restores dtype and shape exactly.

The box is stateless and CPU-only; ``reset`` is accepted as a no-op.
"""
import concurrent.futures as futures
import sys
import logging
import os
import time
import json
import io

sys.path.append("./protos")
import pipeline_pb2
import pipeline_pb2_grpc
from aux import wrap_value, unwrap_value

import numpy as np
import torch

import linear_reconstruction as lr
import pairs_reconstruction as pp
import projective_reconstruction as pr


_PORT_ENV_VAR = 'PORT'
_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

_BOX_KEY = "sfm"
_DTYPES = {"float32": torch.float32, "float64": torch.float64}
_OFFSET_MODES = ("normalize", "estimate", "zero")
_SOLVERS = ("completion", "pairs", "linear")

# Solver keyword arguments that may be passed through config.sfm.parameters
# (value -> coercion), per solver.
_RECON_PARAMS = {
    "iters": int,
    "num_scale_iters": int,
    "rank": int,
    "seed": int,
    "offset_mode": str,
    "removal_iters": lambda v: tuple(int(i) for i in v),
    "min_obs": int,
}
_PAIRS_PARAMS = {
    **_RECON_PARAMS,
    "pair_removal_iters": lambda v: tuple(int(i) for i in v),
    "metric": bool,
    "offset_ridge": float,
}
_LINEAR_PARAMS = {
    "iters": int,
    "depth_weight": float,
    "offset_prior": float,
    "loss": lambda v: None if v in (None, "none", "None") else str(v),
    "loss_scale": lambda v: v if v == "auto" else float(v),
    "loss_k": float,
    "min_obs": int,
    "min_frame_obs": int,
    "pair_min_points": int,
    "seed": int,
}

# Below this many points (or 2 frames) the rank-3 factorization is meaningless.
_MIN_FRAMES = 2
_MIN_POINTS_DEFAULT = 8


def np_to_bytes(arr) -> bytes:
    """Serialize with np.save — keeps the array's dtype and shape in the
    blob, so the client's ``numpy`` codec (``np.load``) restores it exactly."""
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr))
    return buf.getvalue()


def bytes_to_array(name: str, blob) -> np.ndarray:
    """``.npy`` blob (preferred) or ``torch.save`` tensor -> float ndarray."""
    blob = bytes(blob)
    try:
        return np.load(io.BytesIO(blob), allow_pickle=False)
    except Exception:
        pass
    try:
        t = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
        if isinstance(t, torch.Tensor):
            return t.numpy()
    except Exception:
        pass
    raise ValueError(f"data.{name}: expected an np.save (.npy) blob or a "
                     f"torch.save tensor ({len(blob)} bytes could not be loaded)")


def _status(**fields):
    return pipeline_pb2.Envelope(config_json=json.dumps({_BOX_KEY: fields}))


def _error(msg):
    return _status(status="error", error=msg)


class PipelineService(pipeline_pb2_grpc.PipelineServiceServicer):

    def Process(self, request, context):
        start_time = time.time()
        try:
            if not request.config_json:
                return _error("No config JSON")

            config = json.loads(request.config_json)
            box_config = config.get(_BOX_KEY, {}) or {}
            parameters = box_config.get("parameters", {}) or {}
            command = box_config.get("command", "reconstruct")

            # Stateless box: accept "reset" (client convenience) as a no-op.
            if command == "reset" or parameters.get("reset"):
                return _status(status="done", action="reset")
            if command != "reconstruct":
                return _error(f"Unknown command {command!r} "
                              f"(known: reconstruct, reset)")

            arrays = {k: bytes_to_array(k, unwrap_value(v))
                      for k, v in request.data.items()
                      if k in ("tracks", "depths", "intrinsics", "W_mat", "lambda_mat")}
            if not arrays:
                return _status(status="empty_request")

            dtype = _DTYPES.get(parameters.get("dtype", "float32"))
            if dtype is None:
                return _error(f"parameters.dtype must be one of {sorted(_DTYPES)}")

            solver = parameters.get("solver", "completion")
            if solver not in _SOLVERS:
                return _error(f"parameters.solver must be one of {list(_SOLVERS)}")

            W_mat, lambda_mat, mode, K_px = self._build_inputs(arrays, parameters, dtype)
            F_in, P_in = lambda_mat.shape
            observed = torch.isfinite(lambda_mat)
            if F_in < _MIN_FRAMES:
                return _error(f"Need at least {_MIN_FRAMES} frames, got {F_in}")

            if solver in ("completion", "pairs"):
                res = self._run_completion(W_mat, lambda_mat, parameters, solver)
            else:
                res = self._run_linear(W_mat, lambda_mat, parameters)
            if isinstance(res, str):
                return _error(res)

            frame_ids, point_ids = res["frame_ids"], res["point_ids"]
            min_points = int(parameters.get("min_points", _MIN_POINTS_DEFAULT))
            if len(frame_ids) < _MIN_FRAMES or len(point_ids) < min_points:
                return _error(f"Only {len(frame_ids)} frames / {len(point_ids)} points "
                              f"survived the reconstruction (min {_MIN_FRAMES} / {min_points})")

            np_dtype = np.float64 if dtype == torch.float64 else np.float32

            def out(a):
                return wrap_value(np_to_bytes(np.asarray(a, dtype=np_dtype)))

            response_data = {
                "cameras": out(res["cameras"]),                 # (F', 3, 4)
                "points": out(res["points"]),                   # (P', 3)
                "frame_ids": wrap_value(np_to_bytes(frame_ids)),
                "point_ids": wrap_value(np_to_bytes(point_ids)),
                "observed": wrap_value(np_to_bytes(res["observed"])),
                "completed_matrix": out(res["completed"]),
                "depth_scales": out(res["d"]),
                "depth_offsets": out(res["o"]),
            }
            reproj = _reprojection_error(W_mat.double().numpy(), res, K_px)

            return pipeline_pb2.Envelope(
                config_json=json.dumps({
                    _BOX_KEY: {
                        "status": "done",
                        "runtime": time.time() - start_time,
                        "solver": solver,
                        "input_mode": mode,
                        "num_frames_in": F_in,
                        "num_points_in": P_in,
                        "missing_in": float(1.0 - observed.float().mean()),
                        "num_frames": int(len(frame_ids)),
                        "num_points": int(len(point_ids)),
                        "missing": float(1.0 - res["observed"].mean()),
                        "reprojection_error": reproj,
                        **res["status"],
                        "depth_model": "Z = depth_scales * lambda + depth_offsets",
                        # Declared payload encoding (generic visionist_client
                        # contract): every field is an np.save blob.
                        "encoding": {k: "numpy" for k in response_data},
                    }
                }),
                data=response_data,
            )

        except ValueError as e:
            logging.warning(f"Bad request: {e}")
            return _error(str(e))
        except Exception as e:
            logging.exception(f"Error in Process: {e}")
            return _error(str(e))

    @staticmethod
    def _run_completion(W_mat, lambda_mat, parameters, solver="completion"):
        table = _RECON_PARAMS if solver == "completion" else _PAIRS_PARAMS
        kwargs = {key: cast(parameters[key])
                  for key, cast in table.items() if key in parameters}
        if kwargs.get("offset_mode", "normalize") not in _OFFSET_MODES:
            return f"parameters.offset_mode must be one of {list(_OFFSET_MODES)}"
        run = (pr.run_projective_reconstruction if solver == "completion"
               else pp.run_pairs_reconstruction)
        with torch.no_grad():
            rec = run(W_mat, lambda_mat, **kwargs)
        # The pipeline factorizes depths (lambda + o) / s. Report them in
        # the multiply-then-add form Z = d * lambda + o' (d = 1/s,
        # o' = o/s): the same depths, in the documented affine model.
        s = rec["current_scales"]
        return {
            "frame_ids": np.flatnonzero(rec["vf"].cpu().numpy()),
            "point_ids": np.flatnonzero(rec["vp"].cpu().numpy()),
            "cameras": torch.stack(rec["cam_lists"]).cpu().numpy(),
            "points": rec["aligned_shape"].t().cpu().numpy(),
            "observed": (rec["mask_f"][0::3] > 0).cpu().numpy(),
            "completed": rec["compl_W_lam"].cpu().numpy(),
            "d": (1.0 / s).cpu().numpy(),
            "o": (rec["offsets"] / s).cpu().numpy(),
            "status": {"iterations": rec["info"]["iterations"],
                       "removed": rec["info"]["removed"],
                       **({"pairs_fallback": [p["frames"] for p in rec["info"]["pairs"]
                                              if p.get("method") != "pair"]}
                          if solver == "pairs" else {})},
        }

    @staticmethod
    def _run_linear(W_mat, lambda_mat, parameters):
        kwargs = {key: cast(parameters[key])
                  for key, cast in _LINEAR_PARAMS.items() if key in parameters}
        if kwargs.get("loss", "huber") not in ("huber", None):
            return "parameters.loss must be 'huber' or null"
        rec = lr.run_linear_reconstruction(W_mat.double().numpy(),
                                           lambda_mat.double().numpy(), **kwargs)
        info = rec["info"]
        return {
            "frame_ids": np.flatnonzero(rec["vf"]),
            "point_ids": np.flatnonzero(rec["vp"]),
            "cameras": rec["cameras"],
            "points": rec["points"],
            "observed": rec["observed"],
            "completed": rec["completed"],
            "d": rec["d"],
            "o": rec["o"],
            "status": {"iterations": info["iterations"],
                       "converged": info["converged"],
                       "outliers": info["outliers"],
                       "pairs_fallback": [p["frames"] for p in info["pairs"]
                                          if p["method"] != "pair"]},
        }

    @staticmethod
    def _build_inputs(arrays, parameters, dtype):
        """Validate the input arrays and return ``(W_mat (3F,P),
        lambda_mat (F,P), mode, K)`` with NaN in ``lambda_mat`` wherever an
        entry is missing (the pipeline's missing-data mask); ``K`` is the
        pixel camera matrix ((3,3) or (F,3,3); None in mode B)."""
        has_a = {"tracks", "depths", "intrinsics"} <= arrays.keys()
        has_b = {"W_mat", "lambda_mat"} <= arrays.keys()

        if has_b:
            W = np.asarray(arrays["W_mat"], dtype=np.float64)
            lam = np.asarray(arrays["lambda_mat"], dtype=np.float64)
            if W.ndim != 2 or W.shape[0] % 3:
                raise ValueError(f"data.W_mat must be (3F, P), got {W.shape}")
            F = W.shape[0] // 3
            if lam.shape != (F, W.shape[1]):
                raise ValueError(f"data.lambda_mat must be (F, P) = {(F, W.shape[1])}, "
                                 f"got {lam.shape}")
            mode, K = "W_mat", None

        elif has_a:
            tracks = np.asarray(arrays["tracks"], dtype=np.float64)
            depths = np.asarray(arrays["depths"], dtype=np.float64)
            K = np.asarray(arrays["intrinsics"], dtype=np.float64)
            if depths.ndim != 3:
                raise ValueError(f"data.depths must be (F, H, W), got {depths.shape}")
            F, H, Wd = depths.shape
            if tracks.ndim != 2 or tracks.shape[0] != 2 * F:
                raise ValueError(f"data.tracks must be (2F, P) = (2*{F}, P), "
                                 f"got {tracks.shape}")
            if K.shape not in ((3, 3), (F, 3, 3)):
                raise ValueError(f"data.intrinsics must be (3, 3) or ({F}, 3, 3), "
                                 f"got {K.shape}")
            if parameters.get("intrinsics_normalized", False):
                # MoGe-style normalized intrinsics: fx, cx in image widths,
                # fy, cy in image heights.
                K = K.copy()
                K[..., 0, :] *= Wd
                K[..., 1, :] *= H

            u, v = tracks[0::2], tracks[1::2]
            seen = (np.isfinite(u) & np.isfinite(v)
                    & (u >= 0) & (u <= Wd - 1) & (v >= 0) & (v <= H - 1))   # (F, P)
            tracks = np.where(np.repeat(seen, 2, axis=0), tracks, 0.0)       # sample safely
            W_t, lam_t = pr.build_depth_weighted_matrix(
                torch.from_numpy(tracks), torch.from_numpy(depths), torch.from_numpy(K))
            W, lam = W_t.numpy(), lam_t.numpy()
            lam[~seen] = np.nan
            mode = "tracks"

        else:
            raise ValueError("Provide either data.tracks + data.depths + data.intrinsics, "
                             f"or data.W_mat + data.lambda_mat (got {sorted(arrays)})")

        # An entry is missing if its ray or its depth is not finite.
        F = lam.shape[0]
        bad = ~(np.isfinite(lam) & np.isfinite(W.reshape(F, 3, -1)).all(1))
        lam = lam.copy(); lam[bad] = np.nan
        W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)   # masked out via lam
        return torch.from_numpy(W).to(dtype), torch.from_numpy(lam).to(dtype), mode, K


def _reprojection_error(W, res, K):
    """Image error of the reconstruction on the observed entries: project
    ``points`` with ``cameras`` and compare with the input rays. In pixels
    when the intrinsics are known (mode A), else in normalized units."""
    fids, pids = res["frame_ids"], res["point_ids"]
    cams = np.asarray(res["cameras"], dtype=np.float64)
    X = np.asarray(res["points"], dtype=np.float64).T                  # (3, P')
    proj = cams[:, :, :3] @ X[None] + cams[:, :, 3:]                    # (F', 3, P')
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = proj[:, :2] / proj[:, 2:]
    w = W.reshape(-1, 3, W.shape[1])[fids][:, :2][:, :, pids]          # (F', 2, P')
    diff = xy - w
    unit = "normalized"
    if K is not None:
        Kf = np.broadcast_to(K, (len(W) // 3, 3, 3))[fids]
        diff = diff * np.stack([Kf[:, 0, 0], Kf[:, 1, 1]], 1)[:, :, None]
        unit = "px"
    err = np.linalg.norm(diff, axis=1)[res["observed"]]
    err = err[np.isfinite(err)]
    if err.size == 0:
        return {"unit": unit}
    return {"median": float(np.median(err)), "mean": float(err.mean()),
            "p90": float(np.percentile(err, 90)), "unit": unit}


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
        futures.ThreadPoolExecutor(),
        options=[
            ('grpc.max_send_message_length', -1),
            ('grpc.max_receive_message_length', -1),
        ]
    )

    pipeline_pb2_grpc.add_PipelineServiceServicer_to_server(PipelineService(), server)

    service_names = (
        pipeline_pb2.DESCRIPTOR.services_by_name['PipelineService'].full_name,
        grpc_reflection.SERVICE_NAME
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)
