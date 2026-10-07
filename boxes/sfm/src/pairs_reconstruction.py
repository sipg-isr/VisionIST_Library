"""The ``pairs`` solver: 2-frame reconstructions chained into a warm start,
then ONE global pass of the completion ping-pong over all frames.

1. every consecutive pair (f, f+1) is reconstructed on its co-visible points
   with ``run_projective_reconstruction`` -- a 2-frame block has no missing
   entries, so this is reliable;
2. the pairs are chained through their SHARED CAMERA: pair k's coordinate
   frame is camera k, so its pose in the global frame is known, and the
   relative scale comes from the depths of frame k in both (median ratio).
   Pair k then gives camera k+1 and its depth affine Z = d*lam + o. This is
   a similarity chain (no 4x4 homography: the cameras stay rigid);
3. the chained cameras and offsets warm-start ``calibrate_with_completion``
   (camera factor U = [R_f | t_f] / d_f, offsets o_f / d_f, the points from
   one ALS step), which then alternates completion and the affine fit over
   ALL frames and observations -- removing the drift of the chain -- before
   the usual scale correction and factorization.

The cold-started ping-pong fails on heavily partial tracks because its start
(column-mean fill) is far off; the pairs fix the start. The global pass runs
with rigid factors (``metric``), a free o_0 and a ridge on the offsets: see
``run_pairs_reconstruction``.
"""
import logging

import torch

import projective_reconstruction as pr

log = logging.getLogger(__name__)


def make_pair_init(pair_kwargs=None, seed=42, info=None):
    """Return the ``init`` callable for ``run_projective_reconstruction``:
    (tracks_f (3F,P), lam_f (F,P), mask_f (3F,P)) -> (U_init (3F,4), o_init (F,)).
    ``info`` (a dict) collects per-pair diagnostics."""
    pair_kwargs = dict(pair_kwargs or {})
    pair_kwargs.setdefault("removal_iters", (20,))
    info = {} if info is None else info

    def init(tracks_f, lam_f, mask_f):
        F = lam_f.shape[0]
        dtype, dev = lam_f.dtype, lam_f.device
        obs = mask_f[0::3].bool()
        R = [torch.eye(3, dtype=dtype, device=dev)]
        t = [torch.zeros(3, dtype=dtype, device=dev)]
        d = torch.ones(F, dtype=dtype, device=dev)
        o = torch.zeros(F, dtype=dtype, device=dev)
        pairs = []

        for k in range(F - 1):
            both = obs[k] & obs[k + 1]
            rows = torch.arange(3 * k, 3 * k + 6, device=dev)
            entry = {"frames": [k, k + 1], "points": int(both.sum())}
            try:
                rec = pr.run_projective_reconstruction(
                    tracks_f[rows][:, both], lam_f[k:k + 2][:, both], seed=seed, **pair_kwargs)
                if not bool(rec["vf"].all()):
                    raise ValueError("a frame of the pair was dropped")
            except Exception as e:                       # keep the chain going
                log.warning(f"pair ({k},{k + 1}) failed: {e}; copying camera {k}")
                entry["method"] = "copy"
                R.append(R[-1].clone()); t.append(t[-1].clone())
                d[k + 1], o[k + 1] = d[k], o[k]
                pairs.append(entry)
                continue

            # pair depths Z = d_p * lam + o_p  (pair internals: (lam + o) / s)
            d_p = 1.0 / rec["current_scales"]
            o_p = rec["offsets"] / rec["current_scales"]
            cam_a, cam_b = rec["cam_lists"]               # cam_a = [I | 0] (pair frame = camera k)
            lam_a = lam_f[k][both][rec["vp"]]
            Z_pair = d_p[0] * lam_a + o_p[0]
            Z_glob = d[k] * lam_a + o[k]
            s = torch.median(Z_glob / Z_pair)              # pair units -> global units

            # world -> cam k: x_k = R_k X + t_k; pair coords are s^-1 * cam-k coords
            R_b, t_b = cam_b[:, :3].to(dtype), cam_b[:, 3].to(dtype)
            R.append(R_b @ R[k])
            t.append(R_b @ t[k] + s * t_b)
            d[k + 1], o[k + 1] = s * d_p[1], s * o_p[1]
            entry.update(method="pair", scale=float(s))
            pairs.append(entry)

        info["pairs"] = pairs
        # ping-pong model: (lam + o_pp) w = Z w / d = [R | t] X / d
        U = torch.cat([torch.cat([R[f], t[f][:, None]], dim=1) / d[f] for f in range(F)])
        o_pp = o / d
        return U, o_pp - o_pp[0]

    return init


def run_pairs_reconstruction(W_mat, lambda_mat, iters=100, num_scale_iters=4, rank=4,
                             seed=42, offset_mode="estimate", removal_iters=(),
                             min_obs=2, pair_removal_iters=(20,), metric=True,
                             offset_ridge=1.0):
    """``run_projective_reconstruction`` warm-started from chained 2-frame
    solves (affine fit from the first iteration). Same return dict, plus
    ``info["pairs"]``.

    Defaults (from the synthetic + cozinha sweeps): ``metric`` keeps the
    completion factors rigid (without it the global pass drifts away from
    the good warm start on banded tracks); ``offset_mode="estimate"`` frees
    o_0 (the o_0 = 0 anchor biases it); ``offset_ridge=1.0`` holds the
    offsets near 0 -- the ping-pong's algebraic affine fit is not
    scale-invariant and, on real small-baseline video, runs the offsets off
    to a degenerate solution without it (at the price of a bias where the
    true offsets are large)."""
    pair_info = {}
    init = make_pair_init({"removal_iters": tuple(pair_removal_iters)}, seed=seed,
                          info=pair_info)
    rec = pr.run_projective_reconstruction(
        W_mat, lambda_mat, iters=iters, num_scale_iters=num_scale_iters, rank=rank,
        seed=seed, offset_mode=offset_mode, removal_iters=removal_iters, min_obs=min_obs,
        init=init, affine_start=-1, metric=metric, offset_ridge=offset_ridge)
    rec["info"]["pairs"] = pair_info.get("pairs", [])
    return rec
