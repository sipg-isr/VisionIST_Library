"""Global depth-augmented SfM with missing data: rotations from 2-frame
solves, an exact linear solve for translations and points, then joint
Levenberg-Marquardt over everything.

Model, for every OBSERVED entry (frame f, point p):

    (d_f * lam_fp + o_f) * w_fp  =  R_f X_p + t_f

lam_fp is the monocular depth, w_fp = [x, y, 1] the normalized ray. Missing
entries simply contribute no equations: no completion, no stitching.

  1. rotations: chain the consecutive 2-frame solves (no missing entries
     inside a pair -> the factorization is reliable there);
  2. with the R_f fixed and the depths at Z = d_f lam (d_f chained from the
     depth ratios of consecutive frames, o = 0) the residual is linear in
     (t_f, X_p): one exact least-squares solve;
  3. joint Levenberg-Marquardt over (d_f, o_f, R_f, t_f, X_p) (rotation
     increments R_f <- exp(w_f) R_f), Huber IRLS against outlier tracks.

The residual is  w - (R X + t) / (d lam + o)  (see ``_Problem``): x/y are
~ image errors, z the relative depth error (``depth_weight``). It is
scale-invariant, so the gauge d_0 = 1, t_0 = 0, R_0 = I is a pure gauge and
o_0 is estimated, not anchored. Two things that do NOT work on real video
(both tried): the fully linear solve of (d, o, t, X) -- its algebraic
residual, weighted by a fixed 1/lam, rewards shrinking the scene -- and
alternating linear / Procrustes steps, which flatline along the direction
coupling the offsets and the rotations. A weak prior pulls every o_f
towards 0: with little parallax the offsets are nearly unobservable.

The points are eliminated in closed form (their normal-equation blocks are
3x3 and independent), so every step solves an (8F x 8F) Schur system.
"""
import logging

import numpy as np
import torch

import projective_reconstruction as pr

logger = logging.getLogger(__name__)

_NC = 8            # camera unknowns per frame: d, o, t (3), rotation increment (3)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _proj_so3(M):
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def _exp_so3(w):
    """Rodrigues, batched: (F, 3) -> (F, 3, 3)."""
    th = np.linalg.norm(w, axis=1)[:, None, None]
    K = _skew(w)
    small = th < 1e-12
    th = np.where(small, 1.0, th)
    A = np.where(small, 1.0, np.sin(th) / th)
    B = np.where(small, 0.5, (1 - np.cos(th)) / th ** 2)
    return np.eye(3) + A * K + B * (K @ K)


def _skew(v):
    """(N, 3) -> (N, 3, 3) with skew(v) @ x = v x x."""
    S = np.zeros(v.shape[:-1] + (3, 3))
    S[..., 0, 1], S[..., 0, 2] = -v[..., 2], v[..., 1]
    S[..., 1, 0], S[..., 1, 2] = v[..., 2], -v[..., 0]
    S[..., 2, 0], S[..., 2, 1] = -v[..., 1], v[..., 0]
    return S


def _visibility(obs, min_obs=2, min_frame_obs=6):
    """Iteratively drop points seen in < min_obs kept frames and frames with
    < min_frame_obs kept points. Returns (vf (F,), vp (P,)) bool."""
    vf = np.ones(obs.shape[0], bool)
    vp = np.ones(obs.shape[1], bool)
    while True:
        sub = obs & vf[:, None] & vp[None, :]
        nvp = vp & (sub.sum(0) >= min_obs)
        nvf = vf & ((obs & nvp[None, :]).sum(1) >= min_frame_obs)
        if (nvp == vp).all() and (nvf == vf).all():
            return vf, vp
        vf, vp = nvf, nvp


def _rigid(X, Y):
    """argmin_{R,t} sum_i |Y_i - R X_i - t|^2   (X, Y: (3, n))."""
    mx, my = X.mean(1), Y.mean(1)
    R = _proj_so3((Y - my[:, None]) @ (X - mx[:, None]).T)
    return R, my - R @ mx


def _pair_rotation(W, lam, fa, fb, min_points, seed):
    """Relative rotation R_b R_a^T from the 2-frame block (fa, fb) on the
    points seen in both. Returns (R_rel, n_points, method)."""
    both = np.isfinite(lam[fa]) & np.isfinite(lam[fb])
    n = int(both.sum())
    if n < min_points:
        return np.eye(3), n, "none"
    rows = [3 * fa, 3 * fa + 1, 3 * fa + 2, 3 * fb, 3 * fb + 1, 3 * fb + 2]
    Wp = torch.from_numpy(W[rows][:, both])
    lp = torch.from_numpy(lam[[fa, fb]][:, both])
    try:
        with torch.no_grad():
            rec = pr.run_projective_reconstruction(Wp, lp, removal_iters=(20,), seed=seed)
        if bool(rec["vf"].all()):
            c0, c1 = [c.double().numpy() for c in rec["cam_lists"]]
            return _proj_so3(c1[:, :3]) @ _proj_so3(c0[:, :3]).T, n, "pair"
    except Exception as e:                                   # degenerate pair
        logger.info(f"pair ({fa},{fb}) factorization failed: {e}")
    # fallback: rigid fit of the raw back-projections, depth-normalized
    Xa = lam[fa, both] / np.median(lam[fa, both]) * W[3 * fa:3 * fa + 3, both]
    Xb = lam[fb, both] / np.median(lam[fb, both]) * W[3 * fb:3 * fb + 3, both]
    R, _ = _rigid(Xa, Xb)
    return R, n, "procrustes"


# ---------------------------------------------------------------------------
# one (damped) Gauss-Newton step, points eliminated by Schur complement
# ---------------------------------------------------------------------------

class _Problem:
    """Residual, per observation (frame f, point p):

        r = sqrt(h) * row * (w - (R_f X_p + t_f) / Z),   Z = d_f lam + o_f

    i.e. the ray minus the predicted point divided by the CORRECTED depth:
    x/y ~ image error, z = relative depth error (weighted by ``row[2]``).
    Scaling (d, o, t, X) together leaves it unchanged, so d_0 = 1 is a pure
    gauge. (The algebraic residual Z w - R X - t, weighted by a fixed 1/lam,
    is linear but not scale-invariant: on real video it shrinks the scene.)
    """

    def __init__(self, fi, pi, lam, w, F, P, depth_weight=1.0, offset_prior=0.0):
        self.fi, self.pi, self.lam, self.w, self.F, self.P = fi, pi, lam, w, F, P
        self.row = np.array([1.0, 1.0, float(depth_weight)])
        # prior rows a_f o_f / median(lam), a_f = offset_prior * sqrt(#obs in
        # f): a per-observation pull of relative size offset_prior on o_f
        self.a_o = (offset_prior * np.sqrt(np.bincount(fi, minlength=F))
                    / np.median(lam))
        self.z_min = 1e-6 * float(np.median(lam))

    def residual(self, h, d, o, t, Rs, X):
        fi, pi = self.fi, self.pi
        RX = np.einsum("nij,nj->ni", Rs[fi], X[pi])
        Q = RX + t[fi]
        Z = d[fi] * self.lam + o[fi]
        r = np.sqrt(h)[:, None] * self.row * (self.w - Q / Z[:, None])
        return r, RX, Q, Z

    def cost(self, h, d, o, t, Rs, X):
        r, _, _, Z = self.residual(h, d, o, t, Rs, X)
        if (Z < self.z_min).any():                    # a depth went through 0
            return np.inf
        return float((r ** 2).sum() + (self.a_o ** 2 * o ** 2).sum())

    def step(self, h, d, o, t, Rs, X, free_cam, mu):
        """Solve min |r + Jc dc + JX dX|^2 + mu * (Marquardt damping).
        free_cam: (F, 8) bool -- which camera unknowns move this step."""
        fi, pi, F, P = self.fi, self.pi, self.F, self.P
        N = len(fi)
        r, RX, Q, Z = self.residual(h, d, o, t, Rs, X)
        sh = np.sqrt(h)
        s = sh / Z                                                       # (N,)

        Jc = np.empty((N, 3, _NC))
        Jc[:, :, 0] = (s * self.lam / Z)[:, None] * Q                  # dr/dd
        Jc[:, :, 1] = (s / Z)[:, None] * Q                             # dr/do
        Jc[:, :, 2:5] = -s[:, None, None] * np.eye(3)                  # dr/dt
        Jc[:, :, 5:8] = s[:, None, None] * _skew(RX)                   # dr/dw, R <- exp(w) R
        JX = -s[:, None, None] * Rs[fi]                                # dr/dX
        Jc *= self.row[None, :, None]
        JX *= self.row[None, :, None]

        U = np.zeros((F, _NC, _NC))
        np.add.at(U, fi, np.einsum("nia,nib->nab", Jc, Jc))
        gc = np.zeros((F, _NC))
        np.add.at(gc, fi, np.einsum("nia,ni->na", Jc, r))
        U[:, 1, 1] += self.a_o ** 2                                     # offset prior
        gc[:, 1] += self.a_o ** 2 * o
        V = np.zeros((P, 3, 3))                                         # point blocks
        np.add.at(V, pi, np.einsum("nij,nik->njk", JX, JX))
        V = V + mu * V * np.eye(3)                                      # Marquardt
        Vinv = np.linalg.inv(V)
        Linv = np.linalg.inv(np.linalg.cholesky(V))                     # V^-1 = Linv^T Linv
        gX = np.zeros((P, 3))
        np.add.at(gX, pi, np.einsum("nij,ni->nj", JX, r))
        Wb = np.einsum("nia,nib->nab", Jc, JX)                          # (N, 8, 3)

        G = np.zeros((P, F, _NC, 3))
        G[pi, fi] = np.einsum("nab,ncb->nac", Wb, Linv[pi])           # W V^-1 W^T = G G^T
        G = G.transpose(1, 2, 0, 3).reshape(_NC * F, 3 * P)
        S = -G @ G.T
        for f in range(F):
            S[_NC * f:_NC * f + _NC, _NC * f:_NC * f + _NC] += U[f] + mu * np.diag(np.diag(U[f]))
        rhs = -gc
        VgX = np.einsum("pij,pj->pi", Vinv, gX)
        np.add.at(rhs, fi, np.einsum("nab,nb->na", Wb, VgX[pi]))

        free = free_cam.ravel()
        A = S[np.ix_(free, free)]
        A = A + 1e-12 * max(np.trace(A) / len(A), 1e-30) * np.eye(len(A))
        dc = np.zeros(_NC * F)
        dc[free] = np.linalg.solve(A, rhs.ravel()[free])
        dc = dc.reshape(F, _NC)
        dX = np.zeros((P, 3))
        np.add.at(dX, pi, np.einsum("nab,na->nb", Wb, dc[fi]))
        dX = -np.einsum("pij,pj->pi", Vinv, gX + dX)

        return (d + dc[:, 0], o + dc[:, 1], t + dc[:, 2:5],
                _exp_so3(dc[:, 5:8]) @ Rs, X + dX)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_linear_reconstruction(
    W_mat,
    lambda_mat,
    iters: int = 100,
    depth_weight: float = 1.0,
    offset_prior: float = 0.03,
    loss: str = "huber",
    loss_scale="auto",
    loss_k: float = 2.5,
    min_obs: int = 2,
    min_frame_obs: int = 6,
    pair_min_points: int = 8,
    tol: float = 1e-10,
    seed: int = 42,
) -> dict:
    """W_mat (3F, P) rays, lambda_mat (F, P) monocular depth, NaN = missing.

    depth_weight: weight of the depth row of the residual relative to the
    two image rows (~ ray noise / relative depth noise; 1 = equal).
    offset_prior: weak pull of every offset o_f towards 0 (relative to the
    median depth): with little parallax the offsets are nearly unobservable;
    0 disables it.
    loss: robust loss of the IRLS weights -- "huber" or None (plain least
    squares). A redescending loss (Cauchy) was tried and lands in wrong
    minima on small-baseline video, as does a small depth_weight: the depth
    row is what conditions the problem when the parallax is small.
    loss_scale: "auto" (loss_k x 1.4826 x MAD of the residuals, re-estimated
    every iteration) or a float (normalized-ray units ~ pixels / focal).

    Gauge: first kept frame R = I, t = 0, d = 1.

    Returns cameras (F', 3, 4) [R|t] (first kept frame = identity), points
    (P', 3), d, o (F',) with Z = d*lam + o, vf/vp (input masks), observed
    (F', P'), completed (3F', P') = R X + t, and info."""
    W = np.asarray(W_mat, dtype=np.float64)
    lam_all = np.asarray(lambda_mat, dtype=np.float64)
    obs_all = np.isfinite(lam_all)

    vf, vp = _visibility(obs_all, min_obs=min_obs, min_frame_obs=min_frame_obs)
    fids, pids = np.flatnonzero(vf), np.flatnonzero(vp)
    F, P = len(fids), len(pids)
    if F < 2 or P < 3:
        raise ValueError(f"only {F} frames / {P} points after the visibility filter")
    rows = (3 * fids[:, None] + np.arange(3)).ravel()
    W = W[rows][:, pids]
    lam_m = lam_all[fids][:, pids]
    obs = np.isfinite(lam_m)
    fi, pi = np.nonzero(obs)
    lam = lam_m[fi, pi]
    w = W.reshape(F, 3, P)[fi, :, pi]                                 # (N, 3)
    prob = _Problem(fi, pi, lam, w, F, P, depth_weight, offset_prior)

    # --- 1. rotations: chain the consecutive 2-frame solves -----------------
    Rs = np.empty((F, 3, 3)); Rs[0] = np.eye(3)
    pairs = []
    for f in range(F - 1):
        R_rel, n, how = _pair_rotation(W, lam_m, f, f + 1, pair_min_points, seed)
        Rs[f + 1] = R_rel @ Rs[f]
        pairs.append({"frames": [int(fids[f]), int(fids[f + 1])], "points": n, "method": how})

    # --- 2. exact linear solve for (t, X) at fixed depths ---------------------
    # with R and Z = d lam fixed the residual is linear in (t, X): one solve
    gauge = np.ones((F, _NC), bool)
    gauge[0, [0, 2, 3, 4, 5, 6, 7]] = False                          # d_0, t_0, R_0 fixed
    lin = np.zeros((F, _NC), bool); lin[1:, 2:5] = True               # t only (+ X)
    # per-frame depth scale from consecutive frames: the same point has a
    # similar depth in both, so d_{f+1}/d_f ~ median(lam_f / lam_{f+1})
    d = np.ones(F)
    for f in range(F - 1):
        both = obs[f] & obs[f + 1]
        ratio = np.median(lam_m[f, both] / lam_m[f + 1, both]) if both.sum() >= 3 else 1.0
        d[f + 1] = d[f] * (ratio if np.isfinite(ratio) and ratio > 0 else 1.0)
    o, t, X = np.zeros(F), np.zeros((F, 3)), np.zeros((P, 3))
    h = np.ones(len(fi))
    d, o, t, Rs, X = prob.step(h, d, o, t, Rs, X, lin, 0.0)

    # --- 3. joint LM over (d, o, t, R, X) with Huber IRLS --------------------
    if loss not in ("huber", None):
        raise ValueError(f"loss must be 'huber' or None, got {loss!r}")

    def robust_weights(r_plain):
        if loss is None:
            return np.ones_like(r_plain), np.inf
        if loss_scale == "auto":
            # floor: exact data must not down-weight round-off residuals
            delta = max(loss_k * 1.4826 * np.median(r_plain), 1e-6)
        else:
            delta = float(loss_scale)
        return np.minimum(1.0, delta / np.maximum(r_plain, 1e-15)), delta

    ones = np.ones(len(fi))
    mu, it, converged = 1e-4, 0, False
    for it in range(1, iters + 1):
        r_plain = np.linalg.norm(prob.residual(ones, d, o, t, Rs, X)[0], axis=1)
        h, _ = robust_weights(r_plain)
        cost = prob.cost(h, d, o, t, Rs, X)
        for _ in range(10):
            cand = prob.step(h, d, o, t, Rs, X, gauge, mu)
            new = prob.cost(h, *cand)
            if np.isfinite(new) and new <= cost:
                break
            mu *= 4.0
        else:
            converged = True                                          # no descent left
            break
        d, o, t, Rs, X = cand
        mu = max(mu / 3.0, 1e-12)
        if cost - new <= tol * cost:
            converged = True
            break

    r_plain = np.linalg.norm(prob.residual(ones, d, o, t, Rs, X)[0], axis=1)
    h, delta = robust_weights(r_plain)

    cams = np.concatenate([Rs, t[:, :, None]], axis=2)                # (F, 3, 4)
    completed = (Rs @ X.T[None] + t[:, :, None]).reshape(3 * F, P)    # Z * w model
    info = {
        "iterations": it,
        "converged": bool(converged),
        "pairs": pairs,
        "outliers": float((r_plain > delta).mean()),
        "residual_median": float(np.median(r_plain)),
    }
    return dict(cameras=cams, points=X, d=d, o=o, vf=vf, vp=vp,
                observed=obs, completed=completed, info=info)
