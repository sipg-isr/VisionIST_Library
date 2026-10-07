"""
projective_reconstruction.py — depth-augmented projective SfM with MISSING DATA.

Ported from ``~/trackers/src/projective_reconstruction.py`` (+ the pieces of
``src/mat_compl.py`` / ``src/ortho_factorization.py`` it runs), keeping the
active code paths; dead experiment branches (SVT, column-growing init,
joint imputation, k3d plotting) are left out. ``projective_factorization_fast``
is the det-guarded version from ``sfm_core.py`` (Marques & Costeira, CVIU
2009: det = +1, shape re-solved against the final motion).

Only dependency: ``torch``.

Pipeline (``run_projective_reconstruction``)
--------------------------------------------
    W_mat (3F, P) rays [x, y, 1] per frame, lambda_mat (F, P) monocular depth,
    NaN where a track is not observed.

1. ``filter_visibility``   — iteratively drop frames / points with too few
                             observations.
2. ``calibrate_with_completion``
       Rank-4 ALS matrix completion on the OBSERVED entries of
       (lambda + o_f) * W, RANSAC column (track) rejection at fixed
       iterations, and — after a warm-up — the masked per-frame affine depth
       offset fit (``_update_affine_ortho``). Missing entries are filled from
       the low-rank model.
3. Scale correction — repeated ``projective_factorization_fast`` on the
       completed matrix; per-frame scales from the Procrustes singular values.
4. Final factorization -> per-frame [R | t] aligned to the first surviving
       camera, and the shape in that camera's frame.

With a complete matrix the completion is exact, so this reduces to the
plain affine-calibration + factorization scheme.
"""
from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)


# =============================================================================
# 1. Input construction
# =============================================================================

def make_homogenous(obs: torch.Tensor) -> torch.Tensor:
    """(2F, P) with rows [x0,y0, x1,y1, ...] -> (3F, P) rows [x,y,1] per frame."""
    TwoF, P = obs.shape
    F = TwoF // 2
    ones = torch.ones((F, 1, P), device=obs.device, dtype=obs.dtype)
    out = torch.cat([obs.view(F, 2, P), ones], dim=1)
    return out.view(3 * F, P)


def sample_depths(depths: torch.Tensor, tracks: torch.Tensor) -> torch.Tensor:
    """
    depths: (F, H, W), tracks: (2F, P) pixel coords (u row, v row alternating).
    Returns: (F, P) bilinear-sampled depths.
    """
    F, H, W = depths.shape
    P = tracks.shape[1]
    out = []
    for f in range(F):
        u = tracks[2 * f, :]
        v = tracks[2 * f + 1, :]
        u_norm = (u / (W - 1)) * 2 - 1
        v_norm = (v / (H - 1)) * 2 - 1
        grid = torch.stack([u_norm, v_norm], dim=-1).view(1, P, 1, 2)
        z = torch.nn.functional.grid_sample(
            depths[f:f + 1].unsqueeze(0), grid, align_corners=True, mode="bilinear"
        ).view(-1)
        out.append(z)
    return torch.stack(out, dim=0)


def build_depth_weighted_matrix(tracks: torch.Tensor, depths: torch.Tensor,
                                Ks: torch.Tensor):
    """
    Tracks + depth maps -> homogeneous ray matrix and depth estimate.

    Args:
        tracks:  (2F, P) pixel coords (u/v alternating rows).
        depths:  (F, H, W) monocular depth maps.
        Ks:      (F, 3, 3) per-frame intrinsics (or (3,3) shared).
    Returns:
        W_mat:      (3F, P), rows [x, y, 1] per frame, x = (u-cx)/fx.
        lambda_mat: (F, P), depth estimates sampled at the tracks.
    """
    F = depths.shape[0]
    if Ks.ndim == 2:
        Ks = Ks.unsqueeze(0).expand(F, -1, -1)

    z = sample_depths(depths, tracks)
    u = tracks[0::2, :]
    v = tracks[1::2, :]
    # per-frame intrinsics broadcast over the P columns -> (F, 1)
    x = (u - Ks[:, 0, 2, None]) / Ks[:, 0, 0, None]
    y = (v - Ks[:, 1, 2, None]) / Ks[:, 1, 1, None]
    rays = torch.stack([x, y, torch.ones_like(x)], dim=1).reshape(3 * F, -1)
    return rays, z


# =============================================================================
# 2. SVD factorization + metric upgrade (from sfm_core.py, det-guarded)
# =============================================================================

def projective_factorization_fast(A: torch.Tensor):
    """
    A: (3F, P), rows per frame are [X*Lambda, Y*Lambda, Lambda] (i.e. depth
       scaled ray triples). Must be complete (no NaN).

    Returns
        M:      (3F, 3)   stacked per-frame rotations
        S:      (3, P)    shape matrix (zero-mean)
        tvec:   (3F,)     per-row translation component
        scales: (F, 3)    singular values of the Procrustes-rotations
    """
    device, dtype = A.device, A.dtype
    F = A.shape[0] // 3

    # --- center: translations = row means (shape taken zero-mean) ---
    tvec = A.mean(dim=1)
    W_c = A - tvec.unsqueeze(1)

    # --- balanced rank-3 SVD ---
    U, Svals, Vh = torch.linalg.svd(W_c, full_matrices=False)
    S_root = torch.sqrt(Svals[:3])
    M_hat = U[:, :3] * S_root          # (3F, 3)
    S_hat = S_root[:, None] * Vh[:3, :]  # (3, P)

    # --- metric upgrade: find L so M_hat_f @ L is orthonormal for all f ---
    # Constraint m_i^T Q m_j = delta_ij with Q = L L^T (symmetric 3x3).
    i_idx = torch.tensor([0, 0, 0, 1, 1, 2], device=device)
    j_idx = torch.tensor([0, 1, 2, 1, 2, 2], device=device)
    sym = torch.where(i_idx == j_idx,
                      torch.ones(6, device=device, dtype=dtype),
                      torch.full((6,), 2.0, device=device, dtype=dtype))
    pairs_a = torch.tensor([0, 1, 2, 0, 0, 1], device=device)
    pairs_b = torch.tensor([0, 1, 2, 1, 2, 2], device=device)
    b_vec = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)

    Mf = M_hat.reshape(F, 3, 3)
    ma = Mf[:, pairs_a, :]                     # (F, 6, 3)
    mb = Mf[:, pairs_b, :]                     # (F, 6, 3)
    A_lin = (ma[:, :, i_idx] * mb[:, :, j_idx] * sym).reshape(F * 6, 6)  # (6F, 6)
    B = b_vec.unsqueeze(0).expand(F, -1).reshape(F * 6, 1)
    # Q is only defined up to a global scalar -> homogeneous solve via null space
    A_aug = torch.cat([A_lin, -B], dim=1)      # (6F, 7)
    _, _, Vh_full = torch.linalg.svd(A_aug)
    ell = Vh_full[-1]
    ell = ell / ell[6]
    q = ell[:6]
    Q = torch.zeros(3, 3, device=device, dtype=dtype)
    Q[i_idx, j_idx] = q
    Q[j_idx, i_idx] = q                        # symmetrize

    # --- L = matrix square root of Q ---
    Uq, Sq, _ = torch.linalg.svd(Q)
    L = Uq @ torch.diag(torch.sqrt(torch.clamp(Sq, min=1e-9)))
    # Sign convention: force every per-frame motion to be a PROPER rotation
    # (det = +1), not its mirror image (det = -1). The metric upgrade leaves an
    # unavoidable +/- ambiguity in L (hence in M_f = M_hat_f @ L): the same image
    # is produced by a rotation and by its reflection over the shape plane.
    # This is the anti-correlated +/- sign pairing of the "last" (depth) component
    # in Marques & Costeira, CVIU 2009, eqs. (16)-(17). Flipping one column of L
    # leaves M*S (= A) unchanged (the matching row of S = L^-1 S_hat flips too).
    if torch.linalg.det(M_hat[:3] @ L) < 0:
        L[:, -1] *= -1.0

    M = M_hat @ L

    # --- batched Procrustes: project each 3x3 block onto SO(3), forcing det = +1 ---
    # (explicit matrix form of the +/-, -/+ sign pairing, eqs. A.4/A.5)
    Mf2 = M.reshape(F, 3, 3)
    Up, Sp, Vhp = torch.linalg.svd(Mf2)
    dgn = torch.where(torch.linalg.det(Up) * torch.linalg.det(Vhp) < 0, -1.0, 1.0)
    D = torch.eye(3, device=device, dtype=dtype).expand(F, 3, 3).clone()
    D[:, 2, 2] = dgn                                    # (F,3,3)
    R = Up @ D @ Vhp                                     # (F,3,3), det=+1
    M = R.reshape(3 * F, 3)

    # --- shape: least-squares consistent with the FINAL motion ---
    # (Marques & Costeira, CVIU 2009, Algorithm 1 step 3:  S = M^+ Zc)
    S, *_ = torch.linalg.lstsq(M, W_c)                   # (3, P)

    return M, S, tvec, Sp                               # Sp: (F, 3)


# =============================================================================
# 3. Missing-data pieces (from src/mat_compl.py, src/ortho_factorization.py)
# =============================================================================

def _update_affine_ortho(x, y, lam, M, mask=None, eps=1e-6, clamp_pos=True, offset_ridge=0.0):
    """
    Per-frame affine depth fit (closed form, 2x2 normal equations) over the
    observed entries only.

    x, y : (F, P) normalized image coords.   lam : (F, P) depths.
    M    : (3F, P) current low-rank model.    mask : (F, P) bool, True where observed.
    offset_ridge : ridge on the offsets, as a fraction of the data's own
                   information on them (0 = none): holds them near 0 where the
                   data barely constrains them (small parallax).
    Returns d (F,) slopes, s (F,) offsets.
    """
    F, P = lam.shape
    Mx, My = M[0::3], M[1::3]

    if mask is None:
        # infer from NaNs in lam
        mask = ~torch.isnan(lam)

    m = mask.float()  # (F, P)

    # zero out NaN entries so they don't contribute
    x_ = torch.nan_to_num(x, nan=0.0)
    y_ = torch.nan_to_num(y, nan=0.0)
    L = torch.nan_to_num(lam, nan=0.0)
    Mx_ = torch.nan_to_num(Mx, nan=0.0)
    My_ = torch.nan_to_num(My, nan=0.0)

    X2Y2 = x_**2 + y_**2            # (F, P)

    a11 = (m * L * L * X2Y2).sum(1) + eps
    a22 = (m * X2Y2).sum(1) * (1.0 + offset_ridge) + eps
    a12 = (m * L * X2Y2).sum(1)

    b1 = (m * (L * x_ * Mx_ + L * y_ * My_)).sum(1)
    b2 = (m * (x_ * Mx_ + y_ * My_)).sum(1)

    det = a11 * a22 - a12 * a12 + eps
    d = (a22 * b1 - a12 * b2) / det
    s = (-a12 * b1 + a11 * b2) / det

    if clamp_pos:
        d = torch.clamp(d, min=1e-5)

    return d, s


def ransac_subspace(W_filled, mask_w, rank=4, n_iters=100,
                    threshold=None, min_sample=20, generator=None):
    """
    W_filled : (3F, P) — completed/observed matrix
    mask_w   : (3F, P) bool
    Returns  : inlier_cols (P,) bool mask, per-entry |residual| (3F, P)
    """
    F3, P = W_filled.shape
    device = W_filled.device

    if threshold is None:
        # set threshold as median absolute column norm * factor
        col_norms = W_filled.norm(dim=0)  # (P,)
        threshold = col_norms.median() * 0.01

    best_inliers = torch.zeros(P, dtype=torch.bool, device=device)
    best_count = 0

    for _ in range(n_iters):
        # 1. Sample minimal set of columns
        sample_idx = torch.randperm(P, device=device, generator=generator)[:min_sample]
        W_sample = W_filled[:, sample_idx]  # (3F, min_sample)

        # 2. Fit rank-4 subspace from sample — use LEFT singular vectors
        U_s, _, _ = torch.linalg.svd(W_sample, full_matrices=False)
        V_basis = U_s[:, :rank]  # (3F, rank) — column space basis

        # 3. Project all columns onto subspace and measure residual
        W_proj = V_basis @ (V_basis.T @ W_filled)  # (3F, P)
        residuals = (W_filled - W_proj).norm(dim=0)  # (P,)

        # 4. Count inliers
        inliers = residuals < threshold
        count = inliers.sum().item()

        if count > best_count:
            best_count = count
            best_inliers = inliers

    return best_inliers, torch.abs(W_filled - W_proj)


def calibrate_with_completion(tracks, lam, mask, rank=4, iters=100, tol=1e-4, ridge=1e-10,
                              offset_mode="normalize", removal_iters=(10, 20, 30, 40),
                              min_obs=2, generator=None, U_init=None, o_init=None,
                              affine_start=40, metric=False, offset_ridge=0.0):
    """
    Jointly estimates the per-frame depth offset and completes missing
    entries, such that (lam + o) * tracks is rank-4 (ALS completion;
    RANSAC track rejection at ``removal_iters``; affine fit after iter 40).

    Args:
        tracks      : (3F, P) rays [x,y,1] interleaved, NaN where missing
        lam         : (F, P)  monocular depths, NaN where missing
        mask        : (3F, P) bool, True where observed
        rank        : target rank
        iters       : max iterations
        tol         : convergence threshold
        ridge       : ALS regularization
        offset_mode : "estimate" (raw o) | "normalize" (o - o[0], default) | "zero"
        min_obs     : drop points/frames observed fewer times after a removal
        generator   : torch.Generator for the RANSAC sampling
        U_init      : (3F, rank) warm start of the camera factor (e.g. stacked
                      [R_f | t_f] / d_f); None = SVD of the column-mean fill
        o_init      : (F,) warm start of the offsets (kept until the affine
                      fit starts)
        affine_start: the affine fit runs for iterations > affine_start
                      (original: 40; with a warm start, -1 = from the start)
        metric      : keep the factors rigid (needs a metric warm start,
                      rank 4): every camera block U_f = s_f [R_f | t_f]
                      (3x3 part projected on the nearest scaled rotation)
                      and every point V_p = [X_p, 1]
        offset_ridge: ridge of the affine fit on the offsets (see
                      _update_affine_ortho; 0 = original)

    Returns:
        o        : (F,)      offset per frame (0 for removed frames)
        W_final  : (3F, P)   completed matrix (removed rows/cols filled with NaN)
        M_full   : (3F, P)   rank-4 approximation (removed rows/cols filled with NaN)
        mask_out : (3F, P)   final observation mask (float)
        active_frames, active_cols : bool masks of the surviving frames / points
        info     : dict — iterations run, points/frames removed
    """
    F_orig, P_orig = lam.shape
    device = lam.device
    dtype = lam.dtype

    # Active index trackers (boolean over originals)
    active_cols = torch.ones(P_orig, dtype=torch.bool, device=device)
    active_frames = torch.ones(F_orig, dtype=torch.bool, device=device)

    # Working views — will be re-sliced in place
    lam_w = lam.clone()
    tracks_w = tracks.clone()
    mask_w = mask.clone()

    d = torch.ones(F_orig, device=device, dtype=dtype)
    o = torch.zeros(F_orig, device=device, dtype=dtype)

    offset_history = []
    eye_r = ridge * torch.eye(rank, device=device, dtype=dtype)
    removed = []

    if U_init is None:
        # --- SVD initialisation (missing entries filled with the column mean) ---
        lam3_w = lam_w.repeat_interleave(3, dim=0)
        W_init = lam3_w * tracks_w
        col_mean = torch.nanmean(W_init, dim=0)
        W_filled = torch.where(mask_w, W_init, col_mean.unsqueeze(0).expand_as(W_init))
        Ui, Si, Vhi = torch.linalg.svd(W_filled, full_matrices=False)
        U = (Ui[:, :rank] * Si[:rank].sqrt()).contiguous()
        V = (Vhi[:rank].T * Si[:rank].sqrt()).contiguous()
    else:
        # --- warm start: given cameras (+ offsets), points from one ALS V step ---
        U = U_init.to(dtype).clone()
        if o_init is not None:
            o = o_init.to(dtype).clone()
        W_init = (lam_w + o[:, None]).repeat_interleave(3, dim=0) * tracks_w
        mask_f0 = mask_w.float()
        A_V = torch.einsum('ij,ik,il->jkl', mask_f0, U, U) + eye_r
        b_V = (mask_f0 * torch.nan_to_num(W_init)).T @ U
        V = torch.linalg.solve(A_V, b_V.unsqueeze(-1)).squeeze(-1)
    M = U @ V.T

    prev_rho = float('inf')
    it = -1

    for it in range(iters):
        F_w = lam_w.shape[0]
        P_w = lam_w.shape[1]

        lam3_w = lam_w.repeat_interleave(3, dim=0)
        d3 = d.repeat_interleave(3)
        o3 = o.repeat_interleave(3)
        W_scaled = (d3[:, None] * lam3_w + o3[:, None]) * tracks_w
        W_filled = torch.where(mask_w, W_scaled, M)

        # ---- Outlier removal at fixed iterations (RANSAC over columns) ----
        if removal_iters and it in removal_iters:
            inlier_cols, _ = ransac_subspace(
                W_filled, mask_w, rank=rank,
                n_iters=100, min_sample=10, generator=generator,
            )
            keep_cols = inlier_cols
            keep_frames = torch.ones(F_w, dtype=torch.bool, device=device)  # RANSAC on cols only

            # Under-observed after per-entry masking
            # clamp to actual frame/point count so 2-frame windows aren't emptied
            obs_per_point = mask_w[0::3].sum(dim=0)
            keep_cols = keep_cols & (obs_per_point >= min(min_obs, F_w))

            obs_per_frame = mask_w[0::3].sum(dim=1)
            keep_frames = keep_frames & (obs_per_frame >= min(min_obs, P_w))

            n_rem_cols = (~keep_cols).sum().item()
            n_rem_frames = (~keep_frames).sum().item()

            if n_rem_cols > 0 or n_rem_frames > 0:
                log.info(f"iter {it:3d} | removing {n_rem_cols} cols, {n_rem_frames} frames "
                         f"-> ({keep_frames.sum().item()} frames, "
                         f"{keep_cols.sum().item()} points remaining)")
                removed.append({"iter": it, "points": n_rem_cols, "frames": n_rem_frames})

                keep_rows = keep_frames.repeat_interleave(3)

                lam_w = lam_w[keep_frames][:, keep_cols]
                tracks_w = tracks_w[keep_rows][:, keep_cols]
                mask_w = mask_w[keep_rows][:, keep_cols]
                U = U[keep_rows]
                V = V[keep_cols]

                active_cols[active_cols.clone()] = keep_cols
                active_frames[active_frames.clone()] = keep_frames

                d = d[keep_frames]
                o = o[keep_frames]

                lam3_w = lam_w.repeat_interleave(3, dim=0)
                d3 = d.repeat_interleave(3)
                o3 = o.repeat_interleave(3)
                W_scaled = (d3[:, None] * lam3_w + o3[:, None]) * tracks_w
                W_filled = torch.where(mask_w, W_scaled, torch.zeros_like(W_scaled))
                M = U @ V.T

        mask_f = mask_w.float()

        # ---- ALS: update U ----
        A_U = torch.einsum('ij,jk,jl->ikl', mask_f, V, V) + eye_r
        b_U = (mask_f * W_filled) @ V
        U = torch.linalg.solve(A_U, b_U.unsqueeze(-1)).squeeze(-1)

        if metric:
            U = _project_scaled_rotations(U)

        # ---- ALS: update V ----
        if metric:                                   # V_p = [X_p, 1]: solve X_p only
            U3, u4 = U[:, :3], U[:, 3]
            A_V = torch.einsum('ij,ik,il->jkl', mask_f, U3, U3) + eye_r[:3, :3]
            b_V = (mask_f * (W_filled - u4[:, None])).T @ U3
            X = torch.linalg.solve(A_V, b_V.unsqueeze(-1)).squeeze(-1)
            V = torch.cat([X, torch.ones_like(X[:, :1])], dim=1)
        else:
            A_V = torch.einsum('ij,ik,il->jkl', mask_f, U, U) + eye_r
            b_V = (mask_f * W_filled).T @ U
            V = torch.linalg.solve(A_V, b_V.unsqueeze(-1)).squeeze(-1)

        M = U @ V.T

        # ---- Affine calibration (offsets only; scales come later) ----
        if it > affine_start:
            d, o = _update_affine_ortho(
                tracks_w[0::3], tracks_w[1::3], lam_w, M, mask=mask_w[0::3],
                offset_ridge=offset_ridge,
            )
            if offset_mode == "normalize":
                o = o - o[0]
            elif offset_mode == "zero":
                o = torch.zeros_like(o)
        elif U_init is None:
            o = torch.zeros_like(o)

        d = torch.ones_like(d)

        offset_history.append(o.clone())

        # ---- Convergence ----
        lam3_w = lam_w.repeat_interleave(3, dim=0)
        d3 = d.repeat_interleave(3)
        o3 = o.repeat_interleave(3)
        W_scaled = d3[:, None] * (lam3_w + o3[:, None]) * tracks_w

        rho = (W_scaled - M)[mask_w].norm().item()

        if it > max(affine_start, 0) and abs(prev_rho - rho) < tol and (
                len(offset_history) < 2
                or torch.allclose(offset_history[-1], offset_history[-2], atol=tol)):
            break
        prev_rho = rho

    best_rows = active_frames.repeat_interleave(3)

    # ---- Scatter back into full-size tensors ----
    W_final = torch.full((3 * F_orig, P_orig), float('nan'), device=device, dtype=dtype)
    M_full = torch.full((3 * F_orig, P_orig), float('nan'), device=device, dtype=dtype)
    mask_out = torch.zeros(3 * F_orig, P_orig, dtype=torch.bool, device=device)
    o_full = torch.zeros(F_orig, device=device, dtype=dtype)

    # Final W for surviving entries: observed where seen, model elsewhere
    lam3_w = lam_w.repeat_interleave(3, dim=0)
    d3 = d.repeat_interleave(3)
    o3 = o.repeat_interleave(3)
    W_obs = (d3[:, None] * lam3_w + o3[:, None]) * tracks_w
    W_comp = torch.where(mask_w, W_obs, M)

    rows = best_rows.nonzero(as_tuple=True)[0]
    cols = active_cols.nonzero(as_tuple=True)[0]

    W_final[rows[:, None], cols[None, :]] = W_comp
    M_full[rows[:, None], cols[None, :]] = M
    mask_out[rows[:, None], cols[None, :]] = mask_w

    o_full[active_frames] = o

    info = {"iterations": it + 1, "removed": removed, "U": U, "V": V}
    return o_full, W_final, M_full, mask_out.float(), active_frames, active_cols, info


def _project_scaled_rotations(U):
    """(3F, 4) camera factor -> every 3x3 block replaced by the nearest
    scaled rotation s R (s = mean singular value, det R = +1); t kept."""
    F = U.shape[0] // 3
    A = U[:, :3].reshape(F, 3, 3)
    Uu, S, Vh = torch.linalg.svd(A)
    D = torch.ones(F, 3, dtype=U.dtype, device=U.device)
    D[:, 2] = torch.sign(torch.linalg.det(Uu @ Vh))
    R = Uu @ (D[:, :, None] * Vh)
    out = U.clone()
    out[:, :3] = (S.mean(1)[:, None, None] * R).reshape(3 * F, 3)
    return out


def check_visibility(mask_F, rank=4):
    """mask_F: (F, P) bool — True where observed. Logs observation counts."""
    obs_per_frame = mask_F.sum(dim=1)   # (F,) — points per frame
    obs_per_point = mask_F.sum(dim=0)   # (P,) — frames per point
    log.info(f"obs per frame: min={obs_per_frame.min().item()} max={obs_per_frame.max().item()} | "
             f"obs per point: min={obs_per_point.min().item()} max={obs_per_point.max().item()}")
    return obs_per_frame, obs_per_point


def filter_visibility(tracks, lam, mask, rank=4):
    """Iteratively drop frames seeing < rank points and points seen in < rank
    frames. Returns the filtered (tracks, lam, mask (3F, P)) and the bool
    masks of valid frames / points in the original indexing."""
    mask_F = mask[0::3]
    valid_frames = torch.ones(mask_F.shape[0], dtype=torch.bool, device=mask.device)
    valid_points = torch.ones(mask_F.shape[1], dtype=torch.bool, device=mask.device)

    for _ in range(100):
        F, P = mask_F.shape
        frame_thresh = min(rank, P)
        point_thresh = min(rank, F)

        obs_per_frame = mask_F.sum(dim=1)
        obs_per_point = mask_F.sum(dim=0)

        new_vf = obs_per_frame >= frame_thresh
        new_vp = obs_per_point >= point_thresh

        if new_vf.all() and new_vp.all():
            break

        mask_F = mask_F[new_vf][:, new_vp]
        tracks = tracks[new_vf.repeat_interleave(3)][:, new_vp]
        lam = lam[new_vf][:, new_vp]

        valid_frames[valid_frames.clone()] = new_vf
        valid_points[valid_points.clone()] = new_vp

    return tracks, lam, mask_F.repeat_interleave(3, dim=0), valid_frames, valid_points


# =============================================================================
# 4. Main driver
# =============================================================================

def run_projective_reconstruction(
    W_mat: torch.Tensor,
    lambda_mat: torch.Tensor,
    iters: int = 100,
    num_scale_iters: int = 4,
    rank: int = 4,
    seed: int = 42,
    offset_mode: str = "normalize",
    removal_iters: tuple = (10, 20, 30, 40),
    min_obs: int = 2,
    init=None,
    affine_start: int = 40,
    metric: bool = False,
    offset_ridge: float = 0.0,
) -> dict:
    """
    Full projective reconstruction pipeline: filter visibility, complete the
    measurement matrix, run projective factorization with scale correction,
    and return cameras + shape.

    Args:
        W_mat:            Observation matrix (3F, P), NaNs for missing entries.
        lambda_mat:       Depth matrix (F, P), NaNs for missing entries.
        iters:            Iterations for calibrate_with_completion.
        num_scale_iters:  Scale-correction refinement iterations.
        rank:             Rank of the completion model.
        seed:             Seed of the RANSAC sampling (reproducible).
        offset_mode:      "estimate", "normalize" (default), or "zero".
        removal_iters:    Completion iterations at which RANSAC rejects tracks.
        min_obs:          Minimum observations per point/frame after a removal.
        init:             Optional warm start: callable(tracks_f, lam_f, mask_f)
                          -> (U_init (3F, rank), o_init (F,)) on the
                          visibility-filtered matrices (see pairs_reconstruction).
        affine_start:     Iteration after which the affine fit runs (40; -1
                          with a warm start).
        metric:           Keep the completion factors rigid (warm start only).
        offset_ridge:     Ridge of the affine fit on the offsets (0 = original).

    Returns dict with keys:
        cam_lists         list of (3, 4) [R | t], first surviving camera = reference
        aligned_shape     (3, P) shape in the reference camera's frame
        final_W           (3F, P) completed rays
        final_lam         (F, P) final depths
        compl_W_lam       (3F, P) completed, scale-corrected W*lambda (factorized)
        current_scales    (F,) per-frame scale factors
        offsets           (F,) per-frame offsets of the surviving frames
        vf                (F_orig,) bool mask of surviving frames in original indexing
        vp                (P_orig,) bool mask of surviving points in original indexing
        mask_f            (3F, P) final observation mask
        info              completion diagnostics (iterations, removals)
    """
    generator = torch.Generator(device=W_mat.device).manual_seed(seed)

    # --- Build NaN matrices from mask ---
    mask = ~torch.isfinite(lambda_mat)
    W_mat_nan = W_mat.clone()
    W_mat_nan[mask.repeat_interleave(3, dim=0)] = float('nan')
    lambda_mat_nan = lambda_mat.clone()
    lambda_mat_nan[mask] = float('nan')

    # --- Filter visibility ---
    tracks_f, lam_f, mask_f, vf, vp = filter_visibility(
        W_mat_nan, lambda_mat_nan,
        (~mask).repeat_interleave(3, dim=0), rank=2,
    )
    check_visibility(mask_f[0::3], rank=2)
    if tracks_f.shape[0] < 6 or tracks_f.shape[1] < rank:
        raise ValueError(f"not enough co-visible data after visibility filtering "
                         f"({tracks_f.shape[0] // 3} frames, {tracks_f.shape[1]} points)")

    # --- Matrix completion (optionally warm-started) ---
    U_init = o_init = None
    if init is not None:
        U_init, o_init = init(tracks_f, lam_f, mask_f)
    o, compl_W_lam, M, mask_f, surviving_frames, surviving_cols, info = calibrate_with_completion(
        tracks_f, lam_f, mask_f, rank=rank, iters=iters, offset_mode=offset_mode,
        removal_iters=removal_iters, min_obs=min_obs, generator=generator,
        U_init=U_init, o_init=o_init, affine_start=affine_start, metric=metric,
        offset_ridge=offset_ridge)

    # --- Align vf/vp to surviving frames/cols ---
    vp_indices = vp.nonzero(as_tuple=True)[0]
    vp[vp_indices[~surviving_cols]] = False

    vf_indices = vf.nonzero(as_tuple=True)[0]
    vf[vf_indices[~surviving_frames]] = False

    # --- Drop removed rows/cols from working matrices ---
    sel_rows = surviving_frames.repeat_interleave(3)
    compl_W_lam = compl_W_lam[sel_rows][:, surviving_cols]
    mask_f = mask_f[sel_rows][:, surviving_cols]
    o = o[surviving_frames]

    # --- Scale correction ---
    F_frames = compl_W_lam.shape[0] // 3
    current_scales = torch.ones(F_frames, device=compl_W_lam.device, dtype=compl_W_lam.dtype)

    W_corr = compl_W_lam.clone()
    for _ in range(num_scale_iters):
        # per-frame scale
        scl_map = current_scales.repeat_interleave(3)[:, None]
        _, _, _, sigmas = projective_factorization_fast(W_corr / scl_map)
        scales = sigmas.mean(dim=1)
        scales = scales / scales.max()
        current_scales = current_scales * scales

    # --- Final matrices ---
    final_W_lam = W_corr / current_scales.repeat_interleave(3)[:, None]
    final_lam = final_W_lam[2::3]
    final_W = final_W_lam / final_lam.repeat_interleave(3, dim=0)

    motion, shape, tvec, _ = projective_factorization_fast(final_W_lam)
    tvec = tvec.unsqueeze(1)                       # (3F, 1)

    # --- Build camera list aligned to first camera ---
    # cam f in cam0-world:  M'_f = M_f M_0^T ,  t'_f = t_f - M'_f t_0
    R1_inv = motion[:3, :3].t()
    t1_est = tvec[:3]
    cam_lists = []
    for f in range(motion.shape[0] // 3):
        Mi = motion[f * 3: (f + 1) * 3, :]
        ti = tvec[f * 3: (f + 1) * 3]
        Mi_new = Mi @ R1_inv
        cam_lists.append(torch.cat((Mi_new, ti - (Mi_new @ t1_est)), dim=1))

    aligned_shape = motion[:3, :3] @ shape + t1_est

    return dict(
        cam_lists=cam_lists,
        aligned_shape=aligned_shape,
        final_W=final_W,
        final_lam=final_lam,
        compl_W_lam=final_W_lam,
        current_scales=current_scales,
        offsets=o,
        vf=vf,
        vp=vp,
        mask_f=mask_f,
        info=info,
    )


# =============================================================================
# 5. Evaluation helper (Sim(3) alignment + per-frame errors)
# =============================================================================

def umeyama_alignment(src: torch.Tensor, dst: torch.Tensor, with_scale: bool = True):
    """R,t,s such that dst ≈ s * R @ src + t (per column; src,dst: (3, N))."""
    N = src.shape[1]
    mu_s, mu_d = src.mean(1, keepdim=True), dst.mean(1, keepdim=True)
    sc_, dc_ = src - mu_s, dst - mu_d
    cov = (dc_ @ sc_.T) / N
    U, D, Vt = torch.linalg.svd(cov)
    S = torch.eye(3, device=src.device, dtype=src.dtype)
    if torch.linalg.det(U) * torch.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (sc_ ** 2).sum() / N
    s = torch.sum(D * S.diagonal()) / var if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return R, t.squeeze(1), s


def compare_cameras(cam_lists: list, gt_lists: list):
    """
    Align two camera lists with Umeyama (Sim(3)), then report mean relative
    rotation error (deg) and mean translation-direction error (deg) against
    the first camera.
    """
    def centre(m):
        R, t = m[:3, :3], m[:3, 3]
        return -R.t() @ t

    src = torch.stack([centre(c) for c in cam_lists])
    dst = torch.stack([centre(g) for g in gt_lists])
    R, t, s = umeyama_alignment(src.t(), dst.t())
    T = torch.eye(4, dtype=src.dtype)
    T[:3, :3] = s * R
    T[:3, 3] = t
    T_inv = torch.inverse(T)

    def m4(m):
        out = torch.eye(4, dtype=m.dtype)
        out[:3, :] = m
        return out

    M4 = [m4(c) for c in cam_lists]
    G4 = [m4(g) for g in gt_lists]
    M4a = [M @ T_inv for M in M4]
    G4a = [G @ T_inv for G in G4]

    rot_errs, dir_errs = [], []
    ref = 0
    for i in range(len(M4a)):
        if i == ref:
            continue
        Rm = (M4a[i] @ torch.inverse(M4a[ref]))[:3, :3]
        Rg = (G4a[i] @ torch.inverse(G4a[ref]))[:3, :3]
        s_r = torch.sign(torch.linalg.det(Rm)) * torch.abs(torch.linalg.det(Rm)) ** (1.0/3.0)
        Rmp = Rm / s_r
        cosang = ((Rmp @ Rg.t()).trace() - 1) / 2
        rot_errs.append(torch.rad2deg(torch.acos(torch.clamp(cosang, -1, 1))))
        tm = (M4a[i] @ torch.inverse(M4a[ref]))[:3, 3]
        tg = (G4a[i] @ torch.inverse(G4a[ref]))[:3, 3]
        nm, ng = tm.norm(), tg.norm()
        if nm > 1e-3 and ng > 1e-3:
            dot = (tm / nm) @ (tg / ng)
            dir_errs.append(torch.rad2deg(torch.acos(torch.clamp(dot, -1, 1))))
    return {
        "mean_rot": torch.nanmean(torch.stack(rot_errs)).item() if rot_errs else float("nan"),
        "mean_dir": torch.nanmean(torch.stack(dir_errs)).item() if dir_errs else float("nan"),
        "s_align": float(s),
    }
