"""Sanity test for projective_reconstruction: recovery on synthetic data with
per-frame affine depth corruption — complete, with banded missing entries
(tracks that live on a contiguous run of frames), and missing + noise.

Run:  python test/test_projective_reconstruction.py
"""
import math
import os
import sys

import torch

# the core is vendored under ../src (the box ships it next to the service)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from projective_reconstruction import compare_cameras, run_projective_reconstruction  # noqa: E402


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]])


def make_synthetic(F=6, P=300, d_scale=0.5, o_scale=0.1, seed=0):
    torch.manual_seed(seed)
    # cameras on a circular orbit of radius 8 (real translation + rotation)
    Rt = torch.stack([rot_z(f * 0.6 + 0.3) @ rot_x(-0.25) for f in range(F)])
    C = torch.stack([torch.tensor([8.0 * math.cos(f * 0.6 + 0.3),
                                  8.0 * math.sin(f * 0.6 + 0.3),
                                  -1.5]) for f in range(F)])
    tt = [(-Rt[f] @ C[f]) for f in range(F)]

    # point cloud near the origin (radius ~2)
    Xall = torch.randn(3, P * 4) * 1.5
    Xc_all = [Rt[f] @ Xall + tt[f].unsqueeze(1) for f in range(F)]
    Z_all = torch.stack([Xc_all[f][2] for f in range(F)])
    idx = torch.nonzero(Z_all.min(0).values > 1.5).flatten()[:P]
    X = Xall[:, idx]
    Xc_all = [c[:, idx] for c in Xc_all]
    Z_all = torch.stack([c[2] for c in Xc_all])

    d_true = 1 + d_scale * torch.randn(F)
    o_true = o_scale * torch.randn(F)

    W = torch.empty(3 * F, X.shape[1]); lam = torch.empty(F, X.shape[1])
    for f in range(F):
        Z = Z_all[f]
        lam[f] = (Z - o_true[f]) / d_true[f]
        W[3 * f:3 * f + 3] = torch.stack([Xc_all[f][0] / Z, Xc_all[f][1] / Z,
                                         torch.ones_like(Z)])
    gt = [torch.cat([Rt[f], tt[f].unsqueeze(1)], dim=1) for f in range(F)]
    return W, lam, gt, d_true, o_true, Rt, C, X


def banded_missing(lam, seed=1, min_len=3):
    """NaN-out entries so every track is seen on one contiguous (cyclic) run
    of >= min_len frames — the shape real tracker output has."""
    F, P = lam.shape
    g = torch.Generator().manual_seed(seed)
    start = torch.randint(0, F, (P,), generator=g)
    length = torch.randint(min_len, F + 1, (P,), generator=g)
    seen = ((torch.arange(F)[:, None] - start) % F) < length
    out = lam.clone()
    out[~seen] = float("nan")
    return out


def check(label, W, lam, gt, max_rot, max_dir):
    rec = run_projective_reconstruction(W, lam)
    kept = torch.nonzero(rec["vf"]).flatten().tolist()
    ev = compare_cameras([c.float() for c in rec["cam_lists"]], [gt[i] for i in kept])
    missing = torch.isnan(lam).float().mean().item()
    print(f"{label:22s} missing={missing:5.1%} frames={len(kept)} points={int(rec['vp'].sum())} "
          f"| rot {ev['mean_rot']:.3f} deg  dir {ev['mean_dir']:.3f} deg")
    assert ev["mean_rot"] < max_rot and ev["mean_dir"] < max_dir, (label, ev)


def main():
    W, lam, gt, *_ = make_synthetic(F=10, P=300)
    lam_m = banded_missing(lam)
    g = torch.Generator().manual_seed(5)
    W_n = W + 0.003 * torch.randn(W.shape, generator=g)

    check("complete", W, lam, gt, 1.5, 1.0)
    check("banded missing", W, lam_m, gt, 1.5, 1.0)
    check("banded missing + noise", W_n, lam_m, gt, 2.0, 1.5)
    print("PASS")


if __name__ == "__main__":
    main()
