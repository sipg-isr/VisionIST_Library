"""Sanity test for linear_reconstruction (the ``linear`` solver): same
synthetic scene as test_projective_reconstruction (per-frame affine depth
corruption), complete / banded missing / heavier banded missing / missing +
noise. Without the offset prior, noiseless data must be recovered exactly —
cameras AND the depth affine (o_0 is estimated, not anchored); with the
default prior the bias must stay small.

Run:  python test/test_linear_reconstruction.py
"""
import os
import sys

import numpy as np
import torch

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_TEST_DIR, "..", "src"))
sys.path.insert(0, _TEST_DIR)

from linear_reconstruction import run_linear_reconstruction  # noqa: E402
from projective_reconstruction import compare_cameras  # noqa: E402
from test_projective_reconstruction import banded_missing, make_synthetic  # noqa: E402


def check(label, W, lam, gt, d_true, o_true, max_rot, max_dir, max_affine, **kw):
    rec = run_linear_reconstruction(W.double().numpy(), lam.double().numpy(), **kw)
    kept = np.flatnonzero(rec["vf"])
    ev = compare_cameras([torch.from_numpy(c) for c in rec["cameras"]],
                         [gt[i].double() for i in kept])
    # the affine is recovered up to the global scale (gauge d_0 = 1)
    s = rec["d"][0] / d_true[kept[0]].item()
    err_d = np.abs(rec["d"] / s - d_true[kept].numpy()).max()
    err_o = np.abs(rec["o"] / s - o_true[kept].numpy()).max()
    missing = torch.isnan(lam).float().mean().item()
    print(f"{label:24s} missing={missing:5.1%} frames={len(kept)} points={int(rec['vp'].sum())} "
          f"iters={rec['info']['iterations']:2d} | rot {ev['mean_rot']:.4f} deg  dir {ev['mean_dir']:.4f} deg"
          f" | d err {err_d:.1e}  o err {err_o:.1e}")
    assert ev["mean_rot"] < max_rot and ev["mean_dir"] < max_dir, (label, ev)
    assert err_d < max_affine and err_o < max_affine, (label, err_d, err_o)


def main():
    # float64 data: in float32 the ground-truth rotations are orthonormal to
    # ~1e-7 only, which the arccos of the rotation metric turns into ~0.01 deg
    torch.set_default_dtype(torch.float64)
    W, lam, gt, d_true, o_true, *_ = make_synthetic(F=10, P=300)
    g = torch.Generator().manual_seed(5)
    W_n = W + 0.003 * torch.randn(W.shape, generator=g)

    exact = dict(offset_prior=0.0)
    check("complete", W, lam, gt, d_true, o_true, 1e-3, 1e-3, 1e-5, **exact)
    check("banded missing", W, banded_missing(lam), gt, d_true, o_true, 1e-3, 1e-3, 1e-5, **exact)
    check("banded missing, len>=2", W, banded_missing(lam, min_len=2), gt, d_true, o_true,
          1e-3, 1e-3, 1e-5, **exact)
    check("complete, default prior", W, lam, gt, d_true, o_true, 0.1, 0.1, 0.1)
    check("banded missing + noise", W_n, banded_missing(lam), gt, d_true, o_true, 0.5, 0.5, 0.1)
    print("PASS")


if __name__ == "__main__":
    main()
