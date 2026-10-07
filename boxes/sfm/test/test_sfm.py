#!/usr/bin/env python3
"""Test script for the SfM gRPC service (shared envelope interface).

Connects to a running sfm box and, on the synthetic scene from
``test_projective_reconstruction.py`` (orbiting cameras, per-frame
affine-corrupted depth):

  1. mode B, complete   — ``W_mat`` + ``lambda_mat``; cameras vs ground truth
  2. mode B, missing    — same, with ~35% of the entries NaN (banded, like
                          real tracks); still recovered
  3. mode A             — pixel ``tracks`` + painted ``depths`` maps +
                          ``intrinsics``, with NaN / out-of-image track
                          entries -> treated as missing entries
  4. the same three with ``solver: pairs`` (chained 2-frame solves +
     one rigid global pass of the completion ping-pong)
  5. the same three with ``solver: linear`` (no completion; near-exact on
     noiseless data -- the default offset prior biases it slightly)
  6. ``reset`` (no-op) and malformed requests (in-band ``status: error``)

Run (server already up):
    python test/test_sfm.py
    BOX_HOST=localhost:8061 python test/test_sfm.py
"""

import io
import json
import os
import sys

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_TEST_DIR, "..", "protos"))
sys.path.insert(0, os.path.join(_TEST_DIR, "..", "src"))

import grpc  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import pipeline_pb2  # noqa: E402
import pipeline_pb2_grpc  # noqa: E402
import aux  # noqa: E402

from projective_reconstruction import compare_cameras  # noqa: E402
from test_projective_reconstruction import banded_missing, make_synthetic  # noqa: E402


def npy(arr) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr))
    return buf.getvalue()


def call(stub, data, config):
    request = pipeline_pb2.Envelope(
        config_json=json.dumps(config),
        data={k: aux.wrap_value(npy(v)) for k, v in data.items()},
    )
    resp = stub.Process(request)
    sec = json.loads(resp.config_json).get("sfm", {})
    out = {k: np.load(io.BytesIO(aux.unwrap_value(v))) for k, v in resp.data.items()}
    return sec, out


def check_cameras(sec, out, gt, label, max_rot=1.5, max_dir=1.0):
    print(f"[{label}] status={sec.get('status')} solver={sec.get('solver')} "
          f"frames={sec.get('num_frames')}/{sec.get('num_frames_in')} "
          f"points={sec.get('num_points')}/{sec.get('num_points_in')} "
          f"missing in={sec.get('missing_in', 0):.1%} kept={sec.get('missing', 0):.1%} "
          f"iters={sec.get('iterations')} runtime={sec.get('runtime', 0):.3f}s")
    assert sec.get("status") == "done", sec
    for k, v in out.items():
        print(f"    {k:17s} {v.dtype} {v.shape}")
    cams = [torch.from_numpy(c).float() for c in out["cameras"]]
    ev = compare_cameras(cams, [gt[i] for i in out["frame_ids"]])
    rep = sec["reprojection_error"]
    print(f"    mean rot err {ev['mean_rot']:.3f} deg | mean dir err {ev['mean_dir']:.3f} deg | "
          f"reprojection median {rep['median']:.2e} {rep['unit']}")
    assert ev["mean_rot"] < max_rot and ev["mean_dir"] < max_dir, ev
    F, P = len(out["frame_ids"]), len(out["point_ids"])
    assert out["cameras"].shape == (F, 3, 4) and out["points"].shape == (P, 3)
    assert out["observed"].shape == (F, P) and out["completed_matrix"].shape == (3 * F, P)
    return ev


def check_depth_model(sec, out, lam):
    """Z = depth_scales * lambda + depth_offsets must be the depth of the
    reconstruction (the z rows of completed_matrix) on the observed entries:
    exactly for completion / pairs (they factorize those depths), up to the
    fit residual for linear. Gauges: completion -> o_0 = 0, smallest d ~ 1;
    pairs -> smallest d ~ 1 (o_0 free); linear -> d_0 = 1."""
    assert sec["depth_model"] == "Z = depth_scales * lambda + depth_offsets", sec
    lam_k = np.asarray(lam)[out["frame_ids"]][:, out["point_ids"]]
    Z = out["depth_scales"][:, None] * lam_k + out["depth_offsets"][:, None]
    obs = out["observed"]
    diff = np.abs(Z - out["completed_matrix"][2::3])[obs]
    print(f"    depth model: |d*lambda + o - Z_reconstructed| max {diff.max():.2e} "
          f"median {np.median(diff / np.abs(Z[obs])):.1e} (relative), "
          f"scale f0 {out['depth_scales'][0]:.3f} min {out['depth_scales'].min():.3f}, "
          f"offset f0 {out['depth_offsets'][0]:.1e}")
    if sec["solver"] == "linear":
        # completed_matrix is the model's R X + t: equal up to the fit residual
        # (here the small bias of the offset prior); gauge d_0 = 1
        assert np.median(diff / np.abs(Z[obs])) < 1e-3, diff
        assert abs(out["depth_scales"][0] - 1.0) < 1e-6
    else:
        assert diff.max() < 1e-4 * np.abs(Z[obs]).max(), diff.max()
        assert abs(out["depth_scales"].min() - 1.0) < 1e-3
        if sec["solver"] == "completion":
            assert abs(out["depth_offsets"][0]) < 1e-6


def main():
    target = os.getenv("BOX_HOST", "localhost:8061")
    print(f"Target: {target}")
    channel = grpc.insecure_channel(target, options=[
        ("grpc.max_send_message_length", -1),
        ("grpc.max_receive_message_length", -1),
    ])
    stub = pipeline_pb2_grpc.PipelineServiceStub(channel)
    reconstruct = {"sfm": {"command": "reconstruct"}}
    pairs = {"sfm": {"command": "reconstruct", "parameters": {"solver": "pairs"}}}
    linear = {"sfm": {"command": "reconstruct", "parameters": {"solver": "linear"}}}

    W, lam, gt, *_ = make_synthetic(F=10, P=300)
    F, P = lam.shape

    # --- 1. mode B, complete -------------------------------------------------
    sec, out = call(stub, {"W_mat": W.numpy(), "lambda_mat": lam.numpy()}, reconstruct)
    check_cameras(sec, out, gt, "W_mat complete")
    check_depth_model(sec, out, lam)
    assert sec["missing_in"] == 0.0

    # --- 2. mode B, ~35% missing (NaN) ---------------------------------------
    lam_m = banded_missing(lam)
    sec, out = call(stub, {"W_mat": W.numpy(), "lambda_mat": lam_m.numpy()}, reconstruct)
    check_cameras(sec, out, gt, "W_mat missing")
    check_depth_model(sec, out, lam_m)
    assert sec["missing_in"] > 0.3, sec
    assert not out["observed"].all(), "missing entries should stay unobserved"

    # --- 3. mode A: pixel tracks + depth maps + intrinsics --------------------
    # the synthetic scene is very wide-angle (|x| up to ~7), so fit the
    # principal point / image size to it with a 20 px margin
    x, y = W[0::3].numpy(), W[1::3].numpy()
    fpx, margin = 100.0, 20.0
    cx, cy = margin - fpx * x.min(), margin - fpx * y.min()
    Wd = int(np.ceil(cx + fpx * x.max() + margin))
    H = int(np.ceil(cy + fpx * y.max() + margin))
    K = np.array([[fpx, 0, cx], [0, fpx, cy], [0, 0, 1]])
    u, v = K[0, 0] * x + K[0, 2], K[1, 1] * y + K[1, 2]
    tracks = np.empty((2 * F, P)); tracks[0::2], tracks[1::2] = u, v

    depths = np.ones((F, H, Wd), dtype=np.float32)
    hits = np.zeros((F, H, Wd), dtype=int)
    corners = [(du, dv) for du in (0, 1) for dv in (0, 1)]
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    for f in range(F):
        # paint the 4 pixels around each track, so bilinear sampling is exact
        for du, dv in corners:
            depths[f, v0[f] + dv, u0[f] + du] = lam[f].numpy()
            np.add.at(hits[f], (v0[f] + dv, u0[f] + du), 1)
    # a track sharing a painted pixel with another one samples a wrong depth in
    # that frame: make that ENTRY missing (a fixture artifact, not data)
    for f in range(F):
        shared = np.zeros(P, dtype=bool)
        for du, dv in corners:
            shared |= hits[f, v0[f] + dv, u0[f] + du] > 1
        tracks[2 * f, shared] = np.nan
    # plus one NaN entry and one entry leaving the image
    tracks[2, 0] = np.nan
    tracks[5, 1] = H + 10

    sec, out = call(stub, {"tracks": tracks, "depths": depths, "intrinsics": K}, reconstruct)
    check_cameras(sec, out, gt, "tracks")
    expected_missing = (~np.isfinite(tracks[0::2])).sum() + 1      # + the out-of-image one
    assert round(sec["missing_in"] * F * P) == expected_missing, sec
    assert sec["reprojection_error"]["unit"] == "px"

    # --- 4. the pairs solver ----------------------------------------------------
    for label, data, l in [("pairs, W_mat complete", {"W_mat": W.numpy(), "lambda_mat": lam.numpy()}, lam),
                           ("pairs, W_mat missing", {"W_mat": W.numpy(), "lambda_mat": lam_m.numpy()}, lam_m),
                           ("pairs, tracks", {"tracks": tracks, "depths": depths, "intrinsics": K}, None)]:
        sec, out = call(stub, data, pairs)
        # the default offset ridge (1.0) biases this scene by ~1.5 deg
        check_cameras(sec, out, gt, label, max_rot=2.0, max_dir=1.5)
        if l is not None:
            check_depth_model(sec, out, l)
        assert sec["pairs_fallback"] == [], sec

    # --- 5. the linear solver ---------------------------------------------------
    sec, out = call(stub, {"W_mat": W.numpy(), "lambda_mat": lam.numpy()}, linear)
    check_cameras(sec, out, gt, "linear, W_mat complete", max_rot=0.05, max_dir=0.05)
    check_depth_model(sec, out, lam)
    sec, out = call(stub, {"W_mat": W.numpy(), "lambda_mat": lam_m.numpy()}, linear)
    check_cameras(sec, out, gt, "linear, W_mat missing", max_rot=0.05, max_dir=0.05)
    check_depth_model(sec, out, lam_m)
    sec, out = call(stub, {"tracks": tracks, "depths": depths, "intrinsics": K}, linear)
    check_cameras(sec, out, gt, "linear, tracks", max_rot=0.05, max_dir=0.05)
    assert sec["reprojection_error"]["median"] < 0.1, sec       # px (offset-prior bias only)

    # --- 6. reset + in-band errors ----------------------------------------------
    sec, _ = call(stub, {}, {"sfm": {"command": "reset"}})
    print(f"[reset] {sec}")
    assert sec == {"status": "done", "action": "reset"}

    sec, _ = call(stub, {"W_mat": np.zeros((5, 10)), "lambda_mat": np.zeros((1, 10))},
                  reconstruct)
    print(f"[bad shape] {sec}")
    assert sec["status"] == "error"

    sec, _ = call(stub, {"W_mat": W.numpy(), "lambda_mat": lam.numpy()},
                  {"sfm": {"parameters": {"solver": "magic"}}})
    print(f"[bad solver] {sec}")
    assert sec["status"] == "error"

    print("PASS")


if __name__ == "__main__":
    main()
