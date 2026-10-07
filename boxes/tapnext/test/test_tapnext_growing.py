#!/usr/bin/env python3
"""
In-process tests for the TAPNext box's `new_tracks` (growing point set) path.

What this proves (no GPU, no `tapnet` wheel — a deterministic stub model is
injected via `PipelineService(model_factory=...)`):

  1. growth  — empty grid cells are re-seeded every `add_interval` frames,
     each seed = new generation with fresh track ids; response rows grow
     (padded tensors + `birth_frames` + `num_generations` in config)
  2. column/identity — column i == track id i, stable across frames and
     generations (gen0's columns are never shifted by later seeds)
  3. retirement — a fully-invisible generation is dropped after
     `retire_after_invisible` frames and *stops costing model calls*
  4. backfill  — a backfilled generation writes its columns into the
     already-accumulated earlier frames (trajectory from frame 0)
  5. budgets   — `max_new_per_seed` and `max_total_points` bound the growth
  6. reset / parking — both clear the growing state to a fresh generation 0
  7. isolation — sessions don't leak each other's generations
  8. back-compat — `new_tracks` unset: legacy single-grid behaviour, byte-compatible

Run:
    cd boxes/tapnext && python test/test_tapnext_growing.py
"""

import io
import json
import os
import sys
import time

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
BOX_DIR = os.path.dirname(TEST_DIR)
sys.path.insert(0, os.path.join(BOX_DIR, "protos"))
sys.path.insert(0, os.path.join(BOX_DIR, "src"))

import numpy as np
import cv2
import torch

import pipeline_pb2 as pb2
import aux
import tapnext_service as ts


# --------------------------------------------------------------------------
# Deterministic stub TAPNext
# --------------------------------------------------------------------------
class GrowStub:
    """Deterministic stand-in for TAPNext, mimicking BOTH call signatures the
    service uses: init `(video, query_points)` (T may be > 1 for backfill) and
    step `(video, state)` (T == 1).

    Encoded behaviour (so the tests can assert on the *structure*):
      * a point seeded at query `(x, y)` is at position `(x + t, y + t)` on
        frame `t` (init over a T-frame prefix) / after `c` steps (c == the
        number of frames tracked since its seed).
      * visibility is by LOCAL point index parity: even idx visible, odd idx
        hidden — independent of time. `visible="none"` hides everything
        (drives the retirement tests).
      * counters let the tests assert exact numbers of model forward calls.
    """

    def __init__(self, device="cpu", visible="parity"):
        self.device = device
        self.visible = visible
        self.init_calls = 0
        self.step_calls = 0

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, video=None, query_points=None, state=None):
        T = video.shape[1]
        dev = video.device
        if query_points is not None:
            self.init_calls += 1
            n = query_points.shape[1]
            q = query_points.squeeze(0)[:, 1:].to(dev)   # (n, 2) = (x, y)
            new_state = {"n": n, "q": q, "steps": 0}
            # model contract: tracks channels are (y, x) — flip the query
            pos = torch.stack([q[:, 1], q[:, 0]], dim=-1)
            tracks = pos[None, None, :, :] + torch.arange(T, dtype=torch.float32, device=dev).view(1, T, 1, 1)
            return tracks, torch.zeros(1, T, n, device=dev), self._vis(T, n, dev), new_state
        # step call
        state["steps"] += 1
        self.step_calls += 1
        n, q = state["n"], state["q"].to(dev)
        c = float(state["steps"])
        pos = torch.stack([q[:, 1], q[:, 0]], dim=-1)
        tracks = pos[None, None, :, :] + c
        return tracks, torch.zeros(1, 1, n, device=dev), self._vis(1, n, dev), state

    def _vis(self, T, n, dev):
        # Match the REAL TAPNext wire shape: visible_logits carry a trailing
        # logit channel -> (1, T, n, 1). The service must normalize it.
        if self.visible == "none":
            return torch.zeros(1, T, n, 1, device=dev)
        v = torch.zeros(1, T, n, 1, device=dev)
        v[:, :, ::2, 0] = 1.0
        return v


FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  \u2713 {name}")
    else:
        FAILS.append(name)
        print(f"  \u2717 {name}  {detail}")


def make_service(visible="parity"):
    return ts.PipelineService(model_factory=lambda device: GrowStub(device, visible))


def make_frame_bytes():
    img = (np.random.rand(80, 100, 3) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


FRAME_H, FRAME_W = 80, 100
SY, SX = FRAME_H / 256.0, FRAME_W / 256.0   # the box's 256 -> original scaling


def call(svc, sid, frames=1, **params):
    cfg = {"tapnext": {"command": "track", "parameters": params}}
    if sid is not None:
        cfg["tapnext"]["session_id"] = sid
    data = {"images": aux.wrap_value([make_frame_bytes() for _ in range(frames)])}
    return svc.Process(pb2.Envelope(config_json=json.dumps(cfg), data=data), None)


def reset(svc, sid=None):
    cfg = {"tapnext": {"command": "reset"}}
    if sid is not None:
        cfg["tapnext"]["session_id"] = sid
    return svc.Process(pb2.Envelope(config_json=json.dumps(cfg)), None)


def tensor(resp, name):
    return torch.load(io.BytesIO(aux.unwrap_value(resp.data[name])), weights_only=False).numpy()


def cfg_of(resp):
    return json.loads(resp.config_json)["tapnext"]


def birth(resp):
    """The per-track birth list now lives in DATA (typed FloatList)."""
    if "birth_frames" not in resp.data:
        return None
    return [int(x) for x in aux.unwrap_value(resp.data["birth_frames"])]


# grid_size=4 points live at linspace(10, 246, 4) = {10, 88.67, 167.33, 246}
G4 = np.linspace(10.0, 246.0, 4)                  # coordinate values in 256-space
G4IDX = [G4[0] * SY, G4[0] * SX]                  # point 0 = (x=10, y=10) -> (10*sy, 10*sx)


def grow_params(**over):
    p = {"new_tracks": True, "grid_size": 4, "cell_size": 64, "min_dist": 4,
         "add_interval": 5, "max_new_per_seed": 2}
    p.update(over)
    return p


# --------------------------------------------------------------------------
def test_growth():
    print("1. growth: empty cells re-seeded, ids stable, padded tensors")
    svc = make_service()
    r = call(svc, "g1", frames=11, **grow_params())
    c = cfg_of(r)
    t = tensor(r, "tracks")
    v = tensor(r, "visibles")
    p = tensor(r, "observation_matrix")

    check("status done, new_tracks echoed", c.get("status") == "done" and c.get("new_tracks") is True, str(c))
    check("num_points == 20 (16 grid + 2 seeds x 2)", c.get("num_points") == 20, f"got {c.get('num_points')}")
    check("num_generations == 3 (grid, seed@5, seed@10)", c.get("num_generations") == 3,
          f"got {c.get('num_generations')}")
    check("birth_frames (data) == 16x@0 + [5,5] + [10,10]",
          birth(r) == [0] * 16 + [5, 5, 10, 10], str(birth(r))[:60])
    check("config carries the COMPACT birth_hist (not the raw list)",
          c.get("birth_hist") == {"0": 16, "5": 2, "10": 2}
          and "birth_frames" not in c, str(c.get("birth_hist")))
    check("tracks shape (11, 20, 2)", t.shape == (11, 20, 2), f"got {t.shape}")
    check("visibles shape (11, 20)", v.shape == (11, 20), f"got {v.shape}")
    check("observation_matrix (2*11, 20)", p.shape == (22, 20), f"got {p.shape}")

    # gen0 (cols 0..15): parity-visible, position (10+f, 10+f) 256-space at frame f
    check("gen0 row0 col0 == seeded (10,10) scaled",
          np.allclose(t[0, 0], G4IDX), str(t[0, 0]))
    check("gen0 frame3 col0 == (10+3) scaled both axes",
          np.allclose(t[3, 0], [13 * SY, 13 * SX]), str(t[3, 0]))
    check("gen0 visibility = even idx (all 11 frames)",
          bool(v[:, 0:16:2].astype(bool).all()) and not bool(v[:, 1:16:2].astype(bool).any()),
          f"v[0,:6]={v[0, :6]}")
    # not-yet-born padding (0.0 + vis 0)
    check("frames 0-4: cols 16..19 padded 0 / vis 0",
          not np.any(t[:5, 16:]) and not np.any(v[:5, 16:]), "rows 0-4 tail non-zero")
    check("frames 5-9: cols 18-19 padded; 16-17 live",
          not np.any(t[5:10, 18:]) and not np.any(v[5:10, 18:]) and np.all(v[5:10, 16] > 0),
          "seed-1 columns wrong")
    check("frame 10: all 20 cols live, col16 visible col17 hidden",
          np.all(v[10, 16] > 0) and v[10, 17] == 0 and np.any(t[10, 18:] != 0), "frame 10 wrong")
    # P matrix: x/y rows + NaN where not born (x row == the t-row's x column)
    check("P frame0: P[0]==t[0] x-col, P[1]==t[0] y-col (cols 0-15), tail NaN",
          np.allclose(p[0, :16], t[0, :16, 1]) and np.allclose(p[1, :16], t[0, :16, 0])
          and np.all(np.isnan(p[0, 16:])) and np.all(np.isnan(p[1, 16:])), str(p[0]))
    check("P frame10: all 20 cols finite", np.isfinite(p[20]).all() and np.isfinite(p[21]).all())
    check("P frame4 col16 NaN (not yet born), frame5 finite",
          np.isnan(p[8, 16]) and np.isfinite(p[10, 16]), f"{p[8, 16]} / {p[10, 16]}")
    # step cost: gen0 f1-f5 (5) + gen0+gen1 f6-f10 (2 x 5 = 10)
    check("stub saw exactly 3 init + 15 step forwards",
          svc._model.init_calls == 3 and svc._model.step_calls == 15,
          f"init={svc._model.init_calls} step={svc._model.step_calls}")


def test_retirement():
    print("2. retirement: dead generation stops costing model calls")
    svc = make_service(visible="none")
    r = call(svc, "dead", frames=11, **grow_params(retire_after_invisible=3, add_interval=10,
                                                   max_total_points=64))
    c = cfg_of(r)
    t = tensor(r, "tracks")
    check("status done", c.get("status") == "done", str(c))
    check("num_generations == 1 (gen0 retired, only the seed@10 remains)",
          c.get("num_generations") == 1, f"got {c.get('num_generations')}")
    check("gen0 columns zero from frame 4 on (retired at frame 3; 4..9 have no live gens)",
          not np.any(t[4:11, :16]), "retired cols non-zero")
    check("rows 0-3 still carry gen0 (position recorded while alive, vis all 0)",
          np.all(t[:4, :16] > 0) and not np.any(tensor(r, "visibles")[:4, :16]))
    check("seed@10 got fresh ids (cols 16-17), visible=none",
          birth(r)[16:] == [10, 10] and np.any(t[10, 16:] != 0))
    # the point of retirement: gen0 stopped costing forwards after 3 frames
    check("model stepped exactly 3 times (frames 1-3), not 10",
          svc._model.step_calls == 3, f"step_calls={svc._model.step_calls}")
    check("exactly 2 init calls (gen0 + seed@10)", svc._model.init_calls == 2,
          f"init_calls={svc._model.init_calls}")

    # control: same run WITHOUT retirement must step gen0 every frame
    svc2 = make_service(visible="none")
    call(svc2, "control", frames=11, **grow_params(retire_after_invisible=0, add_interval=10))
    check("control (no retirement) stepped 10 times", svc2._model.step_calls == 10,
          f"step_calls={svc2._model.step_calls}")


def test_backfill():
    print("3. backfill: new generation's columns are written into earlier frames")
    svc = make_service()
    r = call(svc, "bf", frames=6, **grow_params(backfill=True))
    c = cfg_of(r)
    t = tensor(r, "tracks")
    v = tensor(r, "visibles")
    check("status done", c.get("status") == "done", str(c))
    check("20 -> 18 cols (one seed of 2 at frame 5)", t.shape == (6, 18, 2) and
          birth(r) == [0] * 16 + [5, 5], f"shape={t.shape} birth={birth(r)[:20]}")
    # seed points: first empty cells (32,32) and (96,32); stub pos at frame f = q + f
    check("col16 backfilled: frame0 == (32,32) scaled, frame5 (birth) == (37,37) scaled",
          np.allclose(t[0, 16], [32 * SY, 32 * SX]) and np.allclose(t[5, 16], [37 * SY, 37 * SX]),
          f"f0={t[0, 16]} f5={t[5, 16]}")
    check("col17 backfilled but hidden all frames (odd local idx)",
          np.allclose(t[0, 17], [32 * SY, 96 * SX]) and not np.any(v[:, 17]), str(t[0, 17]))
    check("all 6 rows carry both seed columns (no padding holes in the backfill)",
          np.all(np.isfinite(t[:, 16:18])) and v.shape == (6, 18))
    check("exactly 2 inits: gen0 + the one backfilling seed (no double-seeds)",
          svc._model.init_calls == 2, f"init_calls={svc._model.init_calls}")


def test_budgets():
    print("4. budgets: max_total_points stops further seeding")
    svc = make_service()
    r = call(svc, "cap", frames=11, **grow_params(max_total_points=18))
    c = cfg_of(r)
    t = tensor(r, "tracks")
    check("num_points stops at 18 (16 + one seed of 2; no room for seed@10)",
          c.get("num_points") == 18 and c.get("num_generations") == 2,
          f"num_points={c.get('num_points')} gens={c.get('num_generations')}")
    check("frame 10: only 18 cols live, no cols 18+ appear", t.shape == (11, 18, 2), f"shape={t.shape}")

    r2 = call(svc, "new", frames=6, **grow_params(max_new_per_seed=1))
    c2 = cfg_of(r2)
    check("max_new_per_seed=1: exactly 1 point at frame 5 -> num_points 17",
          c2.get("num_points") == 17 and c2.get("num_generations") == 2,
          f"num_points={c2.get('num_points')}")


def test_reset_parking():
    print("5. reset & GPU parking both clear the growing state to a fresh gen0")
    svc = make_service()
    call(svc, "rw", frames=6, **grow_params())
    reset(svc, "rw")
    r = call(svc, "rw", frames=1, **grow_params())
    c = cfg_of(r)
    t = tensor(r, "tracks")
    check("after reset: fresh 16-col grid, 1 generation, birth all 0",
          t.shape == (1, 16, 2) and c.get("num_generations") == 1 and
          birth(r) == [0] * 16, f"shape={t.shape} gens={c.get('num_generations')}")
    check("after reset: col0 back at the seed position (new counter)",
          np.allclose(t[0, 0], G4IDX), str(t[0, 0]))

    # parking: tracking state resets (fresh gen0, ids restart at 0) while the
    # accumulated history is kept — the legacy parking semantics
    call(svc, "pk", frames=3, **grow_params())
    svc._last_request_time = time.time() - 200      # older than _IDLE_TIMEOUT
    svc._device = "cuda"
    svc._park_model_if_idle()
    r = call(svc, "pk", frames=1, **grow_params())
    c = cfg_of(r)
    t = tensor(r, "tracks")
    check("after parking: state re-seeded fresh (16 cols, 1 gen), history kept (4 frames)",
          t.shape == (4, 16, 2)
          and c.get("num_generations") == 1 and c.get("frames_processed") == 4
          and birth(r) == [0] * 16, str(c))
    check("after parking: last frame's col0 back at the seed position (new counter)",
          np.allclose(t[-1, 0], G4IDX), str(t[-1, 0]))


def test_isolation():
    print("6. isolation: two growing sessions, disjoint generations/ids")
    svc = make_service()
    rA = call(svc, "A", frames=6, **grow_params())
    rB = call(svc, "B", frames=3, **grow_params())
    cA, cB = cfg_of(rA), cfg_of(rB)
    tA, tB = tensor(rA, "tracks"), tensor(rB, "tracks")
    check("A: seeded at frame 5 (18 cols)", tA.shape == (6, 18, 2) and cA.get("num_generations") == 2,
          f"shape={tA.shape} gens={cA.get('num_generations')}")
    check("B: still under the seed horizon (16 cols, 1 gen), no cross-talk",
          tB.shape == (3, 16, 2) and cB.get("num_generations") == 1 and
          birth(rB) == [0] * 16, f"shape={tB.shape}")
    check("A's seed ids (16-17) do not appear in B",
          np.allclose(tB[0, 0], G4IDX) and not np.any(tB[:, 16:]))


def test_backcompat():
    print("7. back-compat: without new_tracks the legacy single-grid path is untouched")
    svc = make_service()
    r = call(svc, "legacy", frames=4, grid_size=4)
    c = cfg_of(r)
    t = tensor(r, "tracks")
    check("legacy: uniform (4, 16, 2), no growing keys in config or data",
          t.shape == (4, 16, 2) and "new_tracks" not in c and "birth_hist" not in c
          and "num_generations" not in c and "birth_frames" not in r.data, str(c))
    check("legacy: all 16 cols finite in every row (no padding)",
          bool(np.all(t != 0).all()), "found zeros")
    check("legacy: gen0 progress (counter==frame) still tracked",
          np.allclose(t[2, 0], [12 * SY, 12 * SX]), str(t[2, 0]))


def main():
    tests = [
        test_growth,
        test_retirement,
        test_backfill,
        test_budgets,
        test_reset_parking,
        test_isolation,
        test_backcompat,
    ]
    for t in tests:
        t()
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {FAILS}")
        sys.exit(1)
    print("All growing-path tests passed.")


if __name__ == "__main__":
    main()
