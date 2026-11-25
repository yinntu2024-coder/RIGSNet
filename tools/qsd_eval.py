#!/usr/bin/env python3
"""QSD diagnostics from the saved test predictions (manuscript Eq. qsd_vr and Table safety_reg_analysis).

For every multi-vessel test scene: ordered vessel pairs (i, j), i != j, at every future step; the receiver
course comes from consecutive predicted positions (last observed position first). VR = share of
pair/time states with separation < QSD radius of the receiver. Mean min. separation = mean over scenes
(with >= 2 vessels) of the minimum predicted pairwise distance. Sample 0 is used for stochastic models.
The same statistics on the ground-truth futures are reported as a reference.

    python tools/qsd_eval.py --runs runs --out results/qsd.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rigs.config import DATA_ROOT, RConfig  # noqa: E402
from rigs.dims import load_seg_dims  # noqa: E402
from rigs.qsd import scene_qsd_stats  # noqa: E402


def unit(v, fb=None):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    u = v / np.maximum(n, 1e-3)
    if fb is None:
        fb = np.zeros_like(v); fb[..., 1] = 1.0
    return np.where(n > 1e-3, u, fb)


def evaluate_dump(path, L, B, seg_dims=None):
    """L, B: default receiver dimensions; seg_dims: (L, B) per segment of the data set (overrides them)."""
    z = np.load(path)
    pred, gt, hist, scene = z["pred"][:, 0], z["gt"], z["hist"], z["scene"]
    p_last, f_last = hist[:, -1, :2], unit(hist[:, -1, 2:])
    acc = {"pred": [0, 0, []], "gt": [0, 0, []]}
    order = np.argsort(scene, kind="stable")
    bounds = np.flatnonzero(np.diff(scene[order])) + 1
    for idx in np.split(order, bounds):
        if len(idx) < 2:
            continue
        for key, P in (("pred", pred), ("gt", gt)):
            Li, Bi = (seg_dims[0][z["seg"][idx]], seg_dims[1][z["seg"][idx]]) if seg_dims is not None else (L, B)
            v, n, s = scene_qsd_stats(P[idx].astype(np.float64), p_last[idx].astype(np.float64),
                                      f_last[idx].astype(np.float64), Li, Bi)
            acc[key][0] += v; acc[key][1] += n; acc[key][2].append(s)
    out = {}
    for key, (v, n, s) in acc.items():
        out[key] = dict(VR=100.0 * v / max(n, 1), min_sep=float(np.mean(s)) if s else float("nan"),
                        pairs=int(n), scenes=len(s))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_dma")
    ap.add_argument("--out", default="results/qsd.json")
    ap.add_argument("--pattern", default="*/*/seed*/test_dump.npz")
    a = ap.parse_args()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for p in sorted(glob.glob(os.path.join(a.runs, a.pattern))):
        key = os.path.relpath(os.path.dirname(p), a.runs)
        if key in res and res[key].get("mtime") == os.path.getmtime(p):
            continue
        cfg = RConfig.from_dict(json.load(open(os.path.join(os.path.dirname(p), "config.json"))))
        ds = key.split(os.sep)[0]
        r = evaluate_dump(p, cfg.L_default, cfg.B_default, load_seg_dims(os.path.join(DATA_ROOT, ds)))
        r["mtime"] = os.path.getmtime(p)
        res[key] = r
        print(key, f"VR {r['pred']['VR']:.2f}% sep {r['pred']['min_sep']:.1f} | GT VR {r['gt']['VR']:.2f}% "
              f"sep {r['gt']['min_sep']:.1f} | pairs {r['pred']['pairs']}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
