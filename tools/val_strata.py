#!/usr/bin/env python3
"""VALIDATION-only analysis: does the interaction channel help in close encounters?

Step 1 (dump): score a finished run's best.pt on the validation split and save val_strata_dump.npz
               (does not touch val_metrics.json).
Step 2 (table): stratify RECEIVERS by the ground-truth closeness of their nearest encounter,
               r_min = min over senders j and future steps t of d_ij / d_safe_i (receiver QSD, per-vessel L/B),
               and compare ADE/FDE of several runs on the same agents, with a paired scene-bootstrap 95% CI
               of the ADE difference (run - reference).

    python tools/val_strata.py dump  runs_dev_v2/RWA/rigs-dev-v2/seed0 runs_dev_v2/RWA/rigs-noint-dev/seed0 ...
    python tools/val_strata.py table --ref runs_dev_v2/RWA/rigs-noint-dev/seed0 runs_dma/RWA/rigs-lam0/seed0 ...
"""
import argparse
import json
import os
import sys
from dataclasses import fields

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BINS = [0.0, 1.0, 1.5, 2.0, 4.0, np.inf]
LABELS = ["<1 (GT inside)", "1-1.5", "1.5-2", "2-4", ">=4", "single vessel"]
DUMP = "val_strata_dump.npz"


def dump(dirs, device):
    import torch
    from rigs.config import DATA_ROOT, RConfig
    from aistraj.data.dataset import make_loaders
    from aistraj.engine import set_seed
    from rigs.dims import attach_dims
    from rigs.experiment import build_model, evaluate
    names = {f.name for f in fields(RConfig)}
    for d in dirs:
        d = d.rstrip("/")
        dataset, model = d.split("/")[-3], d.split("/")[-2].split("-")[0]
        raw = json.load(open(os.path.join(d, "config.json")))
        cfg = RConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items() if k in names})
        set_seed(cfg.seed)
        root = os.path.join(DATA_ROOT, dataset)
        loaders, sets, stats = make_loaders(root, cfg, stats_path=os.path.join(root, "stats.json"))
        net = build_model(model, cfg, stats)
        attach_dims(net, root)
        net = net.to(device)
        net.load_state_dict(torch.load(os.path.join(d, "best.pt"), map_location=device)["state"])
        dist, end, arr = evaluate(net, loaders["val"], 1, torch.device(device), seed=cfg.seed, dump=True)
        np.savez_compressed(os.path.join(d, DUMP), dist=dist.astype(np.float32), **arr)
        print(f"[dump] {d} N={len(arr['gt'])} ADE {np.linalg.norm(arr['pred'][:, 0] - arr['gt'], axis=-1).mean():.3f}",
              flush=True)


def receiver_rmin(z, seg_dims):
    from qsd_strata import ratios, unit
    gt, hist, scene, seg = z["gt"].astype(np.float64), z["hist"], z["scene"], z["seg"]
    p_last, f_last = hist[:, -1, :2].astype(np.float64), unit(hist[:, -1, 2:].astype(np.float64))
    rmin = np.full(len(gt), np.inf)
    order = np.argsort(scene, kind="stable")
    for idx in np.split(order, np.flatnonzero(np.diff(scene[order])) + 1):
        if len(idx) < 2:
            continue
        r = ratios(gt[idx], p_last[idx], f_last[idx], seg_dims[0][seg[idx]], seg_dims[1][seg[idx]])
        rmin[idx] = r.min(axis=(1, 2))
    return rmin


def table(ref, others, nboot=1000):
    from rigs.config import DATA_ROOT
    from rigs.dims import load_seg_dims
    zr = np.load(os.path.join(ref, DUMP))
    key = np.stack([zr["scene"], zr["seg"]], 1)
    dataset = ref.rstrip("/").split("/")[-3]
    rmin = receiver_rmin(zr, load_seg_dims(os.path.join(DATA_ROOT, dataset)))
    single = ~np.isfinite(rmin)
    b = np.where(single, len(BINS) - 1, np.digitize(np.where(single, 0, rmin), BINS[1:-1]))
    runs = [ref] + list(others)
    ade, fde = {}, {}
    for d in runs:
        z = np.load(os.path.join(d, DUMP))
        assert np.array_equal(np.stack([z["scene"], z["seg"]], 1), key), f"agent order differs: {d}"
        e = np.linalg.norm(z["pred"][:, 0] - z["gt"], axis=-1)
        ade[d], fde[d] = e.mean(-1), e[:, -1]
    name = {d: d.rstrip("/").split("/")[-2] for d in runs}
    rng = np.random.default_rng(0)
    scene = zr["scene"]
    print(f"{dataset} validation, reference = {name[ref]}; diff = run - ref, paired scene bootstrap 95% CI")
    print(f"{'GT r_min':16s} {'n':>7s} " + " ".join(f"{name[d][:14]:>14s}" for d in runs)
          + "   " + "   ".join(f"d({name[d][:10]}) [CI]" for d in others))
    out = {}
    for k, lab in enumerate(LABELS + ["ALL"]):
        s = np.ones(len(b), bool) if lab == "ALL" else b == k
        if not s.any():
            continue
        us, inv = np.unique(scene[s], return_inverse=True)
        row = f"{lab:16s} {s.sum():7d} " + " ".join(f"{ade[d][s].mean():14.3f}" for d in runs)
        out[lab] = dict(n=int(s.sum()), ade={name[d]: float(ade[d][s].mean()) for d in runs},
                        fde={name[d]: float(fde[d][s].mean()) for d in runs})
        for d in others:
            diff = ade[d][s] - ade[ref][s]
            ssum = np.bincount(inv, diff, len(us)); scnt = np.bincount(inv, None, len(us))
            bs = [(ssum[i].sum() / scnt[i].sum()) for i in (rng.integers(0, len(us), len(us)) for _ in range(nboot))]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            row += f"   {diff.mean():+7.3f} [{lo:+.3f},{hi:+.3f}]"
            out[lab][f"diff_{name[d]}"] = [float(diff.mean()), float(lo), float(hi)]
        print(row, flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dump", "table"])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--ref")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json")
    ap.add_argument("--dump", default=DUMP, help="table mode: dump file to read, e.g. test_dump.npz (written by run_jobs at test time)")
    a = ap.parse_args()
    DUMP = a.dump
    if a.mode == "dump":
        dump(a.dirs, a.device)
    else:
        res = table(a.ref, a.dirs)
        if a.json:
            json.dump(res, open(a.json, "w"), indent=1)
