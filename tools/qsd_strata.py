#!/usr/bin/env python3
"""QSD analysis stratified by encounter closeness (RIGS-Net lambda = 0 vs a fixed lambda, default 0.2;
--compare selected gives the validation-selected lambda instead).

Strata are defined from the GROUND TRUTH only, never from a prediction: for every ordered vessel pair
(receiver i, sender j) of a test scene, r_gt = min over the future steps of d_ij / d_safe_i, with the receiver's
QSD (per-vessel AIS length/beam, course from consecutive positions, as in tools/qsd_eval.py).
  bins of r_gt: [0, 1) GT enters the domain, [1, 1.5), [1.5, 2), [2, 4), [4, inf)
Per stratum and run: number of pairs; pair intrusion rate (share of pairs whose prediction enters the receiver's
QSD at least once); pair/time VR; mean predicted r; mean ADE of the receiver over those pairs.

    python tools/qsd_strata.py eval  --pattern "*/rigs-lam*/seed*/test_dump.npz"   # cached in results/qsd_strata.json
    python tools/qsd_strata.py table --out results/tab_qsd_strata.tex
"""
import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rigs.config import DATA_ROOT, DATASETS, LAMBDAS  # noqa: E402
from rigs.dims import load_seg_dims  # noqa: E402
from rigs.qsd import future_forward, qsd_radius  # noqa: E402

BINS = [0.0, 1.0, 1.5, 2.0, 4.0, np.inf]
GEQ, LT = "$\\geq$", "$<$"
LABELS = ["<1 (GT inside)", "1-1.5", "1.5-2", "2-4", ">=4"]


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    fb = np.zeros_like(v); fb[..., 1] = 1.0
    return np.where(n > 1e-3, v / np.maximum(n, 1e-3), fb)


def ratios(P, p_last, f_last, L, B):
    """P [M, T, 2] -> r [M, M, T] = d_ij / d_safe_i (receiver rows), diagonal = inf."""
    with torch.no_grad():
        Pt = torch.from_numpy(P[None]).double()
        fwd = future_forward(Pt, torch.from_numpy(p_last[None]).double(), torch.from_numpy(f_last[None]).double())
        rel = Pt[:, None] - Pt[:, :, None]
        Lt = torch.as_tensor(L, dtype=torch.float64)[None, :, None, None]
        Bt = torch.as_tensor(B, dtype=torch.float64)[None, :, None, None]
        d_safe, d = qsd_radius(rel, fwd[:, :, None].expand_as(rel), Lt, Bt, eps=0.0)
        r = (d / d_safe)[0].numpy()
    M = len(P)
    r[np.arange(M), np.arange(M)] = np.inf
    return r


def eval_dump(path, seg_dims):
    z = np.load(path)
    pred, gt, hist, scene, seg = z["pred"][:, 0].astype(np.float64), z["gt"].astype(np.float64), z["hist"], z["scene"], z["seg"]
    p_last, f_last = hist[:, -1, :2].astype(np.float64), unit(hist[:, -1, 2:].astype(np.float64))
    ade = np.linalg.norm(pred - gt, axis=-1).mean(-1)
    K = len(LABELS)
    acc = dict(pairs=np.zeros(K), intr_pred=np.zeros(K), intr_gt=np.zeros(K), viol_states=np.zeros(K),
               states=np.zeros(K), r_pred=np.zeros(K), ade_recv=np.zeros(K))
    order = np.argsort(scene, kind="stable")
    for idx in np.split(order, np.flatnonzero(np.diff(scene[order])) + 1):
        M = len(idx)
        if M < 2:
            continue
        L, B = seg_dims[0][seg[idx]], seg_dims[1][seg[idx]]
        rg = ratios(gt[idx], p_last[idx], f_last[idx], L, B)
        rp = ratios(pred[idx], p_last[idx], f_last[idx], L, B)
        off = ~np.eye(M, dtype=bool)
        rgm, rpm = rg.min(-1)[off], rp.min(-1)[off]
        vs = (rp < 1).sum(-1)[off]
        recv_ade = np.broadcast_to(ade[idx][:, None], (M, M))[off]
        b = np.digitize(rgm, BINS[1:-1])
        T = rp.shape[-1]
        for k in range(K):
            s = b == k
            if not s.any():
                continue
            acc["pairs"][k] += s.sum(); acc["intr_pred"][k] += (rpm[s] < 1).sum(); acc["intr_gt"][k] += (rgm[s] < 1).sum()
            acc["viol_states"][k] += vs[s].sum(); acc["states"][k] += s.sum() * T
            acc["r_pred"][k] += np.minimum(rpm[s], 10).sum(); acc["ade_recv"][k] += recv_ade[s].sum()
    return {k: v.tolist() for k, v in acc.items()}


def summarize(a):
    a = {k: np.asarray(v, float) for k, v in a.items()}
    n = np.maximum(a["pairs"], 1)
    return dict(pairs=a["pairs"], intr=100 * a["intr_pred"] / n, vr=100 * a["viol_states"] / np.maximum(a["states"], 1),
                r=a["r_pred"] / n, ade=a["ade_recv"] / n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval", "table"])
    ap.add_argument("--runs", default="runs_dma")
    ap.add_argument("--cache", default="results/qsd_strata.json")
    ap.add_argument("--pattern", default="*/rigs-lam*/seed*/test_dump.npz")
    ap.add_argument("--out", default="results/tab_qsd_strata.tex")
    ap.add_argument("--compare", default="0.2",
                    help="second row: a fixed lambda (default 0.2, as in the paper) or 'selected' (validation-selected)")
    a = ap.parse_args()
    res = json.load(open(a.cache)) if os.path.exists(a.cache) else {}
    if a.mode == "eval":
        for p in sorted(glob.glob(os.path.join(a.runs, a.pattern))):
            key = os.path.relpath(os.path.dirname(p), a.runs)
            if key in res and res[key].get("mtime") == os.path.getmtime(p):
                continue
            r = eval_dump(p, load_seg_dims(os.path.join(DATA_ROOT, key.split(os.sep)[0])))
            r["mtime"] = os.path.getmtime(p); res[key] = r
            json.dump(res, open(a.cache, "w"), indent=1)
            s = summarize(r)
            print(key, "pairs", s["pairs"].astype(int).tolist(), "intr%", np.round(s["intr"], 2).tolist(), flush=True)
        sys.exit(0)
    spec = importlib.util.spec_from_file_location("mt", os.path.join(os.path.dirname(__file__), "make_tables.py"))
    mt = importlib.util.module_from_spec(spec); spec.loader.exec_module(mt)
    # pooled over regions and seeds: for each region the seeds present for BOTH rows
    tot = {0: None, 1: None}; used = []
    for ds in DATASETS:
        if not all(os.path.exists(os.path.join(a.runs, ds, f"rigs-lam{l:g}", "seed0", "test_metrics.json")) for l in LAMBDAS):
            continue
        lam = mt.select_lambda(a.runs, ds)[0] if a.compare == "selected" else float(a.compare)
        seeds = sorted({k.split(os.sep)[2] for k in res if k.startswith(f"{ds}{os.sep}rigs-lam0{os.sep}")} &
                       {k.split(os.sep)[2] for k in res if k.startswith(f"{ds}{os.sep}rigs-lam{lam:g}{os.sep}")})
        if not seeds:
            continue
        if lam == 0:
            print(f"note: {ds} selected lambda = 0, so both rows are the same run")
        for s in seeds:
            for row, name in ((0, "rigs-lam0"), (1, f"rigs-lam{lam:g}")):
                r = {k: np.asarray(v, float) for k, v in res[os.path.join(ds, name, s)].items() if k != "mtime"}
                tot[row] = r if tot[row] is None else {k: tot[row][k] + r[k] for k in r}
        used.append(f"{ds}(lambda={lam:g}, {','.join(seeds)})")
    if tot[0] is None:
        print("no region with a finished lambda sweep and diagnostics"); sys.exit(1)
    s0, s1 = summarize(tot[0]), summarize(tot[1])
    print("pooled over:", "; ".join(used))
    print(f"{'GT min d/d_safe':16s} {'pairs':>10s} | {'intr% lam0':>10s} {'intr% cmp':>10s} | {'VR% lam0':>9s} {'VR% cmp':>9s} | "
          f"{'ADE lam0':>8s} {'ADE cmp':>8s}")
    rows = []
    for k, lab in enumerate(LABELS):
        print(f"{lab:16s} {int(s0['pairs'][k]):10d} | {s0['intr'][k]:10.2f} {s1['intr'][k]:10.2f} | {s0['vr'][k]:9.3f} "
              f"{s1['vr'][k]:9.3f} | {s0['ade'][k]:8.2f} {s1['ade'][k]:8.2f}")
        tex_lab = lab.replace(">=", GEQ).replace("<", LT)
        rows.append(f"{tex_lab} & {int(s0['pairs'][k]):,} & {s0['intr'][k]:.2f}\\% & "
                    f"{s1['intr'][k]:.2f}\\% & {s0['vr'][k]:.3f}\\% & {s1['vr'][k]:.3f}\\% & {s0['ade'][k]:.2f} & {s1['ade'][k]:.2f} \\\\")
    open(a.out, "w").write("% pooled over: " + "; ".join(used) + "\n" + "\n".join(rows) + "\n")
