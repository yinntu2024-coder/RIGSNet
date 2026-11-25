#!/usr/bin/env python3
"""Table tab:safety_reg_analysis: RIGS-Net without (lambda_safe = 0) vs with the validation-selected lambda_safe.

ADE/FDE: K = 1 test metrics. QSD VR and mean minimum separation: tools/qsd_eval.py output (per-vessel AIS
dimensions). Mean +- sample std over the seeds available for BOTH rows of a region; the number of seeds is
reported. lambda is selected on validation ADE exactly as in tools/make_tables.py.

    python tools/qsd_eval.py --pattern "*/rigs-lam*/seed*/test_dump.npz" --out results/qsd.json
    python tools/safety_table.py --qsd results/qsd.json --out results/tab_safety_reg.tex
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rigs.config import DATASETS, LAMBDAS  # noqa: E402

_spec = importlib.util.spec_from_file_location("make_tables", os.path.join(os.path.dirname(__file__), "make_tables.py"))
mt = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mt)


def values(runs, qsd, ds, name, seeds):
    out = []
    for s in seeds:
        d = os.path.join(runs, ds, name, s)
        m = json.load(open(os.path.join(d, "test_metrics.json")))["summary"]
        q = qsd[os.path.join(ds, name, s)]
        out.append([m["meanADE"], m["meanFDE"], q["pred"]["VR"], q["pred"]["min_sep"], q["gt"]["VR"], q["gt"]["min_sep"]])
    return np.array(out)


def fmt(v, prec, unit=""):
    m = v.mean()
    s = f"{m:.{prec}f}{unit}"
    if len(v) > 1:
        s += f"$\\pm${v.std(ddof=1):.{prec}f}"
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_dma")
    ap.add_argument("--qsd", default="results/qsd.json")
    ap.add_argument("--out", default="results/tab_safety_reg.tex")
    a = ap.parse_args()
    qsd = json.load(open(a.qsd))
    rows, avg0, avg1, lines = [], [], [], []
    for ds in DATASETS:
        if not all(os.path.exists(os.path.join(a.runs, ds, f"rigs-lam{l:g}", "seed0", "test_metrics.json")) for l in LAMBDAS):
            lines.append(f"{ds}: lambda sweep incomplete -- skipped"); continue
        lam = mt.select_lambda(a.runs, ds)[0]
        sel = f"rigs-lam{lam:g}"

        def seeds_of(name):
            return {os.path.basename(d) for d in mt.runs_of(a.runs, ds, name)
                    if os.path.exists(os.path.join(d, "test_metrics.json")) and os.path.join(ds, name, os.path.basename(d)) in qsd}
        seeds = sorted(seeds_of("rigs-lam0") & seeds_of(sel))
        if not seeds:
            lines.append(f"{ds}: no seed with QSD diagnostics for both rows -- run tools/qsd_eval.py first"); continue
        v0, v1 = values(a.runs, qsd, ds, "rigs-lam0", seeds), values(a.runs, qsd, ds, sel, seeds)
        avg0.append(v0.mean(0)); avg1.append(v1.mean(0))
        rows.append(f"{ds} ($\\lambda$={lam:g}, n={len(seeds)}) & " + " & ".join(
            [fmt(v0[:, 0], 2), fmt(v0[:, 1], 2), fmt(v0[:, 2], 3, "\\%"), fmt(v0[:, 3], 1),
             fmt(v1[:, 0], 2), fmt(v1[:, 1], 2), fmt(v1[:, 2], 3, "\\%"), fmt(v1[:, 3], 1)]) + " \\\\")
        lines.append(f"{ds}: lambda={lam:g} seeds={seeds} | lam0 ADE {v0[:,0].mean():.2f} FDE {v0[:,1].mean():.2f} "
                     f"VR {v0[:,2].mean():.3f}% sep {v0[:,3].mean():.1f} | sel ADE {v1[:,0].mean():.2f} FDE {v1[:,1].mean():.2f} "
                     f"VR {v1[:,2].mean():.3f}% sep {v1[:,3].mean():.1f} | GT VR {v1[:,4].mean():.3f}% sep {v1[:,5].mean():.1f}")
    if len(avg0) == len(DATASETS):
        m0, m1 = np.mean(avg0, 0), np.mean(avg1, 0)
        rows.append("\\midrule")
        rows.append("Avg & " + " & ".join([f"{m0[0]:.2f}", f"{m0[1]:.2f}", f"{m0[2]:.3f}\\%", f"{m0[3]:.1f}",
                                           f"{m1[0]:.2f}", f"{m1[1]:.2f}", f"{m1[2]:.3f}\\%", f"{m1[3]:.1f}"]) + " \\\\")
        lines.append(f"Avg: lam0 VR {m0[2]:.3f}% sep {m0[3]:.1f} | sel VR {m1[2]:.3f}% sep {m1[3]:.1f} | GT VR {m1[4]:.3f}% sep {m1[5]:.1f}")
    open(a.out, "w").write("\n".join(rows) + "\n")
    print("\n".join(lines)); print("---- tex rows ----"); print("\n".join(rows))
