#!/usr/bin/env python3
"""All tables of the manuscript, computed from the run directories (nothing is typed in by hand).

* lambda_safe of RIGS-Net is selected per region on VALIDATION ADE (mean over seeds of the best validation
  epoch); the selected run is the "full" model. The test split plays no role in any selection.
* Values are mean +- sample std over seeds; the bold entry of a column is whichever row is lowest.
* K = 1 table: deterministic models report their single output; stochastic baselines report the mean of
  their predictive distribution (test_mean_metrics.json) = ADE_1 / FDE_1 of the manuscript.

    python tools/make_tables.py --runs runs --out results
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rigs.config import DATA_ROOT, DATASETS, LAMBDAS, MODEL_LABELS, STOCHASTIC  # noqa: E402

BASELINES = ["sstgcnn", "stmgcn", "gstgan", "mrgtraj", "aisformer", "mstformer", "dgcntf"]


def runs_of(runs, ds, name):
    return sorted(glob.glob(os.path.join(runs, ds, name, "seed*")))


def metric(d, key, tag="test"):
    p = os.path.join(d, f"{tag}_metrics.json")
    return json.load(open(p))["summary"][key] if os.path.exists(p) else None


def best_val(d):
    p = os.path.join(d, "history.json")
    return min(r["val_ADE"] for r in json.load(open(p))) if os.path.exists(p) else None


def agg(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None, 0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0), len(v)


def select_lambda(runs, ds):
    scores = {}
    for lam in LAMBDAS:
        v = [best_val(d) for d in runs_of(runs, ds, f"rigs-lam{lam:g}")
             if os.path.exists(os.path.join(d, "test_metrics.json"))]       # finished runs only (no partial histories)
        v = [x for x in v if x is not None]
        if v:
            scores[lam] = float(np.mean(v))
    if not scores:
        return None, scores
    return min(scores, key=scores.get), scores


def cell(m, s, n, bold=False, prec=2):
    if m is None:
        return "--"
    t = f"{m:.{prec}f}" + (f"$\\pm${s:.{prec}f}" if n > 1 else "")
    return f"\\textbf{{{t}}}" if bold else t


def table(rows, cols, title, prec=2):
    """rows: list of (label, {col: (mean, std, n)}). Returns (markdown, latex rows)."""
    best = {c: min((r[1][c][0] for r in rows if r[1].get(c) and r[1][c][0] is not None), default=None) for c in cols}
    md = [f"### {title}", "", "| Model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    tex = []
    for lab, vals in rows:
        cells_md, cells_tex = [], []
        for c in cols:
            m, s, n = vals.get(c, (None, None, 0))
            b = m is not None and best[c] is not None and abs(m - best[c]) < 1e-12
            cells_md.append("--" if m is None else (f"**{m:.{prec}f}**" if b else f"{m:.{prec}f}") + (f" ± {s:.{prec}f}" if n > 1 else "") + (f" (n={n})" if 0 < n < 3 else ""))
            cells_tex.append(cell(m, s, n, b, prec))
        md.append(f"| {lab} | " + " | ".join(cells_md) + " |")
        tex.append(f"{lab} & " + " & ".join(cells_tex) + " \\\\")
    return "\n".join(md) + "\n", "\n".join(tex) + "\n"


def per_model(runs, name, key, tag="test", datasets=DATASETS):
    out, per_seed = {}, {}
    for ds in datasets:
        ds_runs = runs_of(runs, ds, name)
        vals = [metric(d, key, tag) for d in ds_runs]
        out[ds] = agg(vals)
        for d, v in zip(ds_runs, vals):
            if v is not None:
                per_seed.setdefault(os.path.basename(d), {})[ds] = v
    complete = [s for s in per_seed.values() if len(s) == len(datasets)]
    out["Avg"] = agg([np.mean(list(s.values())) for s in complete])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_dma")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    md, tex = [], {}

    sel = {ds: select_lambda(a.runs, ds) for ds in DATASETS}
    md.append("## lambda_safe selected on validation ADE (mean over seeds)\n")
    md.append("| Region | " + " | ".join(f"λ={l:g}" for l in LAMBDAS) + " | selected |")
    md.append("|---" * (len(LAMBDAS) + 2) + "|")
    for ds in DATASETS:
        lam, sc = sel[ds]
        md.append(f"| {ds} | " + " | ".join(f"{sc[l]:.3f}" if l in sc else "--" for l in LAMBDAS) + f" | {lam} |")
    md.append("")
    full_name = {ds: (f"rigs-lam{sel[ds][0]:g}" if sel[ds][0] is not None else None) for ds in DATASETS}
    json.dump({ds: sel[ds][0] for ds in DATASETS}, open(os.path.join(a.out, "lambda_selected.json"), "w"), indent=1)

    def rigs_full(key, tag="test"):
        out, per_seed = {}, {}
        for ds in DATASETS:
            if full_name[ds] is None:
                out[ds] = (None, None, 0)
                continue
            rr = runs_of(a.runs, ds, full_name[ds])
            vals = [metric(d, key, tag) for d in rr]
            out[ds] = agg(vals)
            for d, v in zip(rr, vals):
                if v is not None:
                    per_seed.setdefault(os.path.basename(d), {})[ds] = v
        out["Avg"] = agg([np.mean(list(s.values())) for s in per_seed.values() if len(s) == len(DATASETS)])
        return out

    cols = DATASETS + ["Avg"]
    # ---------------- main comparison, single output (K = 1)
    for key, nm in (("meanADE", "ADE1"), ("meanFDE", "FDE1")):
        rows = [(MODEL_LABELS["cv"], per_model(a.runs, "cv", key))]
        for m in BASELINES:
            rows.append((MODEL_LABELS[m], per_model(a.runs, m, key[4:], tag="test_mean") if m in STOCHASTIC
                         else per_model(a.runs, m, key)))
        rows.append((MODEL_LABELS["rigs"], rigs_full(key)))
        t_md, t_tex = table(rows, cols, f"Main comparison, single output: {nm} (m)")
        md.append(t_md); tex[f"main_{nm}"] = t_tex
    # ---------------- best-of-20 of the stochastic baselines
    for key, nm in (("ADE", "minADE20"), ("FDE", "minFDE20")):
        rows = [(MODEL_LABELS[m], per_model(a.runs, m, key)) for m in BASELINES if m in STOCHASTIC]
        rows.append((MODEL_LABELS["rigs"] + " (K=1)", rigs_full("meanADE" if key == "ADE" else "meanFDE")))
        t_md, t_tex = table(rows, cols, f"Stochastic baselines, best-of-20: {nm} (m); RIGS-Net single output for reference")
        md.append(t_md); tex[f"bo20_{nm}"] = t_tex
    # ---------------- ablation
    for key, nm in (("meanADE", "ADE"), ("meanFDE", "FDE")):
        rows = [("Baseline", per_model(a.runs, "rigs-base", key)),
                ("Baseline + Risk", per_model(a.runs, "rigs-risk", key)),
                ("Baseline + Risk + QSD", per_model(a.runs, "rigs-lam0", key)),
                ("RIGS-Net (full)", rigs_full(key)),
                ("Baseline + Risk + QSD, all vessels as candidates", per_model("runs_allnbr_dma", "rigs-lam0", key))]
        t_md, t_tex = table(rows, cols, f"Ablation: {nm} (m)")
        md.append(t_md); tex[f"abl_{nm}"] = t_tex
    # ---------------- lambda sensitivity (shared lambda)
    for key, nm in (("meanADE", "ADE"), ("meanFDE", "FDE")):
        rows = [(f"λ={lam:g}", per_model(a.runs, f"rigs-lam{lam:g}", key)) for lam in LAMBDAS]
        rows.append(("Val.-selected", rigs_full(key)))
        t_md, t_tex = table(rows, cols, f"Sensitivity to a shared lambda_safe: {nm} (m)")
        md.append(t_md); tex[f"lam_{nm}"] = t_tex

    open(os.path.join(a.out, "tables.md"), "w").write("\n".join(md))
    json.dump(tex, open(os.path.join(a.out, "tables_tex.json"), "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
