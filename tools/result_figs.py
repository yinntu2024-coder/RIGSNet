#!/usr/bin/env python3
"""traj_compare.pdf and ablation_vis.pdf for the manuscript (DMA regions, seed 0, RIGS-Net v2).

Case selection never looks at any model prediction (ground truth / observations only):
  * traj_compare (unchanged rule of tools/figures.py): per region, among vessels with SOG >= 8 kn at the last
    observed and at the final future step, the trajectory whose course change is closest to the 99th percentile.
  * ablation_vis (rule fixed 2026-11-22, before any case was drawn): RWA test set (densest region, as the QSD
    case); ordered pairs (receiver i, sender j), both with SOG >= 5 kn at the last observed frame, whose
    ground-truth closest approach r = min_t d/d_safe_i lies in [1, 1.5) (close approach without entering the
    receiver's QSD); the pair with the median r (lower median, ties by order).
Predictions: deterministic models -> test_dump.npz; stochastic baselines -> test_mean_dump.npz (predictive mean,
the K = 1 output of the tables). RIGS-Net full = validation-selected lambda (results/lambda_selected.json).
Plotted values -> results/plot_data/{traj_compare,ablation_vis}.csv.

    python tools/result_figs.py --out results/figures_res
"""
import argparse
import importlib.util
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rigs.config import DATA_ROOT, DATASETS, MODEL_LABELS, STOCHASTIC  # noqa: E402
from rigs.dims import load_seg_dims  # noqa: E402


def _mod(name):
    s = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), f"{name}.py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


F = _mod("figures")                     # select_case, panel, METHODS, COLORS, KN
MF = _mod("mechanism_figs")             # gt_ratio
KN = F.KN
SEL = json.load(open("results/lambda_selected.json"))


def dump_of(runs, ds, name):
    m = name.split("-")[0]
    f = "test_mean_dump.npz" if m in STOCHASTIC else "test_dump.npz"
    p = os.path.join(runs, ds, name, "seed0", f)
    return np.load(p) if os.path.exists(p) else None


def panel(ax, n, dump, preds, title):
    """As tools/figures.py panel, but the ground truth is drawn on top of the predictions (zorder) so it stays visible."""
    h, g = dump["hist"][n, :, :2], dump["gt"][n]
    o = h[-1]
    ax.plot(*(h - o).T, "k-", lw=2, label="Observed", zorder=4)
    for lab, (p, c) in preds.items():
        ax.plot(*(np.vstack([h[-1:], p]) - o).T, "-", color=c, lw=1.4 if "RIGS" not in lab else 2.0, marker="o", ms=2,
                label=lab, zorder=3 if "RIGS" in lab else 2)
    ax.plot(*(np.vstack([h[-1:], g]) - o).T, "k--", lw=2, label="Ground truth", zorder=5)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("East (m)", fontsize=8); ax.set_ylabel("North (m)", fontsize=8)
    ax.tick_params(labelsize=7)


def fig_traj_compare(runs, out):
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8))
    rows, info = [], {}
    for ax, ds, lt in zip(axes, DATASETS, "abcde"):
        ref = dump_of(runs, ds, "cv")
        n, deg = F.select_case(ref)
        preds = {}
        for m in F.METHODS + ["rigs"]:
            name = f"rigs-lam{SEL[ds]:g}" if m == "rigs" else m
            d = dump_of(runs, ds, name)
            if d is None:
                continue
            assert np.allclose(d["gt"][n], ref["gt"][n]), f"dump rows are not aligned: {ds}/{name}"
            p = d["pred"][n, 0]
            preds[MODEL_LABELS[m]] = (p, F.COLORS[m])
            for t, q in enumerate(p):
                rows.append((ds, m, t + 1, *q))
        panel(ax, n, ref, preds, f"({lt}) {ds} (course change {deg:.0f}°)")
        for t, q in enumerate(ref["gt"][n]):
            rows.append((ds, "ground_truth", t + 1, *q))
        info[ds] = dict(agent=n, course_change_deg=round(deg, 1),
                        ade={k: round(float(np.linalg.norm(v[0] - ref["gt"][n], axis=-1).mean()), 2) for k, v in preds.items()})
    axes[0].legend(fontsize=6.5, loc="best")
    fig.tight_layout(); fig.savefig(os.path.join(out, "traj_compare.pdf"), bbox_inches="tight"); plt.close(fig)
    np.savetxt("results/plot_data/traj_compare.csv", np.array(rows, dtype=object), fmt="%s", delimiter=",",
               header="dataset,model,step,x_m,y_m", comments="")
    return info


ABL_DS = "RWA"


def scene_rows(z):
    order = np.argsort(z["scene"], kind="stable")
    return np.split(order, np.flatnonzero(np.diff(z["scene"][order])) + 1)


def pick_ablation_case(ref, segL, segB):
    cand = []
    for idx in scene_rows(ref):
        if len(idx) < 2:
            continue
        h, g = ref["hist"][idx].astype(np.float64), ref["gt"][idx].astype(np.float64)
        r = MF.gt_ratio(g, h, segL[ref["seg"][idx]], segB[ref["seg"][idx]]).min(-1)
        sog = np.linalg.norm(h[:, -1, 2:], axis=1)
        for a in range(len(idx)):
            for b in range(len(idx)):
                if a != b and sog[a] >= 5 * KN and sog[b] >= 5 * KN and 1.0 <= r[a, b] < 1.5:
                    cand.append((float(r[a, b]), int(idx[a]), int(idx[b])))
    cand.sort()
    m = cand[(len(cand) - 1) // 2]
    return dict(dataset=ABL_DS, gt_min_ratio=m[0], receiver_row=m[1], sender_row=m[2],
                scene=int(ref["scene"][m[1]]), n_candidates=len(cand))


def fig_ablation(runs, out):
    ds = ABL_DS
    segL, segB = load_seg_dims(os.path.join(DATA_ROOT, ds))
    var = [("No interaction", "rigs-noint", "#8c8c8c"), ("Baseline", "rigs-base", "#4e79a7"),
           ("Baseline + Risk", "rigs-risk", "#f28e2b"), (r"Baseline + Risk + QSD ($\lambda=0$)", "rigs-lam0", "#59a14f"),
           (f"RIGS-Net (full, $\\lambda={SEL[ds]:g}$)", f"rigs-lam{SEL[ds]:g}", "#e15759")]
    ref = dump_of(runs, ds, "rigs-noint")
    cache = "results/plot_data/ablation_case.json"            # GT-only pick, reused (delete to re-pick)
    case = json.load(open(cache)) if os.path.exists(cache) else pick_ablation_case(ref, segL, segB)
    json.dump(case, open(cache, "w"), indent=1)
    a, b = case["receiver_row"], case["sender_row"]
    idx = np.flatnonzero(ref["scene"] == case["scene"])
    ia, ib = int(np.flatnonzero(idx == a)[0]), int(np.flatnonzero(idx == b)[0])
    h, g = ref["hist"][idx].astype(np.float64), ref["gt"][idx].astype(np.float64)
    L, B = segL[ref["seg"][idx]], segB[ref["seg"][idx]]
    preds = {}
    for lab, name, c in var:
        d = dump_of(runs, ds, name)
        assert np.array_equal(d["scene"], ref["scene"]) and np.allclose(d["gt"][idx], g), name
        preds[lab] = (d["pred"][idx, 0].astype(np.float64), c)
    ratios = {"ground truth": MF.gt_ratio(g, h, L, B)[ia, ib]}
    for lab, (p, _) in preds.items():
        ratios[lab] = MF.gt_ratio(p, h, L, B)[ia, ib]
    o = h[ia, -1, :2]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.4), gridspec_kw=dict(width_ratios=[1.4, 1]))
    for k, (row, role) in enumerate(((ia, "receiver"), (ib, "sender"))):
        ax.plot(*(h[row, :, :2] - o).T, "k-", lw=2, label="observed" if k == 0 else None)
        ax.plot(*(np.vstack([h[row, -1:, :2], g[row]]) - o).T, "k:", lw=2, label="ground truth" if k == 0 else None)
        for lab, (p, c) in preds.items():
            ax.plot(*(np.vstack([h[row, -1:, :2], p[row]]) - o).T, "--", color=c, lw=1.4 if k == 0 else 0.9,
                    label=lab if k == 0 else None)
        ax.text(*(h[row, -6, :2] - o), f"V{row + 1} ({role})", fontsize=8)
    pts = np.vstack([h[[ia, ib], -6:, :2].reshape(-1, 2), g[[ia, ib]].reshape(-1, 2)] +   # zoom on the horizon
                    [p[[ia, ib]].reshape(-1, 2) for p, _ in preds.values()]) - o              # (last 30 s observed)
    lo, hi = pts.min(0), pts.max(0); c0, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.1
    ax.set_xlim(c0[0] - half, c0[0] + half); ax.set_ylim(c0[1] - half, c0[1] + half)
    ax.set_aspect("equal", adjustable="box"); ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.set_title(f"{ds}: receiver V{ia + 1}, sender V{ib + 1} (thin: sender)", fontsize=9); ax.legend(fontsize=6.5, loc="best")
    tt = np.arange(1, g.shape[1] + 1) * 5.0
    ax2.plot(tt, ratios["ground truth"], "k:", lw=2, label="ground truth")
    for lab, (_, c) in preds.items():
        ax2.plot(tt, ratios[lab], "--", color=c, lw=1.4, label=lab)
    ax2.axhline(1.0, color="0.4", lw=0.8); ax2.text(tt[0], 1.02, "QSD boundary", fontsize=7, color="0.4", va="bottom")
    ax2.set_xlabel("prediction time (s)"); ax2.set_ylabel(f"$d / d^{{\\mathrm{{safe}}}}$ (receiver V{ia + 1})")
    ax2.set_title("separation relative to the receiver's QSD", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, "ablation_vis.pdf"), bbox_inches="tight"); plt.close(fig)
    np.savetxt("results/plot_data/ablation_vis.csv", np.column_stack([tt] + [ratios[k] for k in ratios]), delimiter=",",
               header="t_s," + ",".join(k.replace(",", ";") for k in ratios), comments="")
    return dict(case, receiver_V=ia + 1, sender_V=ib + 1, M=len(idx),
                sog_kn=[round(float(np.linalg.norm(h[r_, -1, 2:]) / KN), 1) for r_ in (ia, ib)],
                L_B_recv=[float(L[ia]), float(B[ia])],
                min_ratio={k: round(float(v.min()), 3) for k, v in ratios.items()},
                ade_recv={k: round(float(np.linalg.norm(p[ia] - g[ia], axis=-1).mean()), 2) for k, (p, _) in preds.items()},
                ade_send={k: round(float(np.linalg.norm(p[ib] - g[ib], axis=-1).mean()), 2) for k, (p, _) in preds.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_dma")
    ap.add_argument("--out", default="results/figures_res")
    ap.add_argument("--only", choices=["traj", "ablation"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs("results/plot_data", exist_ok=True)
    fn = "results/plot_data/result_figs_cases.json"
    info = json.load(open(fn)) if os.path.exists(fn) else {}
    if a.only != "ablation":
        info["traj_compare"] = fig_traj_compare(a.runs, a.out)
    if a.only != "traj":
        info["ablation"] = fig_ablation(a.runs, a.out)
    json.dump(info, open(fn, "w"), indent=1)
    print(json.dumps(info, indent=1))
