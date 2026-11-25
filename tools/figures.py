#!/usr/bin/env python3
"""Figures of the experiments section, from the saved test predictions and the trained RIGS-Net (seed 0).

Case selection never looks at any model's prediction:
  * traj_compare / ablation_vis: per region, the test trajectory whose ground-truth future has the largest
    absolute course change between the last observed velocity and the final future step, among vessels
    with SOG >= 10 kn at the last observed frame (ties: lowest index).
  * proximity_risk_graph / risk_score_evolution: the test scene (any region) with the largest number of
    vessels whose last observed positions lie within 3 km of another vessel of the scene; within it the
    receiver is the vessel with most such neighbours.
The values plotted are exported to results/plot_data/*.csv.

    python tools/figures.py --runs runs --out ../figures_new
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rigs import qsd  # noqa: E402
from rigs.config import DATA_ROOT, DATASETS, MODEL_LABELS, RConfig  # noqa: E402

KN = 0.514444
METHODS = ["sstgcnn", "stmgcn", "gstgan", "mrgtraj", "aisformer", "mstformer", "dgcntf"]
COLORS = {"sstgcnn": "#8c8c8c", "stmgcn": "#b07aa1", "gstgan": "#9c755f", "mrgtraj": "#76b7b2",
          "aisformer": "#edc948", "mstformer": "#59a14f", "dgcntf": "#f28e2b", "rigs": "#e15759"}


def load(run):
    p = os.path.join(run, "test_dump.npz")
    return np.load(p) if os.path.exists(p) else None


def select_case(dump, q=99):
    """Among vessels with SOG >= 8 kn at the last observed and at the final future step, the trajectory whose
    course change is closest to the q-th percentile (a typical manoeuvre, not an interpolation artefact)."""
    h, g = dump["hist"], dump["gt"]
    v0 = h[:, -1, 2:]
    vf = (g[:, -1] - g[:, -2]) / 5.0
    a0, af = np.arctan2(v0[:, 0], v0[:, 1]), np.arctan2(vf[:, 0], vf[:, 1])
    turn = np.degrees(np.abs(np.angle(np.exp(1j * (af - a0)))))
    ok = (np.linalg.norm(v0, axis=1) >= 8 * KN) & (np.linalg.norm(vf, axis=1) >= 8 * KN)
    target = np.percentile(turn[ok], q)
    idx = np.flatnonzero(ok)
    n = int(idx[np.argmin(np.abs(turn[idx] - target))])
    return n, float(turn[n])


def panel(ax, n, dump, preds, title):
    h, g = dump["hist"][n, :, :2], dump["gt"][n]
    o = h[-1]
    ax.plot(*(h - o).T, "k-", lw=2, label="Observed")
    ax.plot(*(np.vstack([h[-1:], g]) - o).T, "k--", lw=2, label="Ground truth")
    for lab, (p, c) in preds.items():
        ax.plot(*(np.vstack([h[-1:], p]) - o).T, "-", color=c, lw=1.4, marker="o", ms=2, label=lab)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("East (m)", fontsize=8); ax.set_ylabel("North (m)", fontsize=8)
    ax.tick_params(labelsize=7)


def fig_traj_compare(runs, out, csv):
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8))
    rows = []
    for ax, ds, lt in zip(axes, DATASETS, "abcde"):
        ref = load(os.path.join(runs, ds, "cv", "seed0"))
        n, deg = select_case(ref)
        lam = json.load(open(os.path.join(os.path.dirname(csv), "..", "lambda_selected.json")))[ds]
        preds = {}
        for m in METHODS + ["rigs"]:
            name = f"rigs-lam{lam:g}" if m == "rigs" else m
            d = load(os.path.join(runs, ds, name, "seed0"))
            if d is None:
                continue
            assert np.allclose(d["gt"][n], ref["gt"][n]), "dump rows are not aligned"
            preds[MODEL_LABELS[m]] = (d["pred"][n, 0], COLORS[m])
            for t, p in enumerate(d["pred"][n, 0]):
                rows.append((ds, m, t + 1, *p))
        panel(ax, n, ref, preds, f"({lt}) {ds} (course change {deg:.0f}°)")
        for t, p in enumerate(ref["gt"][n]):
            rows.append((ds, "ground_truth", t + 1, *p))
    axes[0].legend(fontsize=6.5, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "traj_compare.pdf"), bbox_inches="tight")
    np.savetxt(csv.format("traj_compare"), np.array(rows, dtype=object), fmt="%s", delimiter=",",
               header="dataset,model,step,x_m,y_m", comments="")


def fig_ablation(runs, out, csv, ds="ZSI"):
    lam = json.load(open(os.path.join(os.path.dirname(csv), "..", "lambda_selected.json")))[ds]
    ref = load(os.path.join(runs, ds, "cv", "seed0"))
    n, deg = select_case(ref)
    var = [("Baseline", "rigs-base", "#4e79a7"), ("Baseline + Risk", "rigs-risk", "#f28e2b"),
           ("Baseline + Risk + QSD", "rigs-lam0", "#59a14f"), ("RIGS-Net (full)", f"rigs-lam{lam:g}", "#e15759")]
    preds = {}
    for lab, name, c in var:
        d = load(os.path.join(runs, ds, name, "seed0"))
        if d is not None:
            preds[lab] = (d["pred"][n, 0], c)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    panel(ax, n, ref, preds, f"{ds}: ablation variants (course change {deg:.0f}°)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ablation_vis.pdf"), bbox_inches="tight")


def pick_scene(radius=3000.0):
    from aistraj.data.dataset import AISScenes
    best = None
    for ds in DATASETS:
        sc = AISScenes(os.path.join(DATA_ROOT, ds), RConfig(), "test")
        for i in np.flatnonzero(sc.ends - sc.starts >= 3):
            it = sc[int(i)]
            p = it["hist"][:, -1, :2]
            d = np.linalg.norm(p[:, None] - p[None], axis=-1) + np.eye(len(p)) * 1e9
            close = (d < radius).sum(1)
            score = int((close > 0).sum())
            if best is None or score > best[0]:
                best = (score, ds, int(i), int(np.argmax(close)))
    return best


def fig_graph(runs, out, csv):
    from aistraj.data.dataset import AISScenes, collate_scenes
    from rigs.experiment import build_model
    from rigs import qsd
    score, ds, i, recv = pick_scene()
    lam = json.load(open(os.path.join(os.path.dirname(csv), "..", "lambda_selected.json")))[ds]
    run = os.path.join(runs, ds, f"rigs-lam{lam:g}", "seed0")
    cfg = RConfig.from_dict(json.load(open(os.path.join(run, "config.json"))))
    stats = json.load(open(os.path.join(DATA_ROOT, ds, "stats.json")))
    net = build_model("rigs", cfg, stats)
    net.load_state_dict(torch.load(os.path.join(run, "best.pt"), map_location="cpu")["state"])
    net.eval()
    sc = AISScenes(os.path.join(DATA_ROOT, ds), cfg, "test")
    it = sc[i]
    b = collate_scenes([it])
    with torch.no_grad():
        A, rho_bar, rho = net.graph(b["hist"], b["mask"], *qsd.vessel_dims(net, b))
    A, rho_bar, rho = A[0].numpy(), rho_bar[0].numpy(), rho[0].numpy()
    pos = it["hist"][:, :, :2]; vel = it["hist"][:, -1, 2:]
    M = len(pos)
    p = pos[:, -1]
    d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    prox = np.where(np.eye(M, dtype=bool), 0, 1.0 / np.maximum(d, 1.0))
    prox = prox / prox.sum(1, keepdims=True)
    c = p[recv]
    near = [j for j in range(M) if d[recv, j] <= 5000.0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    for j in near:
        ax.plot(*(pos[j] - c).T, "-", color="0.6", lw=1)
        u = vel[j] / max(np.linalg.norm(vel[j]), 1e-6)
        ax.arrow(*(p[j] - c), *(u * 250), head_width=90, color="k")
        ax.text(*(p[j] - c + 120), f"V{j + 1}", fontsize=8)
        if j != recv:
            ax.annotate("", xy=(0, 0), xytext=p[j] - c,
                        arrowprops=dict(arrowstyle="->", color="#e15759", lw=0.5 + 8 * A[-1, recv, j]))
            ax.text(*((p[j] - c) / 2), f"A={A[-1, recv, j]:.2f}\nprox={prox[recv, j]:.2f}", fontsize=7, color="#e15759")
    ax.set_title(f"(a) {ds}: receiver V{recv + 1} and vessels within 5 km", fontsize=9)
    ax.set_aspect("equal", adjustable="datalim"); ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax = axes[1]
    others = [j for j in range(M) if j != recv]
    ax.scatter(d[recv, others] / 1000, prox[recv, others], marker="o", label="proximity weight (inverse distance)")
    ax.scatter(d[recv, others] / 1000, A[-1, recv, others], marker="^", label=r"learned $A^{T_o}_{ij}$")
    for j in others:
        ax.text(d[recv, j] / 1000 * 1.05, A[-1, recv, j], f"V{j + 1}", fontsize=6)
    ax.set_xscale("log"); ax.set_xlabel("Distance to receiver (km)"); ax.set_ylabel("Normalized edge weight")
    ax.set_title(f"(b) all {len(others)} senders of the scene", fontsize=9); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "proximity_risk_graph.pdf"), bbox_inches="tight")
    # gate evolution for the receiver and its most heavily weighted sender
    j = int(np.argmax(A[-1, recv]))
    t = np.arange(1, A.shape[0] + 1) * cfg.dt
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(t, rho_bar[:, recv, j], "o-", ms=3, label=r"analytical QSD prior $\bar\rho_{ij}$")
    ax.plot(t, rho[:, recv, j], "s-", ms=3, label=r"learned interaction gate $\rho_{ij}$")
    ax.plot(t, A[:, recv, j], "^-", ms=3, label=r"edge weight $A_{ij}$")
    ax.set_xlabel("Observed time (s)"); ax.set_ylabel("Value"); ax.legend(fontsize=7)
    ax.set_title(f"{ds}: receiver V{recv + 1}, sender V{j + 1}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "risk_score_evolution.pdf"), bbox_inches="tight")
    np.savetxt(csv.format("risk_score_evolution"),
               np.c_[t, rho_bar[:, recv, j], rho[:, recv, j], A[:, recv, j]], delimiter=",",
               header="t_s,rho_bar,rho,A", comments="")
    np.savetxt(csv.format("graph_weights"), np.c_[np.arange(M) + 1, d[recv], prox[recv], A[-1, recv]],
               delimiter=",", header="vessel,dist_to_receiver_m,proximity_w,A", comments="")
    return dict(dataset=ds, scene=i, receiver=recv + 1, sender=j + 1, n_close=score,
                dist_m=d[recv].round(0).tolist(), A=A[-1, recv].round(3).tolist(), prox=prox[recv].round(3).tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_dma")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs("results/plot_data", exist_ok=True)
    csv = "results/plot_data/{}.csv"
    fig_traj_compare(a.runs, a.out, csv)
    fig_ablation(a.runs, a.out, csv)
    info = fig_graph(a.runs, a.out, csv)
    json.dump(info, open("results/plot_data/graph_case.json", "w"), indent=1)
    print(json.dumps(info, indent=1))
