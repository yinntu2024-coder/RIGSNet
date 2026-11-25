#!/usr/bin/env python3
"""Mechanism-analysis figures of the manuscript, on the DMA regions (RIGS-Net, seed 0):

  proximity_risk_graph.pdf / risk_score_evolution.pdf  (Directed Interaction Weights)
  safe_reg.pdf                                          (Effect of QSD Regularization)

Case selection never looks at any model prediction (ground truth / observations only):
  * graph case: the test scene (any region) with the largest number of vessels whose last observed position is
    within 3 km of another vessel of the scene; the receiver is the vessel with most such neighbours
    (same rule as tools/figures.py).
  * QSD case (rule fixed 2026-11-20, before any case was drawn): RWA test scenes; ordered vessel pairs
    (receiver i, sender j), both with SOG >= 5 kn at the last observed frame, whose ground-truth future enters
    the receiver's QSD (r = min_t d/d_safe < 1); the pair with the MEDIAN r (lower median, ties by scene/i/j).
    It is compared between the lambda = 0 and the lambda = 0.2 run (as tab:qsd_strata).
    (Earlier rule, superseded: crossing pair closest to the boundary without entering it, validation-selected
    lambda.)
A figure is skipped (with a message) if the needed runs are not finished. Plotted values -> results/plot_data.

    python tools/mechanism_figs.py [--pick-only] --out results/figures_mech
"""
import argparse
import importlib.util
import json
import os
import sys
from dataclasses import fields

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rigs.config import DATA_ROOT, DATASETS, LAMBDAS, RConfig  # noqa: E402  (puts aistraj on sys.path)
from aistraj.data.dataset import AISScenes, collate_scenes  # noqa: E402
from rigs import qsd  # noqa: E402
from rigs.dims import attach_dims, load_seg_dims  # noqa: E402
from rigs.experiment import build_model  # noqa: E402

KN = 0.514444
_spec = importlib.util.spec_from_file_location("make_tables", os.path.join(os.path.dirname(__file__), "make_tables.py"))
make_tables = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(make_tables)


def scenes(ds):
    return AISScenes(os.path.join(DATA_ROOT, ds), RConfig(), "test")


def pick_graph_case(radius=3000.0):
    best = None
    for ds in DATASETS:
        sc = scenes(ds)
        for i in np.flatnonzero(sc.ends - sc.starts >= 3):
            p = sc[int(i)]["hist"][:, -1, :2]
            d = np.linalg.norm(p[:, None] - p[None], axis=-1) + np.eye(len(p)) * 1e9
            close = (d < radius).sum(1)
            score = int((close > 0).sum())
            if best is None or score > best[0]:
                best = (score, ds, int(i), int(np.argmax(close)))
    return dict(n_close=best[0], dataset=best[1], scene=best[2], receiver=best[3])


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def gt_ratio(fut, hist, L, B):
    """min over future steps of d / d_safe for every ordered pair (receiver row). fut [M, T, 2]."""
    P = torch.from_numpy(fut[None]).double()
    fwd = qsd.future_forward(P, torch.from_numpy(hist[None, :, -1, :2]).double(),
                             torch.from_numpy(unit(hist[None, :, -1, 2:])).double())
    rel = P[:, None] - P[:, :, None]                                          # [1, M(i), M(j), T, 2] = p_j - p_i
    Lt = torch.as_tensor(L, dtype=torch.float64)[None, :, None, None]
    Bt = torch.as_tensor(B, dtype=torch.float64)[None, :, None, None]
    d_safe, d = qsd.qsd_radius(rel, fwd[:, :, None].expand_as(rel), Lt, Bt, eps=0.0)
    r = (d / d_safe)[0].numpy()                                               # [M, M, T]
    return r


QSD_DS, QSD_LAM = "RWA", 0.2


def pick_qsd_case():
    cand = []
    ds = QSD_DS
    sc = scenes(ds)
    segL, segB = load_seg_dims(os.path.join(DATA_ROOT, ds))
    for i in np.flatnonzero(sc.ends - sc.starts >= 2):
        it = sc[int(i)]
        h, f = it["hist"], it["fut"][..., :2]
        sog = np.linalg.norm(h[:, -1, 2:], axis=1)
        r = gt_ratio(f, h, segL[it["seg"]], segB[it["seg"]]).min(-1)
        M = len(h)
        for a in range(M):
            for b in range(M):
                if a != b and sog[a] >= 5 * KN and sog[b] >= 5 * KN and r[a, b] < 1.0:
                    cand.append((float(r[a, b]), int(i), a, b))
    cand.sort()
    m = cand[(len(cand) - 1) // 2]
    return dict(gt_min_ratio=m[0], dataset=ds, scene=m[1], receiver=m[2], sender=m[3], n_candidates=len(cand))


def pick_qsd_case_old():
    best = None
    for ds in DATASETS:
        sc = scenes(ds)
        segL, segB = load_seg_dims(os.path.join(DATA_ROOT, ds))
        for i in np.flatnonzero(sc.ends - sc.starts >= 3):
            it = sc[int(i)]
            h, f = it["hist"], it["fut"][..., :2]
            v = h[:, -1, 2:]
            sog = np.linalg.norm(v, axis=1)
            cog = np.degrees(np.arctan2(v[:, 0], v[:, 1]))
            r = gt_ratio(f, h, segL[it["seg"]], segB[it["seg"]]).min(-1)
            M = len(h)
            for a in range(M):
                for b in range(M):
                    if a == b or sog[a] < 5 * KN or sog[b] < 5 * KN:
                        continue
                    dc = abs((cog[a] - cog[b] + 180) % 360 - 180)
                    if not 45 <= dc <= 135 or r[a, b] < 1.0:
                        continue
                    if best is None or r[a, b] < best[0]:
                        best = (float(r[a, b]), ds, int(i), a, b)
    return dict(gt_min_ratio=best[0], dataset=best[1], scene=best[2], receiver=best[3], sender=best[4])


def selected_lambda(runs, ds):
    done = [lam for lam in LAMBDAS if os.path.exists(os.path.join(runs, ds, f"rigs-lam{lam:g}", "seed0", "test_metrics.json"))]
    if len(done) < len(LAMBDAS):
        return None
    return make_tables.select_lambda(runs, ds)[0]


def load_rigs(runs, ds, lam):
    run = os.path.join(runs, ds, f"rigs-lam{lam:g}", "seed0")
    names = {f.name for f in fields(RConfig)}
    raw = json.load(open(os.path.join(run, "config.json")))
    cfg = RConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items() if k in names})
    root = os.path.join(DATA_ROOT, ds)
    net = build_model("rigs", cfg, json.load(open(os.path.join(root, "stats.json"))))
    attach_dims(net, root)
    net.load_state_dict(torch.load(os.path.join(run, "best.pt"), map_location="cpu")["state"])
    return net.eval(), cfg


def qsd_boundary(c, fwd, L, B, n=200):
    """Closed QSD boundary of a receiver at c with forward unit vector fwd (fore 3L, aft 2L, sides 1.5B)."""
    th = np.linspace(0, 2 * np.pi, n)
    ux, uy = np.sin(th), np.cos(th)                                           # local starboard / forward
    Rx = np.where(ux >= 0, 1.5 * B, 1.5 * B); Ry = np.where(uy >= 0, 3 * L, 2 * L)
    r = 1.0 / np.sqrt((ux / Rx) ** 2 + (uy / Ry) ** 2)
    st = np.array([fwd[1], -fwd[0]])
    return c + (r * ux)[:, None] * st + (r * uy)[:, None] * fwd


def fig_graph(runs, out, case):
    ds, i, recv = case["dataset"], case["scene"], case["receiver"]
    lam = selected_lambda(runs, ds)
    if lam is None:
        print(f"[skip graph figure] lambda sweep of {ds} not finished"); return None
    net, cfg = load_rigs(runs, ds, lam)
    it = AISScenes(os.path.join(DATA_ROOT, ds), cfg, "test")[i]
    b = collate_scenes([it])
    with torch.no_grad():
        A, rho_bar, rho = net.graph(b["hist"], b["mask"], *qsd.vessel_dims(net, b))
    A, rho_bar, rho = A[0].numpy(), rho_bar[0].numpy(), rho[0].numpy()
    pos, vel = it["hist"][:, :, :2], it["hist"][:, -1, 2:]
    M = len(pos); p = pos[:, -1]
    d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    prox = np.where(np.eye(M, dtype=bool), 0, 1.0 / np.maximum(d, 1.0)); prox /= prox.sum(1, keepdims=True)
    c = p[recv]
    near = [j for j in range(M) if d[recv, j] <= 5000.0]
    top3 = set(np.argsort(-A[-1, recv])[:3].tolist())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    for j in near:
        ax.plot(*(pos[j] - c).T, "-", color="0.6", lw=1)
        u = unit(vel[j])
        ax.arrow(*(p[j] - c), *(u * 250), head_width=90, color="k")
        ax.text(*(p[j] - c + 120), f"V{j + 1}", fontsize=8)
        if j != recv:
            ax.annotate("", xy=(0, 0), xytext=p[j] - c,
                        arrowprops=dict(arrowstyle="->", color="#e15759", lw=0.5 + 8 * A[-1, recv, j]))
            if j in top3:                                                       # numbers only for the 3 largest A (all in panel b)
                ax.text(*((p[j] - c) * 0.6), f"A={A[-1, recv, j]:.2f}\nprox={prox[recv, j]:.2f}", fontsize=7, color="#e15759")
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
    fig.tight_layout(); fig.savefig(os.path.join(out, "proximity_risk_graph.pdf"), bbox_inches="tight"); plt.close(fig)
    j = int(np.argmax(A[-1, recv]))
    t = np.arange(1, A.shape[0] + 1) * cfg.dt
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(t, rho_bar[:, recv, j], "o-", ms=3, label=r"analytical QSD prior $\bar\rho_{ij}$")
    ax.plot(t, rho[:, recv, j], "s-", ms=3, label=r"learned interaction gate $\rho_{ij}$")
    ax.plot(t, A[:, recv, j], "^-", ms=3, label=r"edge weight $A_{ij}$")
    ax.set_xlabel("Observed time (s)"); ax.set_ylabel("Value"); ax.legend(fontsize=7)
    ax.set_title(f"{ds}: receiver V{recv + 1}, sender V{j + 1}", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, "risk_score_evolution.pdf"), bbox_inches="tight"); plt.close(fig)
    np.savetxt("results/plot_data/risk_score_evolution.csv", np.c_[t, rho_bar[:, recv, j], rho[:, recv, j], A[:, recv, j]],
               delimiter=",", header="t_s,rho_bar,rho,A", comments="")
    rows = []
    for k in range(M):
        cosv = float(np.dot(unit(vel[k]) * np.linalg.norm(vel[k]) - vel[recv], unit(p[recv] - p[k]))) if k != recv else 0.0
        rows.append([k + 1, d[recv, k], prox[recv, k], A[-1, recv, k], np.linalg.norm(vel[k]) / KN, cosv])
    np.savetxt("results/plot_data/graph_weights.csv", np.array(rows), delimiter=",",
               header="vessel,dist_to_receiver_m,proximity_w,A,sog_kn,closing_speed_mps", comments="")
    return dict(case, lam=lam, sender_max_A=j + 1, M=M,
                senders=[dict(v=k + 1, dist_m=round(float(d[recv, k])), prox=round(float(prox[recv, k]), 3),
                              A=round(float(A[-1, recv, k]), 3), closing_mps=round(rows[k][5], 2))
                         for k in others])


def fig_safe(runs, out, case):
    ds, i, a, b_ = case["dataset"], case["scene"], case["receiver"], case["sender"]
    lam = QSD_LAM
    if not all(os.path.exists(os.path.join(runs, ds, f"rigs-lam{l:g}", "seed0", "best.pt")) for l in (0.0, lam)):
        print(f"[skip QSD figure] runs of {ds} not finished"); return None
    segL, segB = load_seg_dims(os.path.join(DATA_ROOT, ds))
    preds = {}
    for lab, l in (("without QSD reg. ($\\lambda=0$)", 0.0), (f"with QSD reg. ($\\lambda={lam:g}$)", lam)):
        net, cfg = load_rigs(runs, ds, l)
        it = AISScenes(os.path.join(DATA_ROOT, ds), cfg, "test")[i]
        with torch.no_grad():
            preds[lab] = net.predict(collate_scenes([it]), 1)[0, 0].numpy()            # [M, T, 2]
    h, g = it["hist"], it["fut"][..., :2]
    L, B = segL[it["seg"]], segB[it["seg"]]
    o = h[a, -1, :2]
    ratios = {"ground truth": gt_ratio(g, h, L, B)[a, b_]}
    for lab, p in preds.items():
        ratios[lab] = gt_ratio(p, h, L, B)[a, b_]
    tstar = int(np.argmin(ratios["ground truth"]))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.6), gridspec_kw=dict(width_ratios=[1.5, 1]))
    styles = {list(preds)[0]: ("#4e79a7", "--"), list(preds)[1]: ("#e15759", "--")}
    for k, (col, role) in enumerate(((a, "receiver"), (b_, "sender"))):
        ax.plot(*(h[col, :, :2] - o).T, "k-", lw=2, label="observed" if k == 0 else None)
        ax.plot(*(np.vstack([h[col, -1:, :2], g[col]]) - o).T, "k:", lw=2, label="ground truth" if k == 0 else None)
        for lab, p in preds.items():
            ax.plot(*(np.vstack([h[col, -1:, :2], p[col]]) - o).T, styles[lab][1], color=styles[lab][0], lw=1.6,
                    label=lab if k == 0 else None)
        ax.text(*(h[col, 0, :2] - o), f"V{col + 1} ({role})", fontsize=8)
    for j in range(len(h)):
        if j not in (a, b_):
            ax.plot(*(np.vstack([h[j, :, :2], g[j]]) - o).T, "-", color="0.75", lw=1)
    for lab, p, col in [("ground truth", g, "k")] + [(lab, p, styles[lab][0]) for lab, p in preds.items()]:
        prev = p[a, tstar - 1] if tstar > 0 else h[a, -1, :2]
        fwd = unit(p[a, tstar] - prev) if np.linalg.norm(p[a, tstar] - prev) > 1e-3 else unit(h[a, -1, 2:])
        bd = qsd_boundary(p[a, tstar], fwd, L[a], B[a])
        ax.plot(*(bd - o).T, "-", color=col, lw=0.8, alpha=0.7)
        ax.plot(*(p[b_, tstar] - o), "o", color=col, ms=4)
    pts = np.vstack([h[[a, b_], :, :2].reshape(-1, 2), g[[a, b_]].reshape(-1, 2)] +
                    [p[[a, b_]].reshape(-1, 2) for p in preds.values()]) - o
    lo, hi = pts.min(0), pts.max(0); c, half = (lo + hi) / 2, max((hi - lo).max() / 2, 3 * L[a]) * 1.15
    ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - half, c[1] + half)
    ax.set_aspect("equal", adjustable="box"); ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.set_title(f"{ds}: receiver V{a + 1}, sender V{b_ + 1}; QSD of V{a + 1} at t = {(tstar + 1) * 5} s", fontsize=9)
    ax.legend(fontsize=7, loc="best")
    tt = np.arange(1, g.shape[1] + 1) * 5.0
    ax2.plot(tt, ratios["ground truth"], "k:", lw=2, label="ground truth")
    for lab in preds:
        ax2.plot(tt, ratios[lab], "--", color=styles[lab][0], lw=1.6, label=lab)
    ax2.axhline(1.0, color="0.4", lw=0.8); ax2.text(tt[0], 1.02, "QSD boundary", fontsize=7, color="0.4", va="bottom")
    ax2.set_xlabel("prediction time (s)"); ax2.set_ylabel(f"$d / d^{{\\mathrm{{safe}}}}$ (receiver V{a + 1})")
    ax2.set_title("separation relative to the receiver's QSD", fontsize=9); ax2.legend(fontsize=7, loc="best")
    fig.tight_layout(); fig.savefig(os.path.join(out, "safe_reg.pdf"), bbox_inches="tight"); plt.close(fig)
    np.savetxt("results/plot_data/safe_reg_ratio.csv",
               np.column_stack([np.arange(1, g.shape[1] + 1) * 5.0] + [ratios[k] for k in ratios]), delimiter=",",
               header="t_s," + ",".join(k.replace(",", ";") for k in ratios), comments="")
    return dict(case, lam=lam, t_star_s=(tstar + 1) * 5, L_recv=float(L[a]), B_recv=float(B[a]),
                min_ratio={k: round(float(v.min()), 3) for k, v in ratios.items()},
                ade_pair={k: round(float(np.linalg.norm(p[[a, b_]] - g[[a, b_]], axis=-1).mean()), 2) for k, p in preds.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_dma")
    ap.add_argument("--out", default="results/figures_mech")
    ap.add_argument("--pick-only", action="store_true")
    ap.add_argument("--qsd-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs("results/plot_data", exist_ok=True)
    cases = dict(qsd=pick_qsd_case()) if a.qsd_only else dict(graph=pick_graph_case(), qsd=pick_qsd_case())
    print(json.dumps(cases, indent=1))
    if not a.pick_only:
        info = dict(qsd=fig_safe(a.runs, a.out, cases["qsd"]))
        if not a.qsd_only:
            info["graph"] = fig_graph(a.runs, a.out, cases["graph"])
        fn = "results/plot_data/mechanism_cases_qsd.json" if a.qsd_only else "results/plot_data/mechanism_cases.json"
        json.dump(info, open(fn, "w"), indent=1)
        print(json.dumps(info, indent=1))
