"""Evaluation metrics of Sec. 5.3 and Sec. 5.4.2. All distances are in metres.

Input convention: dist [N, K, T] = Euclidean error of sample k of test trajectory n at future
step t; end [N, K, 2] = predicted end points. N counts individual vessel trajectories (N_test).
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

MR_THRESHOLDS = (10.0, 20.0, 30.0, 50.0)            # Tables MR@10/20/30/50 m (10 and 20 m added: real errors are ~5-15 m)
HORIZON_THRESHOLDS = (10.0, 20.0, 30.0, 50.0, 100.0)    # Figs. 15-16 (the 50-400 m of the first draft are ~0 % on real data)
COVERAGE_THRESHOLD = 100.0                          # Coverage@100m
K_SWEEP = (1, 5, 10, 20)


def best_of_k(dist: np.ndarray):
    """Per-trajectory errors E_n: min_k ADE and min_k FDE (the minima are taken independently,
    exactly as in the two displayed equations for ADE_K and FDE_K)."""
    return dist.mean(-1).min(1), dist[..., -1].min(1)


def miss_rate(err: np.ndarray, gamma: float) -> float:
    """MR_{E@gamma} = 1 - mean(S(E, gamma)), S = 1 if E <= gamma. Returned in percent."""
    return float(100.0 * (1.0 - (err <= gamma).mean()))


def diversity(end: np.ndarray) -> np.ndarray:
    """Average pairwise end-point distance among the K samples of each trajectory, [N]."""
    K = end.shape[1]
    if K < 2:
        return np.zeros(end.shape[0])
    d = np.linalg.norm(end[:, :, None] - end[:, None, :], axis=-1)      # [N, K, K]
    return d.sum((1, 2)) / (K * (K - 1))


def summarize(dist: np.ndarray, end: np.ndarray, mr_thresholds: Sequence[float] = MR_THRESHOLDS) -> Dict[str, float]:
    ade, fde = best_of_k(dist)
    out = dict(N=int(dist.shape[0]), K=int(dist.shape[1]), ADE=float(ade.mean()), FDE=float(fde.mean()),
               meanADE=float(dist.mean(-1).mean()), meanFDE=float(dist[..., -1].mean()),
               coverage100=float(100.0 * (dist[..., -1].min(1) < COVERAGE_THRESHOLD).mean()),
               diversity=float(diversity(end).mean()))
    for g in mr_thresholds:
        out[f"MR_ADE@{int(g)}"] = miss_rate(ade, g)
        out[f"MR_FDE@{int(g)}"] = miss_rate(fde, g)
        # single-sample miss rate: share of (trajectory, sample) pairs whose own ADE / FDE exceeds g
        out[f"MR1_ADE@{int(g)}"] = miss_rate(dist.mean(-1).reshape(-1), g)
        out[f"MR1_FDE@{int(g)}"] = miss_rate(dist[..., -1].reshape(-1), g)
    return out


def k_sweep(dist: np.ndarray, end: np.ndarray, ks: Sequence[int] = K_SWEEP) -> Dict[str, Dict[str, float]]:
    """ADE_K / FDE_K for K in {1, 5, 10, 20}. Samples are i.i.d., so the first K' of the K drawn
    samples are an unbiased draw of K' samples."""
    return {str(k): summarize(dist[:, :k], end[:, :k]) for k in ks if k <= dist.shape[1]}


def horizon_curves(dist: np.ndarray, thresholds: Sequence[float] = HORIZON_THRESHOLDS) -> Dict[str, list]:
    """Metrics as a function of the forecasting horizon h = 1..T_f (Figs. 14-16).
    ADE(h) uses steps 1..h, FDE(h) uses step h."""
    T = dist.shape[-1]
    cum = np.cumsum(dist, axis=-1) / np.arange(1, T + 1)                # running ADE per sample, [N, K, T]
    ade_best, fde_best = cum.min(1), dist.min(1)                        # [N, T]
    out = dict(horizon=list(range(1, T + 1)),
               ADE_best=ade_best.mean(0).tolist(), FDE_best=fde_best.mean(0).tolist(),
               ADE_mean=cum.mean((0, 1)).tolist(), FDE_mean=dist.mean((0, 1)).tolist())
    for g in thresholds:
        out[f"MR_ADE@{int(g)}"] = (100.0 * (1.0 - (ade_best <= g).mean(0))).tolist()
        out[f"MR_FDE@{int(g)}"] = (100.0 * (1.0 - (fde_best <= g).mean(0))).tolist()
    return out


def full_report(dist: np.ndarray, end: np.ndarray) -> dict:
    return dict(summary=summarize(dist, end), k_sweep=k_sweep(dist, end), horizon=horizon_curves(dist))
