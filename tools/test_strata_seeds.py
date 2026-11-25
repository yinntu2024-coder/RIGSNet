#!/usr/bin/env python3
"""tab:test_strata over all seeds: per test receiver, ADE averaged over the seeds of a run (val-selected rigs-lam*)
and of the reference (rigs-noint); receivers stratified by GT r_min (tools/val_strata.receiver_rmin); cell = mean of
(run - ref) with a paired scene-bootstrap 95% CI. Writes results/test_strata/<R>.json in the format read by
tools/test_strata_tex.py.      python tools/test_strata_seeds.py [--seeds 0,1,2]
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from val_strata import receiver_rmin, BINS, LABELS  # noqa: E402
from rigs.config import DATA_ROOT, DATASETS  # noqa: E402
from rigs.dims import load_seg_dims  # noqa: E402


def seed_avg(runs, ds, name, seeds, key):
    ade, fde = [], []
    for s in seeds:
        z = np.load(os.path.join(runs, ds, name, f"seed{s}", "test_dump.npz"))
        assert np.array_equal(np.stack([z["scene"], z["seg"]], 1), key), f"agent order differs: {ds}/{name}/seed{s}"
        e = np.linalg.norm(z["pred"][:, 0] - z["gt"], axis=-1)
        ade.append(e.mean(-1)); fde.append(e[:, -1])
    return np.mean(ade, 0), np.mean(fde, 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_dma")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--nboot", type=int, default=1000)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    sel = json.load(open("results/lambda_selected.json"))
    os.makedirs("results/test_strata", exist_ok=True)
    for ds in DATASETS:
        z0 = np.load(os.path.join(a.runs, ds, "rigs-noint", f"seed{seeds[0]}", "test_dump.npz"))
        key = np.stack([z0["scene"], z0["seg"]], 1)
        rmin = receiver_rmin(z0, load_seg_dims(os.path.join(DATA_ROOT, ds)))
        single = ~np.isfinite(rmin)
        b = np.where(single, len(BINS) - 1, np.digitize(np.where(single, 0, rmin), BINS[1:-1]))
        run = f"rigs-lam{sel[ds]:g}"
        A, F = {}, {}
        for nm in ("rigs-noint", run):
            A[nm], F[nm] = seed_avg(a.runs, ds, nm, seeds, key)
        scene = z0["scene"]; rng = np.random.default_rng(0); out = {}
        for k, lab in enumerate(LABELS + ["ALL"]):
            s = np.ones(len(b), bool) if lab == "ALL" else b == k
            if not s.any():
                continue
            us, inv = np.unique(scene[s], return_inverse=True)
            diff = A[run][s] - A["rigs-noint"][s]
            ssum = np.bincount(inv, diff, len(us)); scnt = np.bincount(inv, None, len(us))
            bs = [ssum[i].sum() / scnt[i].sum() for i in (rng.integers(0, len(us), len(us)) for _ in range(a.nboot))]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            out[lab] = dict(n=int(s.sum()), seeds=seeds, ade={n: float(A[n][s].mean()) for n in A},
                            fde={n: float(F[n][s].mean()) for n in F}, **{f"diff_{run}": [float(diff.mean()), float(lo), float(hi)]})
            print(f"{ds} {lab:16s} n={s.sum():6d} d={diff.mean():+.3f} [{lo:+.3f},{hi:+.3f}]", flush=True)
        json.dump(out, open(f"results/test_strata/{ds}.json", "w"), indent=1)
