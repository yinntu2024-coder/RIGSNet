"""Re-score the K = 1 (predictive-mean) test metrics of finished stochastic runs from their best.pt,
without retraining and without touching {split}_metrics.json (the best-of-K scores).

The previous {split}_mean_metrics.json / _mean_dump.npz are kept as *.<backup>.json / .npz.

  python tools/rescore_mean.py runs_dma/*/mrgtraj/seed0 --backup z0
"""
import argparse
import json
import os
import shutil
import sys
from dataclasses import fields

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch  # noqa: E402

from rigs.config import DATA_ROOT, RConfig  # noqa: E402  (puts aistraj on sys.path)
from aistraj.data.dataset import make_loaders  # noqa: E402
from aistraj.engine import set_seed  # noqa: E402
from rigs.dims import attach_dims  # noqa: E402
from rigs.experiment import build_model, score_mean  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="run dirs <runs>/<dataset>/<model>/seed<k>")
    ap.add_argument("--split", default="test")
    ap.add_argument("--backup", default="old", help="suffix for the replaced mean files")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    names = {f.name for f in fields(RConfig)}
    for d in a.dirs:
        d = d.rstrip("/")
        dataset, model = d.split("/")[-3], d.split("/")[-2].split("-")[0]
        if not os.path.exists(os.path.join(d, "best.pt")) or not os.path.exists(os.path.join(d, f"{a.split}_metrics.json")):
            print(f"[skip unfinished] {d}")
            continue
        raw = json.load(open(os.path.join(d, "config.json")))
        cfg = RConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items() if k in names})
        set_seed(cfg.seed)
        root = os.path.join(DATA_ROOT, dataset)
        loaders, sets, stats = make_loaders(root, cfg, stats_path=os.path.join(root, "stats.json"))
        net = build_model(model, cfg, stats)
        if not hasattr(net, "sample_mean"):
            print(f"[skip deterministic] {d}")
            continue
        attach_dims(net, root)
        net = net.to(a.device)
        net.load_state_dict(torch.load(os.path.join(d, "best.pt"), map_location=a.device)["state"])
        for ext in ("_mean_metrics.json", "_mean_dump.npz"):
            src = os.path.join(d, a.split + ext)
            if os.path.exists(src):
                stem, suf = os.path.splitext(src)
                shutil.move(src, f"{stem}.{a.backup}{suf}")
        dump = a.split == "test"
        rep = score_mean(net, loaders[a.split], cfg, d, torch.device(a.device), a.split, dump=dump)
        s = rep["summary"]
        print(f"[mean] {d} | N={s['N']} ADE {s['ADE']:.3f} FDE {s['FDE']:.3f}", flush=True)


if __name__ == "__main__":
    main()
