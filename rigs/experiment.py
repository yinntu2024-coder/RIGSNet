"""Model registry, training loop and one-experiment driver.

Model selection: every epoch is scored on the validation split. `best.pt` keeps the epoch with the
lowest validation ADE under the model's native protocol (best-of-20 for stochastic baselines, K = 1 for
deterministic models). For stochastic baselines `best_single.pt` additionally keeps the epoch with the
lowest validation single-sample ADE, so that the K = 1 comparison does not use a checkpoint chosen for
best-of-20. Early stopping when neither criterion has improved for `patience` epochs.
The test split is touched only once, after training.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from aistraj.data.dataset import make_loaders
from aistraj.engine import evaluate, set_seed, to_device
from aistraj.metrics import full_report, summarize

from .dims import attach_dims
from .config import BASELINE_OPT, DATA_ROOT, RIGS_OPT, STOCHASTIC, VARIANTS, RConfig


def model_config(model: str, variant: str = "full", seed: int = 0, overrides=None) -> RConfig:
    cfg = RConfig(seed=seed)
    if model == "rigs":
        cfg = cfg.override(**RIGS_OPT).override(**VARIANTS[variant])
    elif model == "cv":
        cfg = cfg.override(epochs=0)
    else:
        cfg = cfg.override(**BASELINE_OPT[model])
    if model not in STOCHASTIC:
        cfg = cfg.override(K=1)
    return cfg.apply_cli(overrides)


def build_model(model: str, cfg, stats):
    if model == "rigs":
        from .model import RIGSNet
        return RIGSNet(cfg, stats)
    if model == "sstgcnn":
        from .official import SocialSTGCNNOfficial
        return SocialSTGCNNOfficial(cfg, stats)
    if model in ("stmgcn", "gstgan", "dgcntf", "aisformer"):
        if model != "dgcntf" and getattr(cfg, "out_mode", "direct") != "direct":
            raise ValueError(f"{model} (rigs/paperimpl.py) does not implement out_mode={cfg.out_mode}")
        from . import paperimpl as pi
        return {"stmgcn": pi.STMGCN, "gstgan": pi.GSTGAN, "dgcntf": pi.DGCNTransformer,
                "aisformer": pi.AISFormer}[model](cfg, stats)
    if model == "mstformer":
        from .mstformer_official import MSTFormerOfficial
        return MSTFormerOfficial(cfg, stats)
    if model == "mrgtraj":
        from .official import MRGTrajOfficial
        return MRGTrajOfficial(cfg, stats)
    from . import baselines as nb
    new = {"cv": nb.ConstantVelocity, "aisformer": nb.AISFormer, "mstformer": nb.MSTFormer, "dgcntf": nb.DGCNTransformer}
    if model in new:
        return new[model](cfg, stats)
    raise ValueError(f"unknown model: {model}")


def run_name(model, variant="full", tag=""):
    n = model if variant == "full" else f"{model}-{variant}"
    return f"{n}-{tag}" if tag else n


def run_dir(runs, dataset, model, variant="full", tag="", seed=0):
    return os.path.join(runs, dataset, run_name(model, variant, tag), f"seed{seed}")


def fit(model, loaders, cfg, out, device, log=print):
    opts = model.configure_optimizers()
    scheds = [torch.optim.lr_scheduler.StepLR(o, step_size=cfg.lr_step, gamma=cfg.lr_gamma)
              for o in opts.values()] if cfg.lr_step > 0 else []
    stoch = getattr(model, "multimodal", False) and cfg.select_single
    best, best1, bestf, bad, hist = float("inf"), float("inf"), float("inf"), 0, []
    n_it = max(len(loaders["train"]), 1)
    for epoch in range(cfg.epochs):
        model.train()
        t0, agg = time.time(), {}
        for it, batch in enumerate(loaders["train"]):
            logs = model.training_step(to_device(batch, device), opts, epoch + it / n_it)
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + float(v)
        agg = {k: v / n_it for k, v in agg.items()}
        for sc in scheds:
            sc.step()
        dist, end, _ = evaluate(model, loaders["val"], cfg.K, device, seed=cfg.seed)
        val = summarize(dist, end)
        if cfg.select_metric == "loss":
            val["loss"] = val_loss(model, loaders["val"], device)
        eligible = epoch + 1 >= cfg.select_start
        if cfg.select_metric == "loss":
            imp = eligible and val["loss"] < best - 1e-9
        elif cfg.select_metric == "ade_or_fde":
            imp = eligible and (val["ADE"] <= best or val["FDE"] <= bestf)
        else:
            imp = eligible and val["ADE"] < best - 1e-6
        imp1 = stoch and eligible and val["meanADE"] < best1 - 1e-6
        if imp:
            best = val["loss"] if cfg.select_metric == "loss" else min(best, val["ADE"])
            bestf = min(bestf, val["FDE"])
            torch.save(dict(state=model.state_dict(), epoch=epoch + 1, val=val), os.path.join(out, "best.pt"))
        if imp1:
            best1 = val["meanADE"]
            torch.save(dict(state=model.state_dict(), epoch=epoch + 1, val=val), os.path.join(out, "best_single.pt"))
        bad = 0 if (imp or imp1) else (bad + 1 if eligible else 0)   # patience counts from select_start on
        hist.append(dict(epoch=epoch + 1, time=time.time() - t0, **{f"train_{k}": v for k, v in agg.items()},
                         val_ADE=val["ADE"], val_FDE=val["FDE"], val_meanADE=val["meanADE"], val_meanFDE=val["meanFDE"],
                         selected=bool(imp), selected_single=bool(imp1)))
        json.dump(hist, open(os.path.join(out, "history.json"), "w"), indent=2)
        log(f"epoch {epoch + 1:3d} | " + " ".join(f"{k} {v:.4f}" for k, v in agg.items())
            + f" | val ADE {val['ADE']:.3f} FDE {val['FDE']:.3f} single {val['meanADE']:.3f}"
            + (" *" if imp else "") + (" s" if imp1 else "") + f" | {time.time() - t0:.0f}s")
        if cfg.patience > 0 and bad >= cfg.patience:
            log(f"early stop at epoch {epoch + 1}")
            break
    return hist


def mst_stats(ds, root, n=3000):
    """Training-split mean/std of the [d_sog (kn), d_cog (deg)] inputs (official StandardScaler) and the
    dataset time origin for the time features."""
    from .mstformer_official import MSTFormerOfficial
    rng = np.random.default_rng(0)
    d = [MSTFormerOfficial.deltas(torch.from_numpy(ds[int(i)]["hist"][..., 2:])).reshape(-1, 2)
         for i in rng.permutation(len(ds))[:n]]
    d = torch.cat(d)
    meta = json.load(open(os.path.join(root, "meta.json")))
    return dict(mst_mu=d.mean(0).tolist(), mst_sd=d.std(0).clamp_min(1e-6).tolist(), t0=float(meta["t0"]))


@torch.no_grad()
def val_loss(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        l, _ = model.loss(to_device(batch, device), 0.0)
        tot += float(l) * len(batch["mask"]); n += len(batch["mask"])
    return tot / max(n, 1)


def bucket_loader(ds, cfg):
    """Training batches of scenes with equal vessel count (shuffled every epoch)."""
    from aistraj.data.dataset import collate_scenes
    sizes = ds.ends - ds.starts
    if cfg.max_vessels:
        sizes = np.minimum(sizes, cfg.max_vessels)
    g = torch.Generator().manual_seed(cfg.seed)

    class Sampler(torch.utils.data.Sampler):
        def __iter__(self):
            rng = np.random.default_rng(int(torch.randint(0, 2 ** 31, (1,), generator=g)))
            batches = []
            for s in np.unique(sizes):
                idx = rng.permutation(np.flatnonzero(sizes == s))
                batches += [idx[i:i + cfg.batch_size].tolist() for i in range(0, len(idx), cfg.batch_size)]
            for i in rng.permutation(len(batches)):
                yield batches[i]

        def __len__(self):
            return int(sum(-(-np.sum(sizes == s) // cfg.batch_size) for s in np.unique(sizes)))

    return torch.utils.data.DataLoader(ds, batch_sampler=Sampler(), collate_fn=collate_scenes)


def score(model, loader, cfg, out, device, ckpt, tag, dump=True):
    p = os.path.join(out, ckpt)
    if os.path.exists(p):
        model.load_state_dict(torch.load(p, map_location=device)["state"])
    dist, end, arrays = evaluate(model, loader, cfg.K, device, seed=cfg.seed, dump=dump)
    rep = full_report(dist, end)
    json.dump(rep, open(os.path.join(out, f"{tag}_metrics.json"), "w"), indent=2)
    if dump:
        np.savez_compressed(os.path.join(out, f"{tag}_dump.npz"), dist=dist.astype(np.float32), **arrays)
    if hasattr(model, "sample_mean"):                                          # stochastic: K = 1 = predictive mean
        score_mean(model, loader, cfg, out, device, tag, dump)
    return rep


def score_mean(model, loader, cfg, out, device, tag, dump=True):
    """K = 1 score of a stochastic model from its predictive mean (`sample_mean`); writes {tag}_mean_*."""
    model.sample = lambda b, K: model.sample_mean(b)
    try:
        dist, end, arrays = evaluate(model, loader, 1, device, seed=cfg.seed, dump=dump)
    finally:
        del model.sample
    rep = full_report(dist, end)
    json.dump(rep, open(os.path.join(out, f"{tag}_mean_metrics.json"), "w"), indent=2)
    if dump:
        np.savez_compressed(os.path.join(out, f"{tag}_mean_dump.npz"), dist=dist.astype(np.float32), **arrays)
    return rep


def run_experiment(dataset, model="rigs", variant="full", tag="", seed=0, runs="runs", overrides=None,
                   eval_split="test", device=None, log=print, data_root=DATA_ROOT, force=False):
    out = run_dir(runs, dataset, model, variant, tag, seed)
    done = os.path.join(out, f"{eval_split}_metrics.json")
    if os.path.exists(done) and not force:
        log(f"[skip] {out}")
        return json.load(open(done))
    os.makedirs(out, exist_ok=True)
    cfg = model_config(model, variant, seed, overrides)
    cfg.dump(os.path.join(out, "config.json"))
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(seed)
    root = os.path.join(data_root, dataset)
    loaders, sets, stats = make_loaders(root, cfg, stats_path=os.path.join(root, "stats.json"))
    if cfg.bucket:
        loaders["train"] = bucket_loader(sets["train"], cfg)
    if model == "mstformer":
        stats = dict(stats, **mst_stats(sets["train"], root))
    net = build_model(model, cfg, stats)
    attach_dims(net, root)                                                     # per-vessel L, B (if available)
    net = net.to(device)
    log(f"[run] {out} | scenes {len(sets['train'])}/{len(sets['val'])}/{len(sets['test'])} | "
        f"params {sum(p.numel() for p in net.parameters()):,}")
    if cfg.epochs > 0:
        fit(net, loaders, cfg, out, device, log)
    if getattr(net, "multimodal", False) and cfg.select_single:
        score(net, loaders[eval_split], cfg, out, device, "best_single.pt", f"{eval_split}_single",
              dump=(eval_split == "test"))
    rep = score(net, loaders[eval_split], cfg, out, device, "best.pt", eval_split, dump=(eval_split == "test"))
    s = rep["summary"]
    log(f"[{eval_split}] {out} | N={s['N']} K={s['K']} ADE {s['ADE']:.3f} FDE {s['FDE']:.3f} "
        f"single {s['meanADE']:.3f}/{s['meanFDE']:.3f}")
    return rep
