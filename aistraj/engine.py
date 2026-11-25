"""Model-agnostic training and evaluation loops (the same code scores RIGS-Net and every baseline)."""
from __future__ import annotations

import json
import os
import random
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .config import Config
from .metrics import full_report, summarize


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, K: int, device, seed: int = 0, dump: bool = False) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Returns dist [N, K, T_f], end [N, K, 2] over all valid test trajectories, plus optional raw
    arrays for the analysis tools. Sampling noise is seeded so that evaluation is repeatable."""
    model.eval()
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dist, end, extra = [], [], {k: [] for k in ("pred", "gt", "hist", "seg", "scene", "f_obs")}
    for batch in loader:
        batch = to_device(batch, device)
        pred = model.predict(batch, K)                                   # [B, K, M, T_f, 2] metres
        gt = batch["fut"][..., :2]
        m = batch["mask"] & batch["eval"] if "eval" in batch else batch["mask"]   # eval: test_subset selection
        p = pred.permute(0, 2, 1, 3, 4)[m]                               # [n, K, T_f, 2]
        g = gt[m]                                                        # [n, T_f, 2]
        dist.append((p - g[:, None]).norm(dim=-1).cpu().numpy())
        end.append(p[:, :, -1].cpu().numpy())
        if dump:
            extra["pred"].append(p.cpu().numpy().astype(np.float32))
            extra["gt"].append(g.cpu().numpy().astype(np.float32))
            extra["hist"].append(batch["hist"][m].cpu().numpy().astype(np.float32))
            extra["seg"].append(batch["seg"][m].cpu().numpy())
            extra["scene"].append(batch["scene"][:, None].expand_as(m)[m].cpu().numpy())
            extra["f_obs"].append(batch["f_obs"][:, None].expand_as(m)[m].cpu().numpy())
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    arrays = {k: np.concatenate(v) for k, v in extra.items()} if dump else {}
    return np.concatenate(dist), np.concatenate(end), arrays


EPOCH_MD_HEADER = (
    "| 时间 | 数据集 | 模型 | seed | epoch | val ADE20 | val FDE20 | val 单样本 ADE | val 单样本 FDE "
    "| test ADE20 | test FDE20 | test 单样本 ADE | test 单样本 FDE | val 最优 | 训练 s | 合计 s |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")


def _append_epoch_md(path: str, run_dir: str, epoch: int, val: dict, tst: Optional[dict], improved: bool,
                     t_train: float, t_all: float) -> None:
    """One row per epoch; several worker processes append to the same file under an exclusive lock."""
    import fcntl
    parts = os.path.normpath(run_dir).split(os.sep)
    ds, model, seed = parts[-3], parts[-2], parts[-1].replace("seed", "")
    f = lambda d, k: f"{d[k]:.3f}" if d is not None else "–"
    row = (f"| {time.strftime('%m-%d %H:%M')} | {ds} | {model} | {seed} | {epoch} | {f(val, 'ADE')} | {f(val, 'FDE')} "
           f"| {f(val, 'meanADE')} | {f(val, 'meanFDE')} | {f(tst, 'ADE')} | {f(tst, 'FDE')} | {f(tst, 'meanADE')} "
           f"| {f(tst, 'meanFDE')} | {'✓' if improved else ''} | {t_train:.0f} | {t_all:.0f} |\n")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        if fh.tell() == 0:
            fh.write("# 每轮评测记录（训练/验证步长 32，测试步长 1）\n\n"
                     "ADE20/FDE20 = 20 条样本取最优；单样本 = 20 条样本误差的平均。"
                     "“val 最优”打勾的 epoch 会被保存为 best.pt，最终测试结果取该 epoch；测试集只做监控，不参与选模型。\n\n"
                     + EPOCH_MD_HEADER)
        fh.write(row)
        fcntl.flock(fh, fcntl.LOCK_UN)


def fit(model, loaders, cfg: Config, run_dir: str, device, log=print) -> dict:
    """Adam, up to cfg.epochs epochs, model selection and early stopping on validation ADE_K."""
    os.makedirs(run_dir, exist_ok=True)
    opts = model.configure_optimizers()
    if cfg.lr_schedule == "plateau":
        scheds = [torch.optim.lr_scheduler.ReduceLROnPlateau(o, mode="min", factor=0.5, patience=5) for o in opts.values()]
    elif cfg.lr_schedule == "cosine":
        scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=cfg.epochs) for o in opts.values()]
    elif cfg.lr_schedule == "none":
        scheds = []
    else:
        raise ValueError(cfg.lr_schedule)
    best, bad, history = float("inf"), 0, []
    n_it = max(len(loaders["train"]), 1)
    ckpt = os.path.join(run_dir, "best.pt")
    t_start = time.time()
    # with denser training windows one epoch holds ~window_stride/train_window_stride times more scenes; schedules
    # driven by `progress` (KL annealing) are kept on the same number of seen scenes as a window_stride epoch
    pscale = cfg.window_stride / cfg.train_window_stride if cfg.train_window_stride > 0 else 1.0
    budget_hit = False
    for epoch in range(cfg.epochs):
        model.train()
        t0, agg, done, t_log = time.time(), {}, 0, time.time()
        for it, batch in enumerate(loaders["train"]):
            logs = model.training_step(to_device(batch, device), opts, (epoch + it / n_it) * pscale)
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            done = it + 1
            if cfg.log_every_min > 0 and time.time() - t_log > cfg.log_every_min * 60:
                t_log = time.time()
                log(f"  epoch {epoch + 1} it {done}/{n_it} | " + " ".join(f"{k} {v / done:.4f}" for k, v in agg.items())
                    + f" | {time.time() - t0:.0f}s")
            if (cfg.max_train_hours_in_epoch and cfg.max_train_hours > 0
                    and time.time() - t_start > cfg.max_train_hours * 3600):
                budget_hit = True
                log(f"time budget {cfg.max_train_hours} h reached inside epoch {epoch + 1} after {done}/{n_it} iterations")
                break
        agg = {k: v / max(done, 1) for k, v in agg.items()}
        t_train = time.time() - t0
        dist, end, _ = evaluate(model, loaders["val"], cfg.K, device, seed=cfg.seed)
        val = summarize(dist, end)
        tst = None
        if cfg.eval_test_each_epoch:                  # monitoring only, never used for selection or stopping
            dist, end, _ = evaluate(model, loaders["test"], cfg.K, device, seed=cfg.seed)
            tst = summarize(dist, end)
        del dist, end
        for sc in scheds:
            sc.step(val["ADE"]) if cfg.lr_schedule == "plateau" else sc.step()
        improved = val["ADE"] < best - 1e-6
        rec = dict(epoch=epoch + 1, time=time.time() - t0, **{f"train_{k}": v for k, v in agg.items()},
                   val_ADE=val["ADE"], val_FDE=val["FDE"], val_meanADE=val["meanADE"], val_meanFDE=val["meanFDE"],
                   lr=next(iter(opts.values())).param_groups[0]["lr"], selected=improved)
        if tst is not None:
            rec.update(test_ADE=tst["ADE"], test_FDE=tst["FDE"], test_meanADE=tst["meanADE"],
                       test_meanFDE=tst["meanFDE"], test_N=tst["N"])
        history.append(rec)
        with open(os.path.join(run_dir, "history.json"), "w") as fh:
            json.dump(history, fh, indent=2)
        log(f"epoch {epoch + 1:3d}/{cfg.epochs} | " + " ".join(f"{k} {v:.4f}" for k, v in agg.items())
            + f" | val ADE {val['ADE']:.3f} FDE {val['FDE']:.3f}"
            + (f" | test ADE {tst['ADE']:.3f} FDE {tst['FDE']:.3f} single {tst['meanADE']:.3f}/{tst['meanFDE']:.3f}"
               if tst is not None else "") + f" | {time.time() - t0:.1f}s")
        if cfg.epoch_md:
            _append_epoch_md(cfg.epoch_md, run_dir, epoch + 1, val, tst, improved, t_train, time.time() - t0)
        if improved:
            best, bad = val["ADE"], 0
            torch.save(dict(state=model.state_dict(), epoch=epoch + 1, val=val), ckpt)
        else:
            bad += 1
            if cfg.patience > 0 and bad >= cfg.patience:
                log(f"early stop at epoch {epoch + 1} (best val ADE {best:.3f})")
                break
        if budget_hit:
            break
        if cfg.max_train_hours > 0 and time.time() - t_start > cfg.max_train_hours * 3600:
            log(f"time budget {cfg.max_train_hours} h reached after epoch {epoch + 1} (best val ADE {best:.3f})")
            break
    with open(os.path.join(run_dir, "history.json"), "w") as fh:
        json.dump(history, fh, indent=2)
    return dict(best_val_ADE=best, epochs_run=len(history))


def test(model, loader, cfg: Config, run_dir: str, device, dump: bool = True, tag: str = "test") -> dict:
    """Loads the selected checkpoint, evaluates on the held-out split and writes metrics (+ raw dump)."""
    ckpt = os.path.join(run_dir, "best.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device)["state"])
    dist, end, arrays = evaluate(model, loader, cfg.K, device, seed=cfg.seed, dump=dump)
    report = full_report(dist, end)
    with open(os.path.join(run_dir, f"{tag}_metrics.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    if dump:
        np.savez_compressed(os.path.join(run_dir, f"{tag}_dump.npz"), dist=dist.astype(np.float32), **arrays)
    return report
