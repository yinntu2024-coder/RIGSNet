"""Common interface shared by RIGS-Net and all baselines.

Every model sees the same inputs and is scored by the same code:
    batch["hist"] [B, M, T_h, 4]  absolute (x, y, vx, vy) in metres and m/s
    batch["fut"]  [B, M, T_f, 4]
    batch["mask"] [B, M]
Internally, positions are expressed relative to each vessel's last observed position and scaled
by s_pos; velocities are scaled by s_vel (both estimated on the training split). A model outputs
normalised cumulative displacements, which `predict` maps back to absolute metres.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..config import Config


class TrajModel(nn.Module):
    multimodal: bool = True          # False -> deterministic, evaluated with K = 1
    supports_base_cvr: bool = True   # False for models with their own CV handling / training loop (models with their own training loop)

    def __init__(self, cfg: Config, stats: dict):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("s_pos", torch.tensor(float(stats["s_pos"])))
        self.register_buffer("s_vel", torch.tensor(float(stats["s_vel"])))
        self.register_buffer("size_median", torch.tensor(float(stats.get("size_median", 1.0))))

    # ------------------------------------------------------------------ feature helpers
    def feats(self, traj: torch.Tensor, p_last: torch.Tensor) -> torch.Tensor:
        """[B, M, T, 4] absolute -> normalised agent-centric features."""
        return torch.cat([(traj[..., :2] - p_last[:, :, None]) / self.s_pos, traj[..., 2:] / self.s_vel], dim=-1)

    @property
    def base_cvr(self) -> bool:
        return bool(getattr(self.cfg, "baseline_cv_residual", False)) and self.supports_base_cvr

    def cv_disp(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Constant-velocity displacement from the last observed (vx, vy), normalised, [B, M, T_f, 2]."""
        hist = batch["hist"]
        t = torch.arange(1, self.cfg.pred_len + 1, device=hist.device, dtype=hist.dtype) * self.cfg.dt
        return hist[:, :, -1, None, 2:] * t[:, None] / self.s_pos

    def target(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        y = (batch["fut"][..., :2] - batch["hist"][:, :, -1:, :2]) / self.s_pos
        return y - self.cv_disp(batch) if self.base_cvr else y

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean over vessels inside each scene, then mean over scenes. x [B, M], mask [B, M]."""
        m = mask.float()
        return ((x * m).sum(1) / m.sum(1).clamp_min(1.0)).mean()

    # ------------------------------------------------------------------ to be implemented
    def loss(self, batch: Dict[str, torch.Tensor], progress: float) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def sample(self, batch: Dict[str, torch.Tensor], K: int) -> torch.Tensor:
        """Normalised cumulative displacements [B, K, M, T_f, 2]."""
        raise NotImplementedError

    # ------------------------------------------------------------------ shared behaviour
    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor], K: int) -> torch.Tensor:
        """Absolute predicted positions in metres, [B, K, M, T_f, 2]."""
        K = K if self.multimodal else 1
        y = self.sample(batch, K)
        if self.base_cvr:
            y = y + self.cv_disp(batch)[:, None]
        return y * self.s_pos + batch["hist"][:, None, :, -1:, :2]

    def configure_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        return {"main": torch.optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)}

    def training_step(self, batch, opts, progress: float) -> dict:
        loss, logs = self.loss(batch, progress)
        opts["main"].zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.parameters(), self.cfg.grad_clip)
        opts["main"].step()
        logs["loss"] = float(loss.detach())
        return logs
