"""Closest-point-of-approach geometry under constant velocity (used by the STMGCN and DGCN-Transformer baselines)."""
from __future__ import annotations

from typing import Tuple

import torch


def cpa(pos: torch.Tensor, vel: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eq. (dcpa): TCPA and DCPA under the constant-velocity assumption.
    T = max(0, -(r . v) / (||v||^2 + eps)),  D = || r + v T ||  with r = p_n - p_m, v = v_n - v_m."""
    r = pos[:, :, None, :, :] - pos[:, :, :, None, :]
    v = vel[:, :, None, :, :] - vel[:, :, :, None, :]
    tcpa = (-(r * v).sum(-1) / ((v * v).sum(-1) + eps)).clamp_min(0.0)
    dcpa = (r + v * tcpa[..., None]).norm(dim=-1)
    return tcpa, dcpa
