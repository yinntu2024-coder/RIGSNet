"""Per-vessel dimensions for the QSD geometry.

<data root>/<region>/seg_dims.npz (written by tools/build_dma_regions.py --stage dims) holds, for every
trajectory segment of tracks.npz, the vessel length L and beam B in metres (median of the AIS static
reports of that MMSI; manuscript defaults where missing or implausible). Data sets without the file keep the defaults cfg.L_default / cfg.B_default.
"""
from __future__ import annotations

import os

import numpy as np
import torch

L_RANGE = (5.0, 500.0)
B_RANGE = (2.0, 80.0)


def load_seg_dims(root: str):
    p = os.path.join(root, "seg_dims.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z["L"].astype(np.float32), z["B"].astype(np.float32)


def attach_dims(net: torch.nn.Module, root: str) -> bool:
    """Register seg_L / seg_B (non-persistent buffers, indexed by batch['seg']) on the model."""
    d = load_seg_dims(root)
    if d is None:
        return False
    net.register_buffer("seg_L", torch.from_numpy(d[0]), persistent=False)
    net.register_buffer("seg_B", torch.from_numpy(d[1]), persistent=False)
    return True


def clean_dims(L, B, L_default: float, B_default: float):
    L = np.asarray(L, np.float64); B = np.asarray(B, np.float64)
    okL = np.isfinite(L) & (L >= L_RANGE[0]) & (L <= L_RANGE[1])
    okB = np.isfinite(B) & (B >= B_RANGE[0]) & (B <= B_RANGE[1])
    return np.where(okL, L, L_default), np.where(okB, B, B_default), okL & okB
