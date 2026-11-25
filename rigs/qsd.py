"""Quaternion Ship Domain (QSD) geometry, analytical prior, training penalty and evaluation metrics.

Coordinates: local east-north metres. Course psi is clockwise from north, so the forward unit vector
is f = (sin psi, cos psi) and the starboard unit vector is s = (cos psi, -sin psi) = (f_y, -f_x).
Receiver-fixed coordinates of a relative vector v: x_loc = v . s (starboard +), y_loc = v . f (fore +).
"""
from __future__ import annotations

import numpy as np
import torch


def radii(L, B):
    """(fore, aft, starboard, port) radii of the manuscript: 3L, 2L, 1.5B, 1.5B.
    L, B: scalars or tensors broadcastable to the pair shape (receiver dimensions)."""
    return 3.0 * L, 2.0 * L, 1.5 * B, 1.5 * B


def vessel_dims(model, batch):
    """Receiver dimensions [B, M] (metres): per-vessel AIS length / width when the data set provides them
    (buffers seg_L / seg_B attached by rigs.dims.attach_dims), otherwise the manuscript defaults."""
    mask = batch["mask"]
    if getattr(model, "seg_L", None) is not None:
        seg = batch["seg"].clamp_min(0)
        return model.seg_L[seg], model.seg_B[seg]
    full = torch.ones(mask.shape, device=mask.device)
    return full * model.cfg.L_default, full * model.cfg.B_default


def heading_unit(v: torch.Tensor, fallback: torch.Tensor | None = None, min_norm: float = 1e-3) -> torch.Tensor:
    """Unit vector of v [..., 2]; where |v| < min_norm the fallback unit vector (or north) is used."""
    n = v.norm(dim=-1, keepdim=True)
    u = v / n.clamp_min(min_norm)
    if fallback is None:
        fallback = torch.zeros_like(v)
        fallback[..., 1] = 1.0
    return torch.where(n > min_norm, u, fallback)


def qsd_radius(rel: torch.Tensor, fwd: torch.Tensor, L: float, B: float, eps: float = 1e-6):
    """Radial distance Gamma(theta; H_i) from receiver i to its QSD boundary along the bearing of j.
    rel [..., 2] = p_j - p_i (metres), fwd [..., 2] = forward unit vector of the receiver.
    Returns (d_safe, d) with d = |rel|."""
    Rf, Ra, Rs, Rp = radii(L, B)
    st = torch.stack([fwd[..., 1], -fwd[..., 0]], dim=-1)
    xl, yl = (rel * st).sum(-1), (rel * fwd).sum(-1)
    d = torch.sqrt(xl ** 2 + yl ** 2 + eps)
    ux, uy = xl / d, yl / d
    Rf, Ra, Rs, Rp = (torch.as_tensor(r, dtype=ux.dtype, device=ux.device) for r in (Rf, Ra, Rs, Rp))
    Rx = torch.where(ux >= 0, Rs, Rp).expand_as(ux)
    Ry = torch.where(uy >= 0, Rf, Ra).expand_as(uy)
    d_safe = 1.0 / torch.sqrt((ux / Rx) ** 2 + (uy / Ry) ** 2 + 1e-12)
    return d_safe, d


def prior(d_safe: torch.Tensor, d: torch.Tensor, eta: float, eps: float) -> torch.Tensor:
    """Bounded analytical prior rho_bar = sigmoid(eta (d_safe - d) / (d_safe + eps))."""
    return torch.sigmoid(eta * (d_safe - d) / (d_safe + eps))


def future_forward(pred: torch.Tensor, p_last: torch.Tensor, f_last: torch.Tensor) -> torch.Tensor:
    """Receiver course along the predicted path: direction from the previous to the current predicted
    position (the last observed position precedes the first prediction). Where the step is shorter than
    1 mm the last observed course is kept. pred [B, M, T, 2], p_last [B, M, 2], f_last [B, M, 2]."""
    prev = torch.cat([p_last[:, :, None], pred[:, :, :-1]], dim=2)
    return heading_unit(pred - prev, f_last[:, :, None].expand_as(pred))


def pair_mask(mask: torch.Tensor) -> torch.Tensor:
    """Valid ordered pairs (i != j) of a padded scene. mask [B, M] -> [B, M, M]."""
    M = mask.shape[1]
    eye = torch.eye(M, dtype=torch.bool, device=mask.device)
    return mask[:, :, None] & mask[:, None, :] & ~eye


def safe_penalty(pred, p_last, f_last, mask, w, L, B, scale):
    """L_safe (Eq. safe_loss). pred [B, M, T, 2] absolute metres; w [B, M, M] = rho_bar at the last
    observed frame (no gradient); L, B [B, M] receiver dimensions (or scalars). Penetration is measured in
    units of `scale` metres.
    Returns the mean over valid ordered pair/time states (0 if there is none)."""
    fwd = future_forward(pred, p_last, f_last)                                    # [B, M, T, 2]
    rel = pred[:, None, :, :, :] - pred[:, :, None, :, :]                         # [B, i, j, T, 2] = p_j - p_i
    if torch.is_tensor(L):
        L, B = L[:, :, None, None], B[:, :, None, None]
    d_safe, d = qsd_radius(rel, fwd[:, :, None].expand_as(rel), L, B)
    pen = torch.relu(d_safe - d) / scale
    pm = pair_mask(mask)[..., None].float()                                       # [B, M, M, 1]
    num = (w.detach()[..., None] * pen ** 2 * pm).sum()
    den = pm.sum() * pred.shape[2]
    return num / den.clamp_min(1.0)


# ------------------------------------------------------------------ evaluation (numpy, from test dumps)
def scene_qsd_stats(pos: np.ndarray, p_last: np.ndarray, f_last: np.ndarray, L: float, B: float):
    """pos [M, T, 2] predicted (or true) future positions of the vessels of one scene.
    L, B: scalars or per-vessel arrays [M] (receiver dimensions).
    Returns (#violating ordered pair/time states, #ordered pair/time states, minimum separation)."""
    M, T, _ = pos.shape
    if M < 2:
        return 0, 0, np.nan
    with torch.no_grad():
        P = torch.from_numpy(pos[None]).double()
        fwd = future_forward(P, torch.from_numpy(p_last[None]).double(), torch.from_numpy(f_last[None]).double())
        rel = P[:, None] - P[:, :, None]
        if np.ndim(L):
            L = torch.as_tensor(np.asarray(L, np.float64))[None, :, None, None]
            B = torch.as_tensor(np.asarray(B, np.float64))[None, :, None, None]
        d_safe, d = qsd_radius(rel, fwd[:, :, None].expand_as(rel), L, B, eps=0.0)
        off = ~torch.eye(M, dtype=torch.bool)[None, :, :, None].expand_as(d)
        viol = int(((d < d_safe) & off).sum())
        n = int(off.sum())
        sep = float(d[off].min())
    return viol, n, sep
