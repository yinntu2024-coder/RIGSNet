"""MSTFormer (Qiang et al., EAAI 2025) with the network modules imported unchanged from the official code
(Method/MSTFormer/MSTFormer/models: DataEmbedding with timeF/minute features, Multi-head Dynamic-aware
Self-attention in the first encoder layer and ProbSparse attention above it, distilling encoder,
generative-style decoder, ATM CNN branch).

Paper settings (arXiv 2303.11540, Sec. IV): inputs [d_sog, d_cog]; d_model 512, 8 heads, 2 encoder + 1 decoder
layers, d_ff 2048, factor 3, dropout 0.05, batch size 50, Adam lr 5e-6 multiplied by 0.5 every epoch;
knowledge-inspired loss: predicted differences are accumulated onto the last sog/cog, positions are
obtained with the motion model and compared with the truth by distance (km).

Adaptations (documented in the manuscript):
  * Lengths: 20 observed positions -> 19 differences (encoder length 19); start token 12 (official 48/72);
    ATMs: int(L/4)+1 evenly selected points as in the official loader.
  * ATM grid: the official 0.25 km cell is for 1-3 min sampling; with 5 s sampling every displacement would
    stay in the centre cell, so the cell is scaled to 0.25 km x 5/60.
  * Decoder ATMs: the official loader draws them from the ground-truth future positions, which are not
    available at prediction time. We draw them from the constant-velocity extrapolation of the last
    observed state.
  * Motion model and distance in local metric coordinates (equivalent to the haversine formula at these
    scales). Standardisation of the differences uses the training split (official StandardScaler).
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from aistraj.models.base import TrajModel

MST_DIR = os.environ.get("MSTFORMER_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                     "third_party", "MSTFormer", "MSTFormer"))
KN = 0.514444


def _import_official():
    sys.modules.setdefault("ipdb", types.ModuleType("ipdb"))
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "models" or k.startswith("models.")
             or k == "utils" or k.startswith("utils.")}
    sys.path.insert(0, MST_DIR)
    try:
        from models.model import MSTFormer
        from utils.timefeatures import time_features
    finally:
        sys.path.remove(MST_DIR)
        for k in [k for k in sys.modules if k == "models" or k.startswith("models.") or k == "utils"
                  or k.startswith("utils.")]:
            sys.modules["_mst_" + k] = sys.modules.pop(k)
        sys.modules.update(saved)
    # official ProbAttention.data_normal_2d loops over rows in Python; the row shift cancels, so it equals a
    # row-wise min-max normalisation. Vectorised replacement (identical output, checked in tests).
    atten = sys.modules["_mst_models.atten"]
    atten.ProbAttention._official_data_normal_2d = atten.ProbAttention.data_normal_2d
    atten.ProbAttention.data_normal_2d = _data_normal_2d
    return MSTFormer, time_features


def _data_normal_2d(self, orign_data, dim="col"):
    d = 1 if dim == "col" else 0
    mn, mx = orign_data.min(dim=d, keepdim=True)[0], orign_data.max(dim=d, keepdim=True)[0]
    return torch.sub(orign_data, mn).true_divide(mx - mn)


def even_select(N, M):
    """Official data_loader.even_select."""
    if N == M:
        return np.ones(N, dtype=bool)
    if M > N / 2:
        cut = np.zeros(N, dtype=int)
        q, r = divmod(N, N - M)
        cut[[q * i + min(i, r) for i in range(N - M)]] = True
    else:
        cut = np.ones(N, dtype=int)
        q, r = divmod(N, M)
        cut[[q * i + min(i, r) for i in range(M)]] = False
    return np.logical_not(cut)


# official Featuremap constants
SHIP_PROB = 0.8
NB = {0: 0.50002, 1: 0.114675, 2: 0.00516}
MAXVEL_KM = 5 * 1.852
RADIO = [int(round(MAXVEL_KM * 0.8 / 0.25)), int(round(MAXVEL_KM * 1.0 / 0.25))]     # [30, 37]


HEADING_PROB, SPEED_PROB, SPEED_LINE = 0.2, 0.02, 3


def atm(points, cell_m, sog_kn=None, cog_deg=None, heading_deg=None):
    """Vectorised Featuremap.Map2D_current.label_speed_heading for consecutive pairs of `points` [N, P, 2]
    (east, north, metres). sog/cog/heading [N, P] describe each point (the official map uses those of the
    earlier point of the pair). Returns [N, P-1, 2*R0+1, 2*R1+1]."""
    N, P, _ = points.shape
    dev = points.device
    d = points[:, 1:] - points[:, :-1]
    y = (-torch.round(d[..., 1] / cell_m)).clamp(-(RADIO[0] - 3), RADIO[0] - 3).long()
    x = torch.round(d[..., 0] / cell_m).clamp(-(RADIO[1] - 3), RADIO[1] - 3).long()
    out = torch.zeros(N, P - 1, 2 * RADIO[0] + 1, 2 * RADIO[1] + 1, device=dev)
    c0, c1 = RADIO
    out[:, :, c0, c1] = SHIP_PROB
    cx, cy = c0 + y, c1 + x
    n = torch.arange(N, device=dev)[:, None].expand(N, P - 1)
    p = torch.arange(P - 1, device=dev)[None].expand(N, P - 1)
    for dx in (-2, -1, 0, 1, 2):                                                  # official loop order
        for dy in (-2, -1, 0, 1, 2):
            dist = abs(dx) + abs(dy)
            if dist <= 2:
                out[n, p, cx + dx, cy - dy] = NB[dist]
    if heading_deg is not None:                                                   # heading mark
        th = torch.deg2rad(heading_deg[:, :-1])
        r, c = c0 - torch.round(torch.cos(th)).long(), c1 + torch.round(torch.sin(th)).long()
        out.index_put_((n, p, r, c), torch.full_like(th, HEADING_PROB), accumulate=True)
    if sog_kn is not None:                                                        # speed tail (sog in km/h)
        th = torch.deg2rad(cog_deg[:, :-1])
        val = sog_kn[:, :-1] * 1.852 * SPEED_PROB
        for i in range(SPEED_LINE):
            q = i + 1
            r, c = c0 + torch.round(q * torch.cos(th)).long(), c1 - torch.round(q * torch.sin(th)).long()
            out.index_put_((n, p, r, c), val, accumulate=True)
    return out


class MSTFormerOfficial(TrajModel):
    multimodal = False
    supports_base_cvr = False

    def __init__(self, cfg, stats, label_len=12, d_model=512):
        super().__init__(cfg, stats)
        MST, self._time_features = _import_official()
        self.Le, self.Lb, self.Tp = cfg.obs_len - 1, label_len, cfg.pred_len
        self.nx, self.ny = int(self.Le / 4 + 1), int(self.Tp / 4 + 1)
        self.cell = 250.0 * cfg.dt / 60.0
        self.t0 = float(stats.get("t0", 0.0))
        self.net = MST(enc_in=2, dec_in=2, c_out=2, seq_len=self.Le, label_len=label_len, out_len=self.Tp,
                       prob_use=True, factor=3, d_model=d_model, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
                       dropout=0.05, attn='prob', embed='timeF', freq='t', activation='gelu',
                       output_attention=False, distil=True, mix=True, device=torch.device("cuda"))
        with torch.no_grad():                                                     # size of the CNN output head
            f = self._cnn_feat(self.net.encnn, self.nx - 1)
            g = self._cnn_feat(self.net.decnn, self.ny - 1)
        self.net.encnn.out = nn.Linear(f, d_model)
        self.net.decnn.out = nn.Linear(g, d_model)
        self.register_buffer("d_mu", torch.tensor(stats.get("mst_mu", [0.0, 0.0])))
        self.register_buffer("d_sd", torch.tensor(stats.get("mst_sd", [1.0, 1.0])))

    @staticmethod
    def _cnn_feat(cnn, depth):
        x = torch.zeros(1, depth, 2 * RADIO[0] + 1, 2 * RADIO[1] + 1).unsqueeze(1)
        x = cnn.conv3(cnn.conv2(cnn.conv1(x)))
        return int(x[0, 0].numel())

    # ------------------------------------------------------------------ inputs
    @staticmethod
    def sog_cog(v):
        sog = v.norm(dim=-1) / KN                                                  # knots
        cog = torch.rad2deg(torch.atan2(v[..., 0], v[..., 1])) % 360.0             # degrees
        return sog, cog

    @staticmethod
    def deltas(v):
        sog, cog = MSTFormerOfficial.sog_cog(v)
        ds = sog[:, 1:] - sog[:, :-1]
        dc = (cog[:, 1:] - cog[:, :-1] + 180.0) % 360.0 - 180.0
        return torch.stack([ds, dc], -1)                                          # [N, T-1, 2]

    def marks(self, f_last, L, device):
        """timeF/'t' features (official utils.timefeatures, freq 't'): MinuteOfHour, HourOfDay, DayOfWeek,
        DayOfMonth, DayOfYear, each scaled to [-0.5, 0.5]; vectorised, identical values (checked in tests)."""
        fr = f_last.cpu().numpy()[:, None] + np.arange(-L + 1, 1)[None]
        t = (self.t0 + fr * self.cfg.dt).astype("datetime64[s]")
        day = t.astype("datetime64[D]")
        sec = (t - day).astype(np.int64)
        minute, hour = (sec // 60) % 60, sec // 3600
        dow = (day.astype(np.int64) + 3) % 7                                      # 1970-01-01 was a Thursday
        month_start = day.astype("datetime64[M]").astype("datetime64[D]")
        year_start = day.astype("datetime64[Y]").astype("datetime64[D]")
        dom = (day - month_start).astype(np.int64) + 1
        doy = (day - year_start).astype(np.int64) + 1
        tf = np.stack([minute / 59.0 - 0.5, hour / 23.0 - 0.5, dow / 6.0 - 0.5, (dom - 1) / 30.0 - 0.5,
                       (doy - 1) / 365.0 - 0.5], -1)
        return torch.as_tensor(tf, dtype=torch.float32, device=device)

    def forward_det(self, batch):
        hist, mask = batch["hist"], batch["mask"]
        B, M, T, _ = hist.shape
        h = hist[mask]                                                            # [N, T, 4]
        N = h.shape[0]
        dev = h.device
        x = (self.deltas(h[..., 2:]) - self.d_mu) / self.d_sd                     # [N, Le, 2] standardised
        f_obs = batch["f_obs"][:, None].expand(B, M)[mask]
        x_mark = self.marks(f_obs, self.Le, dev)
        y_mark = torch.cat([x_mark[:, -self.Lb:], self.marks(f_obs + self.Tp, self.Tp, dev)], 1)
        x_dec = torch.cat([x[:, -self.Lb:], torch.zeros(N, self.Tp, 2, device=dev)], 1)
        pos = h[..., :2]
        sel_x = torch.as_tensor(even_select(T, self.nx), device=dev)
        cvf = pos[:, -1:] + h[:, -1:, 2:] * (torch.arange(1, self.Tp + 1, device=dev)[None, :, None] * self.cfg.dt)
        sel_y = torch.as_tensor(even_select(self.Tp, self.ny), device=dev)
        sog_o, cog_o = self.sog_cog(h[..., 2:])                                    # heading := COG (no heading field)
        x_map = atm(pos[:, sel_x], self.cell, sog_o[:, sel_x], cog_o[:, sel_x], cog_o[:, sel_x])
        n_y = int(sel_y.sum())
        y_map = atm(cvf[:, sel_y], self.cell, sog_o[:, -1:].expand(-1, n_y), cog_o[:, -1:].expand(-1, n_y),
                    cog_o[:, -1:].expand(-1, n_y))
        out = self.net(x, x_mark, x_map, y_map, x_dec, y_mark)                    # [N, Tp, 2] standardised diffs
        d = out * self.d_sd + self.d_mu                                           # inverse transform
        sog0, cog0 = self.sog_cog(h[:, -1, 2:])
        sog = sog0[:, None] + d[..., 0].cumsum(1)
        cog = cog0[:, None] + d[..., 1].cumsum(1)
        rad = torch.deg2rad(cog)
        step = torch.stack([sog * KN * torch.sin(rad), sog * KN * torch.cos(rad)], -1) * self.cfg.dt
        y = torch.zeros(B, M, self.Tp, 2, device=dev)
        y[mask] = step.cumsum(1) / self.s_pos
        return y

    def sample(self, batch, K):
        return self.forward_det(batch)[:, None]

    def loss(self, batch, progress):
        y_hat = self.forward_det(batch)
        y = (batch["fut"][..., :2] - batch["hist"][:, :, -1:, :2]) / self.s_pos
        dist_km = ((y_hat - y).norm(dim=-1) * self.s_pos / 1000.0)[batch["mask"]].mean()
        return dist_km, dict(dist_km=dist_km.item())

    def configure_optimizers(self):
        return {"main": torch.optim.Adam(self.parameters(), lr=self.cfg.lr)}
