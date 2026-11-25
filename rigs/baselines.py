"""Baselines added for the RIGS-Net comparison. Compact re-implementations of the core design of each
method inside the shared data/scoring pipeline (same inputs, split, normalisation and metric code).

* AISFormer (Yu et al., OE 2025): per-vessel Transformer with a sparse state representation (motion
  states as tokens; only the states that change are attended through a learned sparsity gate) and a
  frequency-aware branch (rFFT of the observed motion), fused before a one-shot decoder.
* MSTFormer (Qiang et al., EAAI 2025): Informer-style encoder-decoder with dynamic-aware attention
  (attention biased by the change of speed and course of the keys) and the knowledge-inspired loss,
  i.e. the decoder outputs per-step speed/course changes that are integrated kinematically into
  positions, and the loss is the distance between integrated and true positions. The map-image CNN
  branch of the public code is omitted (no chart images for these regions).
* DGCN-Transformer (Zhang et al., EAAI 2025): dynamic graph whose adjacency combines a collision-risk
  prior (DCPA/TCPA) with a learned attention graph, two graph-convolution layers per observed frame,
  a Transformer temporal encoder and an MLP decoder.

All three produce one trajectory (K = 1). `out_mode` ("direct" by default) can switch any of them to the
constant-velocity offset parameterisation for the fairness check.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from aistraj.graphs import cpa

from . import qsd
from .model import Deterministic, sinusoid, state_features, wrap


class ConstantVelocity(Deterministic):
    def __init__(self, cfg, stats):
        super().__init__(cfg, stats)
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward_det(self, batch):
        return self.cv_disp(batch)


def mse_loss(model, y_hat, batch):
    y = (batch["fut"][..., :2] - batch["hist"][:, :, -1:, :2]) / model.s_pos
    return model.masked_mean(((y_hat - y) ** 2).sum(-1).mean(-1), batch["mask"])


class AISFormer(Deterministic):
    def __init__(self, cfg, stats, d=128, layers=3, heads=4):
        super().__init__(cfg, stats)
        self.embed = nn.Linear(9, d)
        self.gate = nn.Sequential(nn.Linear(9 + 9, d // 2), nn.ReLU(), nn.Linear(d // 2, 1))   # sparse state gate
        self.register_buffer("pe", sinusoid(cfg.obs_len, d))
        nf = cfg.obs_len // 2 + 1
        self.freq = nn.Sequential(nn.Linear(4 * nf * 2, d), nn.ReLU(), nn.Linear(d, d))      # frequency branch
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, cfg.dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.fuse = nn.Linear(2 * d, d)
        self.dec = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, cfg.pred_len * 2))

    def forward_det(self, batch):
        hist = batch["hist"]
        B, M, T, _ = hist.shape
        x = state_features(hist, self.s_pos, self.s_vel, *qsd.vessel_dims(self, batch)).reshape(B * M, T, -1)
        dx = torch.diff(x, dim=1, prepend=x[:, :1])
        g = torch.sigmoid(self.gate(torch.cat([x, dx], -1)))                      # [N, T, 1]
        h = self.enc(self.embed(x) * g + self.pe)
        spec = torch.fft.rfft(x[..., :4], dim=1)                                  # [N, nf, 4]
        f = self.freq(torch.cat([spec.real, spec.imag], -1).reshape(B * M, -1))
        z = self.fuse(torch.cat([(h * g).sum(1) / g.sum(1).clamp_min(1e-3), f], -1))
        return self.out_to_disp(self.dec(z).view(B, M, self.cfg.pred_len, 2), batch)

    def loss(self, batch, progress):
        l = mse_loss(self, self.forward_det(batch), batch)
        return l, dict(mse=l.item())


class DynAttention(nn.Module):
    """Multi-head attention whose logits are biased by a learned function of the keys' motion change."""

    def __init__(self, d, heads, dropout):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.bias = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, heads))
        self.drop = nn.Dropout(dropout)

    def forward(self, xq, xk, dyn=None, causal=False):
        N, Tq, _ = xq.shape
        Tk = xk.shape[1]
        q = self.q(xq).view(N, Tq, self.h, self.dh).transpose(1, 2)
        k = self.k(xk).view(N, Tk, self.h, self.dh).transpose(1, 2)
        v = self.v(xk).view(N, Tk, self.h, self.dh).transpose(1, 2)
        a = q @ k.transpose(-1, -2) / math.sqrt(self.dh)                          # [N, h, Tq, Tk]
        if dyn is not None:
            a = a + self.bias(dyn).permute(0, 2, 1)[:, :, None, :]
        if causal:
            a = a.masked_fill(torch.ones(Tq, Tk, dtype=torch.bool, device=a.device).triu(1), float("-inf"))
        out = (self.drop(torch.softmax(a, -1)) @ v).transpose(1, 2).reshape(N, Tq, -1)
        return self.o(out)


class _Block(nn.Module):
    def __init__(self, d, heads, dropout, cross=False):
        super().__init__()
        self.att = DynAttention(d, heads, dropout)
        self.cross = DynAttention(d, heads, dropout) if cross else None
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, dyn=None, mem=None, mem_dyn=None, causal=False):
        x = self.n1(x + self.drop(self.att(x, x, dyn, causal)))
        if self.cross is not None:
            x = self.n2(x + self.drop(self.cross(x, mem, mem_dyn)))
        return self.n3(x + self.drop(self.ff(x)))


class MSTFormer(Deterministic):
    def __init__(self, cfg, stats, d=256, heads=8, e_layers=2, d_layers=1, label_len=10):
        super().__init__(cfg, stats)
        self.label_len = label_len
        self.enc_embed = nn.Linear(9, d)
        self.dec_embed = nn.Linear(9, d)
        self.register_buffer("pe", sinusoid(cfg.obs_len + cfg.pred_len, d))
        self.enc = nn.ModuleList(_Block(d, heads, cfg.dropout) for _ in range(e_layers))
        self.dec = nn.ModuleList(_Block(d, heads, cfg.dropout, cross=True) for _ in range(d_layers))
        self.proj = nn.Linear(d, 2)                                               # (d speed, d course) per step
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)          # start from constant velocity

    def kin(self, batch):
        """Per-step changes (speed in s_vel units, course in rad) for the decoder, integrated into positions."""
        hist = batch["hist"]
        B, M, T, _ = hist.shape
        x = state_features(hist, self.s_pos, self.s_vel, *qsd.vessel_dims(self, batch)).reshape(B * M, T, -1)
        v = hist[..., 2:].reshape(B * M, T, 2)
        spd = v.norm(dim=-1)
        psi = torch.atan2(v[..., 0], v[..., 1])
        dyn = torch.stack([torch.diff(spd, dim=1, prepend=spd[:, :1]) / self.s_vel,
                           wrap(torch.diff(psi, dim=1, prepend=psi[:, :1])) / math.pi], -1)
        h = self.enc_embed(x) + self.pe[:T]
        for blk in self.enc:
            h = blk(h, dyn)
        Lb, Tp = self.label_len, self.cfg.pred_len
        xd = torch.cat([x[:, -Lb:], torch.zeros_like(x[:, :Tp])], 1)             # start token + placeholders
        g = self.dec_embed(xd) + self.pe[T - Lb:T + Tp]
        for blk in self.dec:
            g = blk(g, None, h, dyn, causal=True)
        delta = 0.5 * torch.tanh(self.proj(g[:, -Tp:]))                           # [N, T_p, 2], bounded per step
        spd_f = spd[:, -1:] + delta[..., 0].cumsum(1) * self.s_vel                # m/s
        psi_f = psi[:, -1:] + delta[..., 1].cumsum(1)                             # rad (<= 0.5 rad per 5 s step)
        step = torch.stack([spd_f * torch.sin(psi_f), spd_f * torch.cos(psi_f)], -1) * self.cfg.dt
        return (step.cumsum(1) / self.s_pos).view(B, M, Tp, 2)

    def forward_det(self, batch):
        # zero predicted changes reproduce constant-velocity extrapolation, so out_mode does not apply
        return self.kin(batch)

    def loss(self, batch, progress):
        l = self.pred_loss(self.forward_det(batch), batch)                        # distance on integrated positions
        return l, dict(dist=l.item())


class DGCNTransformer(Deterministic):
    def __init__(self, cfg, stats, d=128, layers=2, heads=4, sigma_d=1852.0, sigma_t=600.0):
        super().__init__(cfg, stats)
        self.sigma_d, self.sigma_t = sigma_d, sigma_t
        self.embed = nn.Linear(9, d)
        self.qk = nn.Linear(d, 2 * d)
        self.mix = nn.Parameter(torch.tensor(0.0))
        self.gcn = nn.ModuleList(nn.Linear(d, d) for _ in range(2))
        self.register_buffer("pe", sinusoid(cfg.obs_len, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, cfg.dropout, batch_first=True)
        self.tf = nn.TransformerEncoder(layer, layers)
        self.dec = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, cfg.pred_len * 2))

    def forward_det(self, batch):
        hist, mask = batch["hist"], batch["mask"]
        B, M, T, _ = hist.shape
        pos, vel = hist[..., :2].transpose(1, 2), hist[..., 2:].transpose(1, 2)    # [B, T, M, 2]
        tcpa, dcpa = cpa(pos, vel)
        risk = torch.exp(-dcpa / self.sigma_d) * torch.exp(-tcpa.clamp_min(0) / self.sigma_t)   # collision-risk prior
        eye = torch.eye(M, dtype=torch.bool, device=hist.device)
        pm = (mask[:, None, :, None] & mask[:, None, None, :]).expand(B, T, M, M)
        risk = torch.where(eye, torch.ones_like(risk), risk) * pm
        a_r = risk / risk.sum(-1, keepdim=True).clamp_min(1e-6)
        h = self.embed(state_features(hist, self.s_pos, self.s_vel, *qsd.vessel_dims(self, batch))).transpose(1, 2)          # [B, T, M, d]
        for lin in self.gcn:
            q, k = self.qk(h).chunk(2, -1)
            att = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            att = torch.softmax(att.masked_fill(~pm, float("-inf")), -1).nan_to_num(0.0)
            w = torch.sigmoid(self.mix)
            h = h + F.relu(lin((w * a_r + (1 - w) * att) @ h))
        seq = h.permute(0, 2, 1, 3).reshape(B * M, T, -1) + self.pe
        z = self.tf(seq)[:, -1]
        return self.out_to_disp(self.dec(z).view(B, M, self.cfg.pred_len, 2), batch)

    def loss(self, batch, progress):
        l = mse_loss(self, self.forward_det(batch), batch)
        return l, dict(mse=l.item())
