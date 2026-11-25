"""Baselines without any public or local code release, implemented from their published descriptions.
Settings that the available descriptions do not fix are listed in the manuscript's baseline table.

* STMGCN (Liu et al., IEEE TII 2022): built on the Social-STGCNN framework (official modules reused):
  three graphs from the social force (inverse distance), the time to closest point of approach and the
  size of the surrounding vessels; a spatio-temporal multigraph convolution layer (one graph convolution
  per graph, summed, followed by the ST-GCN temporal unit); a self-attention temporal convolution layer
  mapping the observed to the future time axis; bivariate Gaussian output; Social-STGCNN training.
* G-STGAN (Zhang et al., OE 2023, "gated spatio-temporal graph aggregation network"): spatial gated encoder
  (gated graph convolution over the vessels of each frame), temporal gated encoder (gate between a
  short-term causal convolution and long-term self-attention), temporal convolutional network aggregating
  the two streams; deterministic, MSE loss.
* DGCN-Transformer (Zhang et al., EAAI 2025): dynamic graph per frame combining a collision-risk prior
  (QSD-normalised DCPA and TCPA) with a learned attention graph, two graph convolutions, Transformer
  temporal encoder, MLP decoder; deterministic, MSE loss.
* AISFormer (Yu et al., OE 2025): TrAISformer-style causal Transformer over four-hot tokens (discretised
  position x, position y, speed, course) trained as multi-head classification, with frequency-aware
  attention (FFT branch inside each block), multi-resolution supervision (fine + coarse position bins)
  and a physics-informed consistency term between the predicted displacement and the predicted
  speed/course. Autoregressive decoding with the arg-max class (single output).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from aistraj.graphs import cpa
from aistraj.models.base import TrajModel

from . import qsd
from .model import Deterministic, sinusoid, state_features
from .official import SocialSTGCNNOfficial, bivariate_nll, st_gcn

KN = 0.514444


# ======================================================================= STMGCN
def norm_lap(A, mask):
    """networkx normalized Laplacian of graphs with unit self-loops, padded nodes isolated. A [n, T, N, N]."""
    N = A.shape[-1]
    eye = torch.eye(N, dtype=torch.bool, device=A.device)
    pm = (mask[:, None, :, None] & mask[:, None, None, :]).float()
    A = torch.where(eye, torch.ones_like(A), A) * pm
    deg = A.sum(-1)
    dinv = torch.where(deg > 0, deg.clamp_min(1e-12).rsqrt(), torch.zeros_like(deg))
    return torch.eye(N, device=A.device) * (deg > 0).float()[..., None] - dinv[..., :, None] * A * dinv[..., None, :]


class SATCN(nn.Module):
    """Self-attention temporal convolution: self-attention along the observed time axis of every node,
    followed by the time-extrapolating convolutions (T_o -> T_p channels)."""

    def __init__(self, c, To, Tp, n_conv=3):
        super().__init__()
        self.q, self.k = nn.Linear(c, c), nn.Linear(c, c)
        self.first = nn.Conv2d(To, Tp, 3, padding=1)
        self.convs = nn.ModuleList(nn.Conv2d(Tp, Tp, 3, padding=1) for _ in range(n_conv - 1))
        self.prelus = nn.ModuleList(nn.PReLU() for _ in range(n_conv))
        self.out = nn.Conv2d(Tp, Tp, 3, padding=1)

    def forward(self, v):                                                         # v [n, c, T, N]
        x = v.permute(0, 3, 2, 1)                                                 # [n, N, T, c]
        att = torch.softmax(self.q(x) @ self.k(x).transpose(-1, -2) / math.sqrt(x.shape[-1]), -1)
        x = x + att @ x
        v = x.permute(0, 2, 1, 3)                                                 # [n, T, N, c] (time as channels)
        v = self.prelus[0](self.first(v))
        for conv, act in zip(self.convs, self.prelus[1:]):
            v = act(conv(v)) + v
        return self.out(v)                                                        # [n, T_p, N, c]


class STMGCN(SocialSTGCNNOfficial):
    def __init__(self, cfg, stats):
        TrajModel.__init__(self, cfg, stats)
        self.units = nn.ModuleList(st_gcn(2, 5, (3, cfg.obs_len)) for _ in range(3))
        self.satcn = SATCN(5, cfg.obs_len, cfg.pred_len)

    def graphs(self, hist, mask, L):
        pos, vel = hist[..., :2].transpose(1, 2), hist[..., 2:].transpose(1, 2)    # [n, T, N, 2]
        d = (pos[:, :, :, None] - pos[:, :, None]).norm(dim=-1)
        a_sf = 1.0 / (d / 1852.0 + 1e-3)                                          # social force, nmi
        tcpa, _ = cpa(pos, vel)
        a_t = torch.where(tcpa > 0, 1.0 / (1.0 + tcpa / 60.0), torch.zeros_like(tcpa))   # TCPA in minutes
        a_sz = (L[:, None, None, :] / 1852.0) / (d / 1852.0 + 1e-3)               # size: neighbour length (AIS)
        return [norm_lap(a, mask) for a in (a_sf, a_t, a_sz)]

    def params(self, batch):
        hist, mask = batch["hist"], batch["mask"]
        V = self._rel(hist, hist[:, :, 0, :2]).transpose(1, 2)
        v = V.permute(0, 3, 1, 2)                                                 # [n, 2, T, N]
        h = sum(u(v, A)[0] for u, A in zip(self.units, self.graphs(hist, mask, qsd.vessel_dims(self, batch)[0])))  # multigraph ST layer
        out = self.satcn(h)                                                       # [n, T_p, N, 5]
        return out.permute(0, 2, 1, 3)


# ======================================================================= G-STGAN
class GatedGraphConv(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin, self.gate = nn.Linear(2 * d, d), nn.Linear(2 * d, d)

    def forward(self, h, A):                                                      # h [n, T, N, d], A [n, T, N, N]
        z = torch.cat([h, A @ h], -1)
        return h + torch.sigmoid(self.gate(z)) * torch.tanh(self.lin(z))


class GSTGAN(Deterministic):
    def __init__(self, cfg, stats, d=64, heads=4):
        super().__init__(cfg, stats)
        self.embed = nn.Linear(9, d)
        self.sge = nn.ModuleList(GatedGraphConv(d) for _ in range(2))
        self.short = nn.Conv1d(d, d, 3, padding=2)                                # causal (trimmed)
        self.long = nn.MultiheadAttention(d, heads, dropout=cfg.dropout, batch_first=True)
        self.tgate = nn.Linear(2 * d, d)
        self.register_buffer("pe", sinusoid(cfg.obs_len, d))
        chans = [2 * d, d, d, d]
        self.tcn = nn.ModuleList(nn.Conv1d(a, b, 3, padding=2 * 2 ** i, dilation=2 ** i)
                                 for i, (a, b) in enumerate(zip(chans[:-1], chans[1:])))
        self.res = nn.Conv1d(2 * d, d, 1)
        self.head = nn.Linear(d, cfg.pred_len * 2)

    def forward_det(self, batch):
        hist, mask = batch["hist"], batch["mask"]
        B, M, T, _ = hist.shape
        x = self.embed(state_features(hist, self.s_pos, self.s_vel, *qsd.vessel_dims(self, batch)))
        pos = hist[..., :2].transpose(1, 2)
        dist = (pos[:, :, :, None] - pos[:, :, None]).norm(dim=-1) / 1852.0
        pm = qsd.pair_mask(mask)[:, None]
        A = torch.where(pm, 1.0 / (dist + 1e-3), torch.zeros_like(dist))
        A = A / A.sum(-1, keepdim=True).clamp_min(1e-6)
        hs = x.transpose(1, 2)                                                    # [B, T, M, d]
        for g in self.sge:
            hs = g(hs, A)                                                         # spatial gated encoder
        ht = x.reshape(B * M, T, -1)                                              # temporal gated encoder
        s = self.short(ht.transpose(1, 2))[..., :T].transpose(1, 2)
        l, _ = self.long(ht + self.pe, ht + self.pe, ht + self.pe,
                         attn_mask=torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1))
        g = torch.sigmoid(self.tgate(torch.cat([s, l], -1)))
        ht = g * s + (1 - g) * l
        z = torch.cat([hs.transpose(1, 2).reshape(B * M, T, -1), ht], -1).transpose(1, 2)   # [BM, 2d, T]
        y = z
        for i, conv in enumerate(self.tcn):
            y = F.relu(conv(y)[..., :T])
        y = y + self.res(z)
        return self.head(y[..., -1]).view(B, M, self.cfg.pred_len, 2)

    def loss(self, batch, progress):
        y = (batch["fut"][..., :2] - batch["hist"][:, :, -1:, :2]) / self.s_pos
        l = self.masked_mean(((self.forward_det(batch) - y) ** 2).sum(-1).mean(-1), batch["mask"])
        return l, dict(mse=l.item())


# ======================================================================= DGCN-Transformer
class DGCNTransformer(Deterministic):
    def __init__(self, cfg, stats, d=128, layers=2, heads=4, sigma_t=600.0):
        super().__init__(cfg, stats)
        self.sigma_t = sigma_t
        self.embed = nn.Linear(9, d)
        self.qk = nn.ModuleList(nn.Linear(d, 2 * d) for _ in range(2))
        self.mix = nn.Parameter(torch.zeros(2))
        self.gcn = nn.ModuleList(nn.Linear(d, d) for _ in range(2))
        self.register_buffer("pe", sinusoid(cfg.obs_len, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, cfg.dropout, batch_first=True)
        self.tf = nn.TransformerEncoder(layer, layers)
        self.dec = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, cfg.pred_len * 2))

    def risk_prior(self, hist, mask, L, B):
        """Collision-risk prior: DCPA relative to the receiver's QSD radius and TCPA (converging pairs only)."""
        pos, vel = hist[..., :2].transpose(1, 2), hist[..., 2:].transpose(1, 2)
        tcpa, dcpa = cpa(pos, vel)
        rel = pos[:, :, None] - pos[:, :, :, None]
        d_safe, _ = qsd.qsd_radius(rel, qsd.heading_unit(vel)[:, :, :, None].expand_as(rel),
                                   L[:, None, :, None], B[:, None, :, None])
        r = torch.exp(-torch.relu(dcpa - d_safe) / d_safe) * torch.exp(-tcpa / self.sigma_t)
        pm = qsd.pair_mask(mask)[:, None].expand_as(r)
        r = r * pm
        return r / r.sum(-1, keepdim=True).clamp_min(1e-6), pm

    def forward_det(self, batch):
        hist, mask = batch["hist"], batch["mask"]
        B, M, T, _ = hist.shape
        a_r, pm = self.risk_prior(hist, mask, *qsd.vessel_dims(self, batch))
        h = self.embed(state_features(hist, self.s_pos, self.s_vel, *qsd.vessel_dims(self, batch))).transpose(1, 2)
        for qk, lin, w in zip(self.qk, self.gcn, torch.sigmoid(self.mix)):
            q, k = qk(h).chunk(2, -1)
            att = torch.softmax((q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])).masked_fill(~pm, float("-inf")), -1)
            att = att.nan_to_num(0.0)
            h = h + F.relu(lin((w * a_r + (1 - w) * att) @ h))
        z = self.tf(h.permute(0, 2, 1, 3).reshape(B * M, T, -1) + self.pe)[:, -1]
        return self.out_to_disp(self.dec(z).view(B, M, self.cfg.pred_len, 2), batch)   # out_mode (fairness check)

    def loss(self, batch, progress):
        y = (batch["fut"][..., :2] - batch["hist"][:, :, -1:, :2]) / self.s_pos
        l = self.masked_mean(((self.forward_det(batch) - y) ** 2).sum(-1).mean(-1), batch["mask"])
        return l, dict(mse=l.item())


# ======================================================================= AISFormer
class FreqBlock(nn.Module):
    """Causal self-attention block with a frequency-aware branch: learnable filter on the rFFT of the
    (causally available) sequence, added before the feed-forward layer."""

    def __init__(self, d, heads, dropout, T):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.filt = nn.Parameter(torch.ones(T // 2 + 1, d))
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(dropout))
        self.T = T

    def forward(self, x):
        L = x.shape[1]
        m = torch.ones(L, L, dtype=torch.bool, device=x.device).triu(1)
        h = self.ln1(x)
        x = x + self.att(h, h, h, attn_mask=m)[0]
        # frequency branch, causal: for position t only x[:t+1] is used (prefix FFT via zero padding to T)
        h = self.ln2(x)
        pref = torch.stack([F.pad(h[:, :t + 1], (0, 0, 0, self.T - t - 1)) for t in range(L)], 1)   # [B, L, T, d]
        spec = torch.fft.rfft(pref, dim=2) * self.filt
        f = torch.fft.irfft(spec, n=self.T, dim=2)[:, torch.arange(L), torch.arange(L)]               # [B, L, d]
        x = x + f
        return x + self.ff(self.ln3(x))


class AISFormer(Deterministic):
    POS_RANGE, POS_BIN, POS_COARSE = 2500.0, 10.0, 50.0                           # metres
    SOG_MAX, SOG_BIN, COG_BIN = 30.0, 0.5, 5.0                                    # knots, knots, degrees

    def __init__(self, cfg, stats, d=256, layers=4, heads=8, lam_phys=0.1, lam_coarse=0.5):
        super().__init__(cfg, stats)
        self.np_ = int(2 * self.POS_RANGE / self.POS_BIN)
        self.nc_ = int(2 * self.POS_RANGE / self.POS_COARSE)
        self.ns, self.nq = int(self.SOG_MAX / self.SOG_BIN), int(360 / self.COG_BIN)
        self.lam_phys, self.lam_coarse = lam_phys, lam_coarse
        q = d // 4
        self.ex, self.ey = nn.Embedding(self.np_, q), nn.Embedding(self.np_, q)
        self.es, self.ec = nn.Embedding(self.ns, q), nn.Embedding(self.nq, q)
        self.pos = nn.Parameter(torch.zeros(cfg.obs_len + cfg.pred_len, d))
        T = cfg.obs_len + cfg.pred_len - 1
        self.blocks = nn.ModuleList(FreqBlock(d, heads, cfg.dropout, T) for _ in range(layers))
        self.lnf = nn.LayerNorm(d)
        self.hx, self.hy = nn.Linear(d, self.np_), nn.Linear(d, self.np_)
        self.hs, self.hc = nn.Linear(d, self.ns), nn.Linear(d, self.nq)
        self.hxc, self.hyc = nn.Linear(d, self.nc_), nn.Linear(d, self.nc_)

    # ---- discretisation (positions relative to the last observed position)
    def tokens(self, traj, p0):
        rel = traj[..., :2] - p0[:, None]
        ix = ((rel[..., 0] + self.POS_RANGE) / self.POS_BIN).long().clamp(0, self.np_ - 1)
        iy = ((rel[..., 1] + self.POS_RANGE) / self.POS_BIN).long().clamp(0, self.np_ - 1)
        sog = traj[..., 2:].norm(dim=-1) / KN
        cog = torch.rad2deg(torch.atan2(traj[..., 2], traj[..., 3])) % 360
        js = (sog / self.SOG_BIN).long().clamp(0, self.ns - 1)
        jc = (cog / self.COG_BIN).long().clamp(0, self.nq - 1)
        return ix, iy, js, jc

    def embed(self, ix, iy, js, jc):
        return torch.cat([self.ex(ix), self.ey(iy), self.es(js), self.ec(jc)], -1)

    def run(self, e):
        h = e + self.pos[:e.shape[1]]
        for b in self.blocks:
            h = b(h)
        return self.lnf(h)

    def centre(self, idx, bin_):
        return (idx.float() + 0.5) * bin_ - self.POS_RANGE

    def loss(self, batch, progress):
        mask = batch["mask"]
        traj = torch.cat([batch["hist"], batch["fut"]], 2)[mask]                  # [N, T, 4]
        p0 = batch["hist"][mask][:, -1, :2]
        ix, iy, js, jc = self.tokens(traj, p0)
        h = self.run(self.embed(ix, iy, js, jc)[:, :-1])                          # predict t+1 from <= t
        tgt = lambda a: a[:, 1:]
        ce = (F.cross_entropy(self.hx(h).transpose(1, 2), tgt(ix)) + F.cross_entropy(self.hy(h).transpose(1, 2), tgt(iy))
              + F.cross_entropy(self.hs(h).transpose(1, 2), tgt(js)) + F.cross_entropy(self.hc(h).transpose(1, 2), tgt(jc)))
        rel = traj[:, 1:, :2] - p0[:, None]
        cx = ((rel[..., 0] + self.POS_RANGE) / self.POS_COARSE).long().clamp(0, self.nc_ - 1)
        cy = ((rel[..., 1] + self.POS_RANGE) / self.POS_COARSE).long().clamp(0, self.nc_ - 1)
        coarse = F.cross_entropy(self.hxc(h).transpose(1, 2), cx) + F.cross_entropy(self.hyc(h).transpose(1, 2), cy)
        # physics-informed consistency: expected displacement vs expected speed/course of the step
        px = (F.softmax(self.hx(h), -1) * self.centre(torch.arange(self.np_, device=h.device), self.POS_BIN)).sum(-1)
        py = (F.softmax(self.hy(h), -1) * self.centre(torch.arange(self.np_, device=h.device), self.POS_BIN)).sum(-1)
        sog = (F.softmax(self.hs(h), -1) * ((torch.arange(self.ns, device=h.device) + 0.5) * self.SOG_BIN)).sum(-1) * KN
        ang = torch.deg2rad((torch.arange(self.nq, device=h.device) + 0.5) * self.COG_BIN)
        pc = F.softmax(self.hc(h), -1)
        u = torch.stack([(pc * torch.sin(ang)).sum(-1), (pc * torch.cos(ang)).sum(-1)], -1)
        prev = traj[:, :-1, :2] - p0[:, None]
        disp = torch.stack([px, py], -1) - prev
        phys = ((disp - sog[..., None] * u * self.cfg.dt) / self.s_pos).pow(2).sum(-1).mean()
        loss = ce + self.lam_coarse * coarse + self.lam_phys * phys
        return loss, dict(ce=ce.item(), coarse=coarse.item(), phys=phys.item())

    @torch.no_grad()
    def forward_det(self, batch):
        mask = batch["mask"]
        hist = batch["hist"][mask]
        p0 = hist[:, -1, :2]
        toks = list(self.tokens(hist, p0))
        out = []
        for _ in range(self.cfg.pred_len):
            h = self.run(self.embed(*toks))[:, -1]
            nx, ny = self.hx(h).argmax(-1), self.hy(h).argmax(-1)
            ns, nc = self.hs(h).argmax(-1), self.hc(h).argmax(-1)
            out.append(torch.stack([self.centre(nx, self.POS_BIN), self.centre(ny, self.POS_BIN)], -1))
            toks = [torch.cat([t, n[:, None]], 1) for t, n in zip(toks, (nx, ny, ns, nc))]
        B, M = mask.shape
        y = torch.zeros(B, M, self.cfg.pred_len, 2, device=hist.device)
        y[mask] = torch.stack(out, 1) / self.s_pos
        return y

    def configure_optimizers(self):
        return {"main": torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, betas=(0.9, 0.95), weight_decay=0.1)}
