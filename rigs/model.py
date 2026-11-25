"""RIGS-Net (manuscript Sec. 4): risk-informed directed graph -> serial spatial/temporal encoding ->
one-shot deterministic decoder; differentiable QSD regularisation during training.

Shapes: B scenes, M vessels (padded), T_o observed frames, T_p predicted frames.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from aistraj.models.base import TrajModel

from . import qsd

R0 = 1000.0       # metres; scale of pairwise distances in the pair descriptor


def mlp2(d_in, d_hid, d_out, act=nn.ReLU):
    return nn.Sequential(nn.Linear(d_in, d_hid), act(), nn.Linear(d_hid, d_out))


def sinusoid(T, d):
    pe = torch.zeros(T, d)
    pos = torch.arange(T, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


def wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


class Deterministic(TrajModel):
    """Single-output model; `forward_det` returns normalised cumulative displacements [B, M, T_p, 2]."""
    multimodal = False
    supports_base_cvr = False

    def out_to_disp(self, out, batch):
        if getattr(self.cfg, "out_mode", "direct") == "cv":
            return out + self.cv_disp(batch)
        return out

    def sample(self, batch, K):
        return self.forward_det(batch)[:, None]

    def pred_loss(self, y_hat, batch):
        """L_pred: mean Euclidean displacement error (normalised units), averaged over valid vessels."""
        y = (batch["fut"][..., :2] - batch["hist"][:, :, -1:, :2]) / self.s_pos
        e = (y_hat - y).norm(dim=-1).mean(-1)                                     # [B, M]
        return self.masked_mean(e, batch["mask"])

    def loss(self, batch, progress):
        l = self.pred_loss(self.forward_det(batch), batch)
        return l, dict(pred=l.item())


def state_features(hist, s_pos, s_vel, L, B):
    """x_i^t = [position relative to the last observed position, velocity, speed, course (sin, cos), L, B].
    L, B: scalars or per-vessel tensors [B, M]."""
    p_last = hist[:, :, -1:, :2]
    v = hist[..., 2:]
    spd = v.norm(dim=-1, keepdim=True)
    psi = torch.atan2(v[..., :1], v[..., 1:2])                                   # clockwise from north
    size = torch.ones_like(spd)
    if torch.is_tensor(L):
        L, B = L[:, :, None, None], B[:, :, None, None]
    return torch.cat([(hist[..., :2] - p_last) / s_pos, v / s_vel, spd / s_vel, torch.sin(psi), torch.cos(psi),
                      size * (L / 100.0), size * (B / 20.0)], dim=-1)            # [B, M, T, 9]


class RIGSNet(Deterministic):
    def __init__(self, cfg, stats):
        super().__init__(cfg, stats)
        d, h = cfg.d_model, cfg.enc_hidden
        self.E_n = mlp2(9, h, d)                                                  # vessel-state encoder
        n_rel = (6 if cfg.use_qsd_edge else 5) + (2 if cfg.cpa_feat else 0)     # r_ij (d_safe only with QSD)
        n_u = n_rel + (1 if cfg.use_qsd_edge else 0)                             # u_ij = [r_ij, rho_bar]
        if cfg.msg_mode == "edge":                                               # v2: edge-conditioned messages
            self.Phi = mlp2(d + n_rel + (d if cfg.use_risk_enc else 0), h, d)
        elif cfg.msg_mode != "node":
            raise ValueError(f"msg_mode={cfg.msg_mode}")
        if cfg.null_slot:
            self.null_logit = nn.Parameter(torch.zeros(()))
        if cfg.use_risk_enc:
            self.E_r = mlp2(n_u, h, d)                                            # risk encoder
            self.P_g = nn.Linear(d, 1)                                            # interaction-gate projection
            self.P_e = nn.Linear(n_rel + d, 1)                                    # edge projection
        else:
            self.P_e = nn.Linear(n_rel, 1)                                        # learned affinity from r_ij only
        self.S = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d))   # spatial interaction encoder
        self.norm = nn.LayerNorm(d)
        self.register_buffer("pe", sinusoid(cfg.obs_len, d))
        layer = nn.TransformerEncoderLayer(d, cfg.t_heads, cfg.t_ff, cfg.dropout, batch_first=True)
        self.T = nn.TransformerEncoder(layer, cfg.t_layers)                       # temporal encoder
        self.D = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, cfg.pred_len * 2))   # decoder

    # ------------------------------------------------------------------ graph
    def graph(self, hist, mask, L, B):
        """Directed adjacency A [B, T, M, M] (row i = receiver), analytical prior rho_bar [B, T, M, M],
        learned gate rho [B, T, M, M] (None without the risk encoder)."""
        cfg = self.cfg
        pos = hist[..., :2].transpose(1, 2)                                        # [B, T, M, 2]
        vel = hist[..., 2:].transpose(1, 2)
        spd = vel.norm(dim=-1)
        psi = torch.atan2(vel[..., 0], vel[..., 1])
        fwd = qsd.heading_unit(vel)
        rel = pos[:, :, None, :, :] - pos[:, :, :, None, :]                        # [B, T, i, j, 2] = p_j - p_i
        d_safe, d = qsd.qsd_radius(rel, fwd[:, :, :, None].expand_as(rel), L[:, None, :, None], B[:, None, :, None])
        ds = (spd[:, :, None, :] - spd[:, :, :, None]) / self.s_vel
        dpsi = wrap(psi[:, :, None, :] - psi[:, :, :, None]) / math.pi
        r = [rel[..., 0] / R0, rel[..., 1] / R0, ds, dpsi, d / R0]
        rho_bar = qsd.prior(d_safe, d, cfg.qsd_eta, cfg.qsd_eps)
        if cfg.use_qsd_edge:
            r.append(d_safe / R0)
        if cfg.cpa_feat:                                                           # constant-velocity closest approach
            vrel = vel[:, :, None, :, :] - vel[:, :, :, None, :]                   # v_j - v_i
            t_cpa = (-(rel * vrel).sum(-1) / ((vrel * vrel).sum(-1) + 1e-6)).clamp(0.0, cfg.cpa_tmax)
            d_cpa = (rel + vrel * t_cpa[..., None]).norm(dim=-1)
            r += [d_cpa / R0, t_cpa / cfg.cpa_tmax]
        r = torch.stack(r, dim=-1)                                                 # [B, T, M, M, n_rel]
        rho, z = None, None
        if cfg.use_risk_enc:
            u = torch.cat([r, rho_bar[..., None]], -1) if cfg.use_qsd_edge else r
            z = self.E_r(u)
            rho = torch.sigmoid(self.P_g(z)).squeeze(-1)
            e = self.P_e(torch.cat([r, z], -1)).squeeze(-1) + cfg.alpha * rho
        else:
            e = self.P_e(r).squeeze(-1)
        if cfg.use_qsd_edge:
            q = (d_safe / (d + cfg.qsd_eps)).clamp(max=cfg.q_max)
            e = e + cfg.beta * q
        pm = qsd.pair_mask(mask)[:, None].expand_as(e)                             # [B, T, M, M]
        if cfg.nbr_radius > 0:                                                     # candidate set N_t(i)
            pm = pm & (d <= cfg.nbr_radius)
        e = e.masked_fill(~pm, float("-inf"))
        if cfg.null_slot:                                                          # extra "attend to nobody" column
            e = torch.cat([e, self.null_logit.expand(e.shape[:-1])[..., None]], -1)
            A = torch.softmax(e, dim=-1)[..., :-1]
        else:
            A = torch.softmax(e, dim=-1)
        A = torch.nan_to_num(A, nan=0.0) * pm.float()                              # receivers without neighbours
        if cfg.no_interaction:
            A = torch.zeros_like(A)
        self._edge = (r, z)                                                        # pair features for edge messages
        return A, rho_bar, rho

    # ------------------------------------------------------------------ forward
    def encode(self, batch):
        hist, mask = batch["hist"], batch["mask"]
        B, M, To, _ = hist.shape
        L, Bm = qsd.vessel_dims(self, batch)
        A, rho_bar, rho = self.graph(hist, mask, L, Bm)
        h = self.E_n(state_features(hist, self.s_pos, self.s_vel, L, Bm))
        h = h.transpose(1, 2)                                                      # [B, T, M, d]
        if self.cfg.msg_mode == "edge" and not self.cfg.no_interaction:
            r, zr = self._edge
            hj = h[:, :, None].expand(-1, -1, M, -1, -1)                           # [B, T, i, j, d] sender states
            feat = [hj, r] + ([zr] if zr is not None else [])
            m = (A[..., None] * self.Phi(torch.cat(feat, -1))).sum(-2)             # receiver-relative messages
        else:
            m = A @ h                                                              # directed messages
        ht = self.norm(h + self.S(torch.cat([h, m], -1)))                          # [B, T, M, d]
        seq = ht.permute(0, 2, 1, 3).reshape(B * M, To, -1) + self.pe
        z = self.T(seq)[:, -1]                                                     # sequence readout
        return z.view(B, M, -1), A, rho_bar, rho

    def forward_det(self, batch):
        z, *_ = self.encode(batch)
        B, M, _ = z.shape
        out = self.D(z).view(B, M, self.cfg.pred_len, 2)
        return self.out_to_disp(out, batch)

    def loss(self, batch, progress):
        cfg = self.cfg
        hist, mask = batch["hist"], batch["mask"]
        z, A, rho_bar, _ = self.encode(batch)
        B, M, _ = z.shape
        y_hat = self.out_to_disp(self.D(z).view(B, M, cfg.pred_len, 2), batch)
        l_pred = self.pred_loss(y_hat, batch)
        logs = dict(pred=l_pred.item())
        loss = l_pred
        if cfg.lambda_safe > 0:
            p_last = hist[:, :, -1, :2]
            pred = y_hat * self.s_pos + p_last[:, :, None]
            f_last = qsd.heading_unit(hist[:, :, -1, 2:])
            L, Bm = qsd.vessel_dims(self, batch)
            l_safe = qsd.safe_penalty(pred, p_last, f_last, mask, rho_bar[:, -1], L, Bm,
                                      float(self.s_pos))
            loss = loss + cfg.lambda_safe * l_safe
            logs["safe"] = l_safe.item()
        return loss, logs

    def configure_optimizers(self):
        return {"main": torch.optim.Adam(self.parameters(), lr=self.cfg.lr)}
