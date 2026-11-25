"""Ports of the official implementations of Social-STGCNN and MRGTraj into the shared pipeline.

Sources (downloaded unchanged to third_party/):
  * Social-STGCNN: github.com/abduallahmohamed/Social-STGCNN (model.py, utils.py, metrics.py, train.py, train.sh)
  * MRGTraj:       github.com/wisionpeng/MRGTraj (model.py, mlp.py, transformer.py, train.py, utils.py)

The network modules are copied line by line (only device handling is changed). Adaptations, all
documented in the manuscript:
  1. Units. The official code consumes metric pedestrian coordinates (~0.4 m per step). AIS steps are
     ~10-40 m, so positions/displacements are divided by a dataset constant estimated on the training
     split (s_pos), and predictions are mapped back to metres before scoring.
  2. Batching. The official Social-STGCNN processes one scene per forward pass and accumulates the
     loss of 128 scenes before an update. Here 128 scenes with the same number of vessels are stacked
     in one forward pass (size-bucketed batches, no padding); the loss is the same average over scenes.
  3. MRGTraj social refiner. In the official code the refiner's attention mask is True for agents of
     the SAME scene and masked_fill(True, -1e9) therefore blocks exactly those agents, so every agent
     attends only to agents of other, unrelated scenes in the mini-batch. This contradicts the paper's
     description (interaction among the agents of a scene) and would make predictions depend on batch
     composition. We implement the paper's semantics: attention within the scene.
  4. MRGTraj inference draws one latent code per scene (official: one per mini-batch).
  5. Numerical safeguards (not in the official code; agreed for the RESS revision and disclosed): the
     bivariate correlation tanh(.) is clipped to [-CORR_MAX, CORR_MAX] in the loss and in sampling (at +-1 the
     covariance is singular and training diverges to NaN), and an optimisation step whose loss is not finite
     is skipped (counted in the training log as `skipped`).
  6. Single-output (K = 1) prediction: `sample_mean` returns the mean of the predictive distribution
     (bivariate-Gaussian mean for Social-STGCNN / STMGCN; for MRGTraj the average of 20 trajectories
     sampled with z ~ N(0, I), i.e. a Monte-Carlo estimate of the predictive mean -- z = 0 is out of distribution).
  7. MRGTraj gradient-norm clipping at 10 (not in the official code, applied in every region). The official loss
     divides the summed squared error by a constant (min(N, pred_len) * seq_len), so the gradient grows with the
     number of agents per mini-batch; with SGD lr 1e-3 and no clipping, training diverged to NaN within 5 steps
     on the dense regions GOT and ORN (~1200 agents per batch of 128 scenes vs ~640 in KIB).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from aistraj.models.base import TrajModel


# =============================================================== Social-STGCNN (official model.py)
class ConvTemporalGraphical(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, t_padding=0,
                 t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(t_kernel_size, 1), padding=(t_padding, 0),
                              stride=(t_stride, 1), dilation=(t_dilation, 1), bias=bias)

    def forward(self, x, A):                      # A [n, t, v, w] (official: [t, v, w] for one scene)
        x = self.conv(x)
        x = torch.einsum('nctv,ntvw->nctw', (x, A))
        return x.contiguous(), A


class st_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, use_mdn=False, stride=1, dropout=0, residual=True):
        super().__init__()
        assert len(kernel_size) == 2 and kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(nn.BatchNorm2d(out_channels), nn.PReLU(),
                                 nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
                                 nn.BatchNorm2d(out_channels), nn.Dropout(dropout, inplace=True))
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                                          nn.BatchNorm2d(out_channels))
        self.prelu = nn.PReLU()

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        if not self.use_mdn:
            x = self.prelu(x)
        return x, A


class social_stgcnn(nn.Module):
    def __init__(self, n_stgcnn=1, n_txpcnn=1, input_feat=2, output_feat=5, seq_len=8, pred_seq_len=12, kernel_size=3):
        super().__init__()
        self.n_stgcnn, self.n_txpcnn = n_stgcnn, n_txpcnn
        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(input_feat, output_feat, (kernel_size, seq_len)))
        for _ in range(1, self.n_stgcnn):
            self.st_gcns.append(st_gcn(output_feat, output_feat, (kernel_size, seq_len)))
        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len, pred_seq_len, 3, padding=1))
        for _ in range(1, self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1))
        self.tpcnn_ouput = nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1)
        self.prelus = nn.ModuleList(nn.PReLU() for _ in range(self.n_txpcnn))

    def forward(self, v, a):
        for k in range(self.n_stgcnn):
            v, a = self.st_gcns[k](v, a)
        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])
        v = self.prelus[0](self.tpcnns[0](v))
        for k in range(1, self.n_txpcnn - 1):
            v = self.prelus[k](self.tpcnns[k](v)) + v
        v = self.tpcnn_ouput(v)
        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])
        return v, a


def stgcnn_graph(V, mask):
    """Official utils.seq_to_graph: A_hk = 1/||v_h - v_k|| of the per-step relative displacements (0 if equal),
    A_hh = 1, then networkx normalized Laplacian L = I - D^-1/2 A D^-1/2 (degrees include the self-loop;
    isolated padded nodes give a zero row). V [n, T, N, 2], mask [n, N] -> [n, T, N, N]."""
    N = V.shape[2]
    d = (V[:, :, :, None] - V[:, :, None, :]).norm(dim=-1)
    A = torch.where(d > 0, 1.0 / d.clamp_min(1e-12), torch.zeros_like(d))
    eye = torch.eye(N, dtype=torch.bool, device=V.device)
    A = torch.where(eye, torch.ones_like(A), A)
    pm = (mask[:, None, :, None] & mask[:, None, None, :]).float()
    A = A * pm
    deg = A.sum(-1)
    dinv = torch.where(deg > 0, deg.clamp_min(1e-12).rsqrt(), torch.zeros_like(deg))
    L = torch.eye(N, device=V.device) * (deg > 0).float()[..., None] - dinv[..., :, None] * A * dinv[..., None, :]
    return L


CORR_MAX = 0.99


def bivariate_corr(x):
    """tanh correlation of the official code, clipped to [-CORR_MAX, CORR_MAX] (safeguard 5)."""
    return torch.tanh(x).clamp(-CORR_MAX, CORR_MAX)


def safe_step(loss, opt, params, clip=0.0):
    """Backward + step; the step is skipped when the loss is not finite (safeguard 5). Returns 1 if skipped."""
    opt.zero_grad(set_to_none=True)
    if not torch.isfinite(loss):
        return 1
    loss.backward()
    if clip > 0:
        nn.utils.clip_grad_norm_(params, clip)
    if not all(torch.isfinite(p.grad).all() for p in params if p.grad is not None):
        opt.zero_grad(set_to_none=True)
        return 1
    opt.step()
    return 0


def bivariate_nll(V_pred, V_trgt):
    """Official metrics.bivariate_loss without the final mean: -log pdf per element, [..]."""
    normx = V_trgt[..., 0] - V_pred[..., 0]
    normy = V_trgt[..., 1] - V_pred[..., 1]
    sx, sy, corr = torch.exp(V_pred[..., 2]), torch.exp(V_pred[..., 3]), bivariate_corr(V_pred[..., 4])
    sxsy = sx * sy
    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negRho = 1 - corr ** 2
    result = torch.exp(-z / (2 * negRho)) / (2 * np.pi * (sxsy * torch.sqrt(negRho)))
    return -torch.log(torch.clamp(result, min=1e-20))


class SocialSTGCNNOfficial(TrajModel):
    """train.sh: n_stgcnn 1, n_txpcnn 5, kernel 3, output 5; SGD lr 0.01, batch 128 scenes, 250 epochs,
    StepLR(150, 0.2); checkpoint with the lowest validation loss; 20 samples from the bivariate Gaussian."""
    multimodal = True
    supports_base_cvr = False

    def __init__(self, cfg, stats):
        super().__init__(cfg, stats)
        self.net = social_stgcnn(n_stgcnn=1, n_txpcnn=5, input_feat=2, output_feat=5,
                                 seq_len=cfg.obs_len, pred_seq_len=cfg.pred_len, kernel_size=3)

    def _rel(self, traj, p_prev):
        pos = traj[..., :2]
        prev = torch.cat([p_prev[:, :, None], pos[:, :, :-1]], dim=2)
        return (pos - prev) / self.s_pos                                          # [B, M, T, 2]

    def params(self, batch):
        hist, mask = batch["hist"], batch["mask"]
        V = self._rel(hist, hist[:, :, 0, :2]).transpose(1, 2)                   # first step 0 (official)
        A = stgcnn_graph(V, mask)
        v = V.permute(0, 3, 1, 2)                                                 # [n, 2, T, N]
        out, _ = self.net(v, A)                                                   # [n, 5, T_p, N]
        return out.permute(0, 3, 2, 1)                                            # [n, N, T_p, 5]

    def loss(self, batch, progress):
        y = self._rel(batch["fut"], batch["hist"][:, :, -1, :2])
        nll = bivariate_nll(self.params(batch), y).mean(-1)                       # [B, M]
        l = self.masked_mean(nll, batch["mask"])
        return l, dict(nll=l.item())

    def sample(self, batch, K):
        p = self.params(batch)
        mean = p[..., :2]
        sx, sy, corr = torch.exp(p[..., 2]), torch.exp(p[..., 3]), bivariate_corr(p[..., 4])
        cov = torch.stack([torch.stack([sx * sx, corr * sx * sy], -1), torch.stack([corr * sx * sy, sy * sy], -1)], -2)
        mvn = torch.distributions.MultivariateNormal(mean, cov)
        s = mvn.sample((K,)).transpose(0, 1)                                      # [B, K, M, T_p, 2]
        return s.cumsum(dim=3)

    def sample_mean(self, batch):
        """K = 1 prediction: mean of the bivariate Gaussian at every step, integrated. [B, 1, M, T_p, 2]"""
        return self.params(batch)[..., :2].cumsum(dim=2)[:, None]

    def configure_optimizers(self):
        return {"main": torch.optim.SGD(self.parameters(), lr=self.cfg.lr)}

    def training_step(self, batch, opts, progress):
        loss, logs = self.loss(batch, progress)
        logs["skipped"] = safe_step(loss, opts["main"], list(self.parameters()), self.cfg.grad_clip)  # official --clip_grad
        logs["loss"] = float(loss.detach())
        return logs


# =============================================================== MRGTraj (official transformer.py / mlp.py / model.py)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


def scaled_dot_attention(Q, K, V, attn_mask=None, dropout=None):
    scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Q.shape[-1])
    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask, -1e9)
    attn = nn.Softmax(dim=-1)(scores)
    if dropout is not None:
        attn = dropout(attn)
    return torch.matmul(attn, V), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.d_model, self.d_k = n_heads, d_model, d_model // n_heads
        self.W_Q = nn.Parameter(torch.Tensor(n_heads, d_model, self.d_k))
        self.W_K = nn.Parameter(torch.Tensor(n_heads, d_model, self.d_k))
        self.W_V = nn.Parameter(torch.Tensor(n_heads, d_model, self.d_k))
        self.fc = nn.Linear(self.d_k * n_heads, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        for w in (self.W_Q, self.W_K, self.W_V):
            nn.init.xavier_uniform_(w, gain=1.414)

    def forward(self, input_Q, input_K, input_V, attn_mask=None):
        residual, bs = input_Q, input_Q.size(0)
        Q = torch.matmul(input_Q.unsqueeze(1), self.W_Q)
        K = torch.matmul(input_K.unsqueeze(1), self.W_K)
        V = torch.matmul(input_V.unsqueeze(1), self.W_V)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = scaled_dot_attention(Q, K, V, attn_mask=attn_mask, dropout=self.dropout)
        context = context.transpose(1, 2).reshape(bs, -1, self.n_heads * self.d_k)
        # official: nn.LayerNorm(d_model).cuda()(...) created afresh each call = parameter-free layer norm
        return nn.functional.layer_norm(self.fc(context) + residual, (self.d_model,)), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff=1024):
        super().__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(nn.Linear(d_model, d_ff, bias=False), nn.ReLU(), nn.Linear(d_ff, d_model, bias=False))

    def forward(self, inputs):
        return nn.functional.layer_norm(self.fc(inputs) + inputs, (self.d_model,))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model)

    def forward(self, x, mask=None):
        x, attn = self.enc_self_attn(x, x, x, attn_mask=mask)
        return self.pos_ffn(x), attn


class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, n_heads, max_len=20, if_emb=True, if_pos=True):
        super().__init__()
        self.if_emb, self.if_pos = if_emb, if_pos
        if if_emb:
            self.src_emb = nn.Linear(input_dim, d_model)
            self.d_model = d_model
        else:
            self.d_model = input_dim
        if if_pos:
            self.pos_emb = PositionalEncoding(self.d_model, max_len=max_len)
        self.layers = nn.ModuleList([EncoderLayer(self.d_model, n_heads) for _ in range(n_layers)])

    def forward(self, x, mask=None):
        if self.if_emb:
            x = self.src_emb(x)
        if self.if_pos:
            x = self.pos_emb(x)
        for layer in self.layers:
            x, _ = layer(x, mask)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512)):
        super().__init__()
        dims = [input_dim, *hidden_size, output_dim]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.activation = nn.ReLU()

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
        return x


class SocialLatentGenerator(nn.Module):
    def __init__(self, dim_in=4, d_model=256, dim_z=16, dff=1024, dropout=0.3):
        super().__init__()
        self.dim_z, self.d_model = dim_z, d_model
        self.emb_layer = nn.Sequential(nn.Linear(dim_in, d_model), nn.ReLU(), nn.Dropout(dropout))
        self.latent = nn.Sequential(nn.Linear(d_model, dff), nn.ReLU6(), nn.Dropout(dropout),
                                    nn.Linear(dff, dff), nn.ReLU6(), nn.Dropout(dropout),
                                    nn.Linear(dff, dim_z * 2), nn.ReLU6())

    def forward(self, inp, scene_mask):
        """inp [N, T_p, 4] (all agents of the mini-batch), scene_mask [N, N] (1 = same scene).
        Official: per agent, max over the agents of its scene, then mean over time."""
        e = self.emb_layer(inp)                                                   # [N, T, d], >= 0 (ReLU)
        # official: (e_j * same_scene_ij).max over j. Since e >= 0, the zeros of other scenes never win, so this
        # is the max over the agents of the own scene; computed per scene to avoid the [T, N, N, d] tensor.
        scene = scene_mask.float().argmax(1)                                      # first agent of the scene = id
        idx = scene[:, None, None].expand_as(e)
        smax = torch.zeros_like(e).scatter_reduce(0, idx, e, reduce="amax", include_self=False)
        lat = self.latent(smax[scene].mean(1))
        mu, log_var = lat[:, :self.dim_z], lat[:, self.dim_z:]
        z = torch.randn_like(mu) * log_var.mul(0.5).exp() + mu
        return z, mu, log_var


class MRGTrajOfficial(TrajModel):
    """train.py defaults: d_model 256, 4 heads, 1 layer, noise 64, SGD lr 1e-3, batch 128, 300 epochs,
    validation (best-of-20) from epoch 11 on, checkpoint when ADE or FDE improves."""
    multimodal = True
    supports_base_cvr = False

    def __init__(self, cfg, stats, d_model=256, n_layers=1, n_heads=4, noise_dim=64):
        super().__init__(cfg, stats)
        self.pred_len, self.noise_dim = cfg.pred_len, noise_dim
        self.past_encoder = Encoder(2, d_model, n_layers, n_heads)
        self.social_latent_generator = SocialLatentGenerator(dim_in=4, d_model=d_model, dim_z=noise_dim)
        self.temporal_mapper = MLP(cfg.obs_len, cfg.pred_len, hidden_size=(512, 1024, 512))
        self.social_refiner = Encoder(d_model + noise_dim, d_model, n_layers, n_heads, if_emb=False, if_pos=False)
        self.prediction_layer = nn.Linear(d_model + noise_dim, 2)

    # ---- flatten padded scenes into the official agent-major layout
    def _flat(self, batch):
        hist, fut, mask = batch["hist"], batch["fut"], batch["mask"]
        B, M = mask.shape
        p, m = hist[:, :, -1, :2], mask.float()[..., None]
        c = (p * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1.0)  # scene centroid
        scene = torch.arange(B, device=mask.device)[:, None].expand(B, M)[mask]
        def absrel(traj, prev):
            pos = traj[..., :2]
            rel = pos - torch.cat([prev[:, :, None], pos[:, :, :-1]], 2)
            return torch.cat([(pos - c[:, :, None]) / self.s_pos, rel / self.s_pos], -1)[mask]
        past = absrel(hist, hist[:, :, 0, :2])
        future = absrel(fut, hist[:, :, -1, :2])
        same = (scene[:, None] == scene[None, :])
        return past, future, same, scene

    def _decode(self, past, z, same):
        enc = self.past_encoder(past[..., 2:])
        dec = self.temporal_mapper(enc.transpose(-1, -2)).transpose(-1, -2).transpose(0, 1)   # [T_p, N, d]
        dec = torch.cat((dec, z.unsqueeze(0).repeat(self.pred_len, 1, 1)), dim=-1)
        dec = self.social_refiner(dec, (~same).unsqueeze(0).expand(self.pred_len, -1, -1))    # attend within scene
        return self.prediction_layer(dec)                                         # [T_p, N, 2] per-step rel

    def loss(self, batch, progress):
        past, future, same, _ = self._flat(batch)
        z, mu, log_var = self.social_latent_generator(future, same.float())
        pred = self._decode(past, z, same)
        tgt = future[..., 2:].transpose(0, 1)
        N = past.shape[0]
        l2 = ((tgt - pred) ** 2).sum() / (min(N, self.pred_len) * (self.cfg.obs_len + self.pred_len))  # official normaliser
        kld = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())).mean()
        return l2 + kld, dict(l2=l2.item(), kld=kld.item())

    def sample(self, batch, K):
        past, _, same, scene = self._flat(batch)
        B, M = batch["mask"].shape
        outs = []
        for _ in range(K):
            zs = torch.randn(B, self.noise_dim, device=past.device)[scene]          # one code per scene
            rel = self._decode(past, zs, same).transpose(0, 1).cumsum(1)         # [N, T_p, 2]
            y = torch.zeros(B, M, self.pred_len, 2, device=past.device)
            y[batch["mask"]] = rel
            outs.append(y)
        return torch.stack(outs, 1)

    MEAN_SAMPLES = 20

    def sample_mean(self, batch):
        """K = 1 prediction: Monte-Carlo predictive mean, the average of MEAN_SAMPLES trajectories decoded from
        z ~ N(0, I). (z = 0 is not used: the posterior collapses to the prior, so the decoder only sees codes with
        |z| ~ sqrt(64) = 8 and z = 0 is out of distribution -- test ADE 175-500 m vs 18-25 m per random sample.)
        Calls the class method explicitly because `score` temporarily shadows `self.sample`. [B, 1, M, T_p, 2]"""
        return type(self).sample(self, batch, self.MEAN_SAMPLES).mean(1, keepdim=True)

    def configure_optimizers(self):
        return {"main": torch.optim.SGD(self.parameters(), lr=self.cfg.lr)}

    def training_step(self, batch, opts, progress):
        loss, logs = self.loss(batch, progress)
        logs["skipped"] = safe_step(loss, opts["main"], list(self.parameters()), self.cfg.grad_clip)  # safeguard 7
        logs["loss"] = float(loss.detach())
        return logs
