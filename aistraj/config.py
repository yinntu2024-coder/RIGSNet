"""Shared data/training configuration (AIS scenes, chronological split, optimisation defaults).

RIGS-Net extends this dataclass in rigs/config.py; the experiments override most fields there and in jobs/*.json.
"""
from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, fields, replace
from typing import Tuple

NMI = 1852.0          # metres per nautical mile
KNOT = 0.514444       # m/s per knot


@dataclass
class Config:
    # ------------------------------------------------------------------ data (Sec. 5.1)
    obs_len: int = 20                 # [paper] 20 frames = 100 s
    pred_len: int = 12                # [paper] 12 frames = 60 s
    dt: float = 5.0                   # [paper] uniform 5 s resampling
    split: Tuple[float, float, float] = (0.70, 0.15, 0.15)   # [paper] chronological
    window_stride: int = 12           # [assumed] sliding-window stride in frames (60 s = T_f: future segments do not overlap)
    min_vessels: int = 1              # [assumed] minimum vessels for a scene to be kept
    max_vessels: int = 0              # [assumed] 0 = no cap on scene size
    # pre-processing protocol
    sog_max_kn: float = 35.0          # [paper]
    implied_speed_max_kn: float = 45.0  # [paper]
    stationary_sog_kn: float = 0.5    # [paper]
    stationary_min_s: float = 300.0   # [paper] longer than 5 min
    max_gap_s: float = 60.0           # [paper] split when gap > 60 s

    # ------------------------------------------------------------------ model (Sec. 4, 5.2)
    in_dim: int = 4                   # [assumed] (x, y, vx, vy) built from lon/lat/SOG/COG
    hidden: int = 64                  # [paper] hidden size 64
    ssm_layers: int = 2               # [paper] two selective SSM layers
    n_heads: int = 8                  # [paper] 8 heads in fusion and conditioning modules
    dropout: float = 0.1              # [paper]
    stgcn_layers: int = 2             # [paper] two-layer STGCN backbone
    stgcn_kernel: int = 3             # [assumed] temporal kernel size
    latent_dim: int = 32              # [assumed] dimension of z_m
    ssm_d_state: int = 16             # [assumed] Mamba defaults
    ssm_d_conv: int = 4               # [assumed]
    ssm_expand: int = 2               # [assumed]
    ssm_backend: str = "torch"        # "torch" (pure PyTorch, runs anywhere) | "mamba_ssm" (CUDA kernel)
    share_ssm_enc: bool = True        # [assumed] "the same selective state-space backbone" for X and Y
    temporal_backbone: str = "ssm"    # "ssm" | "transformer" (comparison with Transformer-based encoders/decoders)
    K: int = 20                       # [paper] samples at inference
    # graph construction
    fov_range_nmi: float = 6.0        # [paper] Upsilon
    sigma_d_nmi: float = 1.0          # [paper]
    sigma_t_s: float = 600.0          # [paper]
    eta: float = 0.5                  # [paper] risk-temperature coefficient
    risk_topk: int = 3                # [assumed] K_r in Eq. (risk_score)
    social_eps: float = 1e-3          # [assumed] epsilon in Eq. (adj_s), in units of social_unit_m
    social_unit_m: float = NMI        # [assumed] distance unit used inside Eq. (adj_s)
    cpa_eps: float = 1e-6             # [assumed] epsilon in Eq. (dcpa), (m/s)^2
    risk_graph_form: str = "cpa_exp"  # "cpa_exp" (Eq. adj_cr) | "inv_tcpa" (A = 1/TCPA, the form used by CIE-MGCN)

    # ------------------------------------------------------------------ ablation switches (Sec. 5.5)
    use_ricm: bool = True             # Risk-aware Interaction Context Modeling
    use_intention: bool = True        # Risk-conditioned Intention Inference (CVAE)
    decoder_type: str = "ssm"         # "ssm" | "mlp" | "xattn" (Transformer decoder attending to H_X, direct positions)
    decoder_use_context: bool = True  # condition the decoder on f_m
    use_multi_relation: bool = True   # False -> every branch receives one uniform fully-connected graph
    fusion: str = "cgaf"              # "cgaf" | "sum" (fixed equal-weight sum) | "mmt" (CIE-MGCN multimodal transformer)
    use_pga: bool = True              # past-guided attention in the posterior encoder
    cvae_use_context: bool = True     # condition prior and posterior on the fused context f_m
    graphs: Tuple[str, ...] = ("fv", "s", "cr")
    risk_temperature: bool = True     # risk-adaptive sampling temperature, Eq. (tg3)

    # ------------------------------------------------------------------ optimisation (Sec. 5.2, Alg. 1)
    lr: float = 1e-4                  # [paper]
    epochs: int = 100                 # [paper] "up to 100 epochs"
    batch_size: int = 32              # [assumed]
    weight_decay: float = 0.0         # [assumed]
    grad_clip: float = 1.0            # [assumed]
    patience: int = 15                # [assumed] early stopping on validation ADE_K
    max_train_hours: float = 0.0      # [assumed] stop training after the epoch that exceeds this wall-clock budget; 0 = off
    # sensitivity runs only (defaults keep every earlier run unchanged):
    train_window_stride: int = 0      # >0: training windows use this stride; the split boundaries and val/test windows
                                      #     still come from window_stride, so no window crosses into val/test
    test_window_stride: int = 0       # >0: test windows use this stride, starting at the window_stride test boundary
    max_train_hours_in_epoch: bool = False  # True: the time budget is also checked inside an epoch (partial last epoch,
                                            #       then validation and stop)
    log_every_min: float = 0.0        # >0: progress line every this many minutes inside an epoch
    eval_test_each_epoch: bool = False  # (2026-11-05) also score the test split after every epoch, for monitoring
                                        # only: checkpoint selection and early stopping stay on validation ADE_K
    epoch_md: str = ""                # markdown file that receives one row per epoch (val + test), appended under a lock
    split_file: str = ""              # (2026-11-05) name of <data_root>/<DS>/<name>.npz from tools/make_difficulty_split.py:
                                      #     explicit window starts (stride-1 grid) per split instead of the chronological
                                      #     cut; window_stride / test_window_stride are then ignored
    test_subset: str = ""             # (2026-11-05) name of <data_root>/<DS>/<name>.npz from tools/make_hard_test.py:
                                      #     only the listed test trajectories are scored (scenes keep all vessels as
                                      #     context); train/val untouched. "" = whole test split
    lambda_mse: float = 1.0           # [assumed] lambda_M
    lambda_kl: float = 1.0            # [assumed] lambda_KL (final value after annealing)
    lambda_var: float = 1.0           # [assumed] lambda_V
    kl_anneal_epochs: int = 10        # [assumed] linear schedule (annealing itself is [paper], Sec. 1)
    kl_free_bits: float = 0.0         # [assumed] free bits (nats per latent dimension) against posterior collapse; 0 = off
    lr_schedule: str = "none"         # [assumed] "none" | "plateau" (x0.5 after 5 epochs without val improvement) | "cosine"
    decoder_cv_residual: bool = False # [assumed] decoder predicts the deviation from constant-velocity extrapolation
    baseline_cv_residual: bool = False  # fairness check (2026-11-04): baselines predict the deviation from CV extrapolation
                                      # (target - CV in training, + CV at prediction); off = original baselines
    variety_mode: str = "vessel"      # [assumed] "vessel" (text below Eq. var) | "scene" (Eq. var literally)
    risk_temp_train: bool = True      # [assumed] variety samples from the risk-calibrated prior (Sec. 4.5)
    train_K: int = 20                 # [assumed] K used for the variety term (Alg. 1 uses K)
    num_workers: int = 0
    seed: int = 0

    # ------------------------------------------------------------------ helpers
    @property
    def seq_len(self) -> int:
        return self.obs_len + self.pred_len

    def to_dict(self) -> dict:
        return asdict(self)

    def override(self, **kw) -> "Config":
        return replace(self, **kw)

    @staticmethod
    def from_dict(d: dict) -> "Config":
        names = {f.name for f in fields(Config)}
        d = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names}
        return Config(**d)

    def apply_cli(self, pairs) -> "Config":
        """pairs: iterable of 'key=value' strings (python literals where possible)."""
        kw = {}
        names = {f.name for f in fields(Config)}
        for p in pairs or []:
            k, v = p.split("=", 1)
            if k not in names:
                raise KeyError(f"unknown config key: {k}")
            try:
                v = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                pass
            kw[k] = tuple(v) if isinstance(v, list) else v
        return replace(self, **kw)

    def dump(self, path):
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
