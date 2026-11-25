"""Configuration of the RIGS-Net experiments (extends aistraj.config.Config; only the fields below are new)."""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass, fields, replace

from aistraj.config import Config

# Dense-traffic regions from full-rate DMA AIS, 2024-08-01 .. 2024-10-31 (tools/build_dma_regions.py).
DATA_ROOT = os.environ.get("RIGS_DATA_ROOT",
                           os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed"))
DATASETS = ["RWA", "ESB", "KIB", "GOT", "ORN"]
DATASET_LABELS = {"RWA": "Rostock-Warnem\u00fcnde", "ESB": "Esbjerg", "KIB": "Kiel Bight",
                  "GOT": "Gothenburg approach", "ORN": "\u00d8resund North"}


@dataclass
class RConfig(Config):
    # ---------------------------------------------------------------- RIGS-Net (manuscript Sec. 4, Table 2)
    d_model: int = 128             # [paper] d_h = d_r = 128
    enc_hidden: int = 64           # [paper] hidden width of E_n and E_r
    t_layers: int = 2              # [paper] Transformer layers
    t_heads: int = 4               # [paper]
    t_ff: int = 256                # [paper]
    alpha: float = 1.0             # [paper] weight of the learned gate in the edge logit
    beta: float = 0.5              # [paper] weight of the QSD ratio in the edge logit
    qsd_eta: float = 5.0           # [paper] sharpness of the analytical prior
    qsd_eps: float = 1e-6          # [paper]
    q_max: float = 10.0            # [impl.] upper clip of the QSD ratio q = d_safe / d (unbounded as d -> 0)
    L_default: float = 80.0        # [paper] default length (m); the AIS files carry no dimensions
    B_default: float = 15.0        # [paper] default beam (m)
    lambda_safe: float = 0.05      # [paper] QSD regularisation weight (selected per region on validation)
    nbr_radius: float = 0.0        # candidate neighbours within this distance (m) of the receiver; 0 = all vessels
    use_risk_enc: bool = True      # ablation: learned pairwise risk encoder + interaction gate
    use_qsd_edge: bool = True      # ablation: QSD radius / prior in the pair descriptor and the ratio q in the edge logit
    # ---------------------------------------------------------------- RIGS-Net v2 (2026-11-20; defaults = v1)
    msg_mode: str = "node"         # "node" (v1): m_i = sum_j A_ij h_j; "edge": m_i = sum_j A_ij phi([h_j, r_ij, z_ij])
    null_slot: bool = False        # learnable "no interaction" logit in the neighbour softmax (rows may sum to < 1)
    cpa_feat: bool = False         # constant-velocity CPA features (d_cpa, t_cpa) appended to the pair descriptor r_ij
    cpa_tmax: float = 600.0        # seconds; t_cpa is clipped to [0, cpa_tmax] and divided by it
    no_interaction: bool = False   # ablation: A = 0 (receiver sees no neighbour)
    out_mode: str = "direct"       # baselines: native output. RIGS-Net: "cv" (selected on validation, RIGS_OPT);
                                   # "cv": decoder outputs the offset from constant-velocity extrapolation
    # ---------------------------------------------------------------- shared training settings
    select_single: bool = True     # also keep the checkpoint with the best validation single-sample ADE
    select_metric: str = "ade"     # "ade" | "loss" (official S-STGCNN) | "ade_or_fde" (official MRGTraj)
    select_start: int = 0          # first epoch (1-based) eligible for checkpoint selection
    lr_step: int = 0               # StepLR step (epochs); 0 = constant lr
    lr_gamma: float = 0.1
    bucket: bool = False           # training batches contain scenes with the same number of vessels (no padding)

    def apply_cli(self, pairs) -> "RConfig":
        kw, names = {}, {f.name for f in fields(RConfig)}
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

    @staticmethod
    def from_dict(d: dict) -> "RConfig":
        names = {f.name for f in fields(RConfig)}
        return RConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names})


# RIGS-Net training settings stated in the manuscript (Sec. 5.2)
RIGS_OPT = dict(lr=3e-4, out_mode="cv", nbr_radius=3704.0, batch_size=64, grad_clip=5.0, epochs=100, patience=10, dropout=0.1)  # lr, out_mode: validation-selected

# Ablation rows of Table "ablation" (manuscript order). The full model uses the validation-selected lambda_safe.
VARIANTS = {
    "full": {},
    "noint": dict(no_interaction=True, use_risk_enc=False, use_qsd_edge=False, lambda_safe=0.0),
    "base":dict(use_risk_enc=False, use_qsd_edge=False, lambda_safe=0.0),
    "risk": dict(use_risk_enc=True, use_qsd_edge=False, lambda_safe=0.0),
    "risk_qsd": dict(use_risk_enc=True, use_qsd_edge=True, lambda_safe=0.0),
}
LAMBDAS = [0.0, 0.025, 0.05, 0.10, 0.20]

# Baselines: settings of the official code or of the papers; deviations are listed in the manuscript's baseline table.
BASELINE_OPT = {
    # official releases (third_party/), settings of their train scripts
    "sstgcnn": dict(lr=0.01, batch_size=128, epochs=250, patience=10,   # deviation: early stopping (official: 250 epochs, none)
                lr_step=150, lr_gamma=0.2,
                    select_metric="loss", bucket=True, select_single=False, grad_clip=10.0),
    "mrgtraj": dict(lr=1e-3, batch_size=128, epochs=100, patience=10,   # deviation: <=100 epochs + early stopping (official: 300)
                    select_metric="ade_or_fde",
                    select_start=11, select_single=False,
                    grad_clip=10.0),   # NOT official (no clipping): SGD diverges to NaN on dense regions (GOT, ORN); safeguard 7
    # official modules (Method/MSTFormer); paper settings (arXiv 2303.11540 Sec. IV), patience/max epochs of the script
    "mstformer": dict(lr=5e-6, batch_size=50, dropout=0.05, epochs=50, patience=5, lr_step=1, lr_gamma=0.5,
                      grad_clip=0.0),
    # implemented from the published descriptions (no code released); unspecified settings in the paper table
    "stmgcn": dict(lr=0.01, batch_size=128, epochs=250, patience=10,   # deviation: early stopping (official: 250 epochs, none)
                lr_step=150, lr_gamma=0.2,
                   select_metric="loss", bucket=True, select_single=False, grad_clip=10.0),   # Social-STGCNN framework
    "gstgan": dict(lr=1e-3, dropout=0.1, batch_size=64, grad_clip=5.0),
    "dgcntf": dict(lr=1e-3, dropout=0.1, batch_size=64, grad_clip=5.0),
    "aisformer": dict(lr=6e-4, dropout=0.1, batch_size=32, grad_clip=1.0, epochs=50),   # TrAISformer settings
}
for _v in BASELINE_OPT.values():
    _v.setdefault("epochs", 100)
    _v.setdefault("patience", 10)

MODEL_LABELS = {
    "cv": "Constant velocity",
    "sstgcnn": "S-STGCNN", "stmgcn": "STMGCN", "gstgan": "G-STGAN", "mrgtraj": "MRGTraj",
    "aisformer": "AISFormer", "mstformer": "MSTFormer", "dgcntf": "DGCN-Transformer", "rigs": "RIGS-Net",
}
STOCHASTIC = {"sstgcnn", "stmgcn", "mrgtraj"}
