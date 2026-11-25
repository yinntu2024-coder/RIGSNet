# RIGS-Net: Risk-Informed Graph Learning for Safety-Consistent Multi-Vessel Trajectory Prediction

Code for the experiments of the manuscript *Risk-Informed Graph Learning for Safety-Consistent Multi-Vessel Trajectory Prediction* (Reliability Engineering & System Safety, under review).

RIGS-Net predicts 12 future positions (60 s) of every vessel in a scene from 20 observed AIS states (100 s, 5 s sampling). It uses a directed, receiver-conditioned interaction graph that combines relative kinematics, a learned interaction gate and the Quaternion Ship Domain (QSD), and it is trained with an optional differentiable QSD regularizer.

## Repository layout

```
aistraj/            shared AIS scene pipeline: config, pre-processing, data loading, training/evaluation loop, metrics
rigs/               RIGS-Net and the baselines
  config.py         RIGS-Net settings (RConfig), regions, lambda grid, baseline settings, ablation variants
  model.py          RIGS-Net
  qsd.py            Quaternion Ship Domain geometry and the QSD regularizer
  dims.py           per-vessel length/beam from AIS static reports
  official.py       Social-STGCNN and MRGTraj, ported from the official code
  mstformer_official.py  MSTFormer wrapper around the official modules
  paperimpl.py      STMGCN, G-STGAN, DGCN-Transformer, AISFormer, implemented from the papers
  baselines.py      constant-velocity reference
  experiment.py     training, checkpoint selection, scoring
tools/              data building, job runner, tables and figures of the manuscript
jobs/               JSON job lists of every run reported in the manuscript
results/            the generated LaTeX tables and JSON summaries reported in the manuscript
```

## Installation

Python 3.10, PyTorch 2.4 (CUDA 12.1) were used.

```bash
pip install -r requirements.txt
```

MSTFormer imports the network modules of the official release. Clone it and point `MSTFORMER_DIR` to the folder that contains `models/` and `utils/`:

```bash
git clone https://github.com/simple316/MSTFormer third_party/MSTFormer
export MSTFORMER_DIR=$PWD/third_party/MSTFormer/MSTFormer   # default location if unset
```

## Data

The experiments use the public full-rate AIS archive of the Danish Maritime Authority (http://aisdata.ais.dk/), 2024-08-01 to 2024-10-31, in five regions: RWA (Rostock-Warnemünde), ESB (Esbjerg), KIB (Kiel Bight), GOT (Gothenburg approach) and ORN (Øresund North). The region boxes are defined in `tools/build_dma_regions.py`.

The processed data of the five regions are distributed as `RIGS-Net-data.tar.gz` in the [v1.0 release](https://github.com/yinntu2024-coder/RIGSNet/releases/tag/v1.0); extract it in the repository root to obtain `data/processed/`:

```bash
wget https://github.com/yinntu2024-coder/RIGSNet/releases/download/v1.0/RIGS-Net-data.tar.gz
tar xzf RIGS-Net-data.tar.gz
```
 To rebuild them from the raw archive instead, download the daily zip files into `data/dma_zips/` (or set `DMA_ZIPS`), then:

```bash
python tools/build_dma_regions.py --stage extract   # Class A reports inside each region box
python tools/build_dma_regions.py --stage raw       # one CSV per region + vessel dimensions
python tools/build_dma_regions.py --stage process   # cleaning, segmentation, 5 s resampling, chronological 70/15/15 split
python tools/build_dma_regions.py --stage dims      # per-segment vessel length/beam
```

The processed data are written to `data/processed/<REGION>/` (override with `RIGS_DATA_ROOT`).

## Running the experiments

Each job list in `jobs/` is a list of runs (`dataset`, `model`, `tag`, `seed`, `overrides`). Workers claim jobs atomically, so several workers can share one list:

```bash
# one worker per GPU; results in runs_dma/<REGION>/<run>/seed<k>/
tools/launch.sh runs_dma test "0 1" jobs/rigs_v2.json
tools/launch.sh runs_dma test "2 3" jobs/main_seed0.json jobs/main_seed12.json
tools/stop.sh                                         # stop all workers
```

| Job list | Content |
|---|---|
| `rigs_v2.json` | RIGS-Net (final design), 5 regions x 3 seeds: lambda_safe in {0, 0.025, 0.05, 0.1, 0.2}, ablations (base, risk, no interaction) |
| `main_seed0.json`, `main_seed12.json` | baselines and the constant-velocity reference, seeds 0 and 1-2 (the RIGS-Net entries in these lists are an earlier design; the reported RIGS-Net results come from `rigs_v2.json`) |
| `tune.json`, `tune_radius.json`, `tune_fair.json`, `tune_fair_dgcntf.json` | validation-only tuning (run with `val` as evaluation split, e.g. into `runs_tuning_dma`) |
| `dev_v2.json` | validation-only selection of the interaction-module design (into `runs_dev_v2`) |

A single run can also be started directly:

```python
from rigs.experiment import run_experiment
run_experiment("RWA", model="rigs", tag="lam0.025", seed=0, runs="runs_dma",
               overrides=["lambda_safe=0.025", "msg_mode=edge", "null_slot=True", "cpa_feat=True"])
```

All settings were selected on the validation splits; the test splits were used only for the final evaluation.

## Tables and figures

The scripts read the run directories and print the LaTeX rows used in the manuscript (nothing is typed in by hand):

| Script | Manuscript item |
|---|---|
| `tools/main_table_tex.py` | main comparison (K = 1, ADE/FDE) |
| `tools/ablation_tex.py` | ablation table |
| `tools/lambda_tex.py` | sensitivity to lambda_safe |
| `tools/qsd_eval.py`, `tools/safety_table.py` | QSD violation rate and minimum separation |
| `tools/qsd_strata.py` | QSD analysis stratified by encounter closeness |
| `tools/val_strata.py`, `tools/test_strata_seeds.py`, `tools/test_strata_tex.py` | interaction effect by encounter closeness |
| `tools/tuning_table.py` | validation sensitivity of the tuned settings |
| `tools/mechanism_figs.py`, `tools/result_figs.py` | mechanism and qualitative figures |
| `tools/dma_grid_survey.py` | encounter-density survey used to select the regions |

`results/` contains the outputs reported in the manuscript.

## Baselines

| Method | Source |
|---|---|
| Social-STGCNN | ported from https://github.com/abduallahmohamed/Social-STGCNN |
| MRGTraj | ported from https://github.com/wisionpeng/MRGTraj |
| MSTFormer | official modules from https://github.com/simple316/MSTFormer |
| STMGCN, G-STGAN, DGCN-Transformer, AISFormer | implemented from the published descriptions (no code released) |

All deviations from the original settings (units, batching, numerical safeguards, early stopping, and the omission of the Transformer component of G-STGAN's spatial encoder) are listed in the docstrings of `rigs/official.py`, `rigs/paperimpl.py`, `rigs/mstformer_official.py` and in the baseline table of the manuscript.

## License

The code is released under the MIT License (`LICENSE`). Modules ported from third-party repositories keep their original terms; see `THIRD_PARTY_NOTICES.md`. The AIS data are subject to the terms of the Danish Maritime Authority.

## Citation

If you use this code, please cite the manuscript (citation to be added on publication).
