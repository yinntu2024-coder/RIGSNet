#!/usr/bin/env python3
"""Appendix table: validation sensitivity of the tuned settings (RIGS-Net) and of the baselines' learning rate.

All numbers are VALIDATION ADE/FDE (m, K = 1) of seed 0, read from
  runs_tuning_dma/<R>/<run>/seed0/val_metrics.json            (tuning grid)
  runs_dma/<R>/<model>/seed0/history.json (selected epoch)     (baselines at their published setting)
Cells whose run does not exist yet are printed as "--".

The "-cv-" baseline runs of tune_fair.json are not used (out_mode is not implemented by those baselines).
DGCN-Transformer honours out_mode and is run by jobs/tune_fair_dgcntf.json
(lr 1e-3 and 3e-4); AISFormer (token classification, autoregressive) has no displacement output to offset.

    python tools/tuning_table.py [--out results/tab_tuning_sensitivity.tex]
"""
import argparse
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TUNE, MAIN = os.path.join(ROOT, "runs_tuning_dma"), os.path.join(ROOT, "runs_dma")
REGIONS = ["RWA", "ESB", "KIB"]


def tuned(region, run):
    p = os.path.join(TUNE, region, run, "seed0", "val_metrics.json")
    if not os.path.exists(p):
        return None
    s = json.load(open(p))["summary"]
    return s["ADE"], s["FDE"]


def main_run(region, model):
    """Validation ADE/FDE of the checkpoint kept by the main (test) run, if that run has finished training."""
    d = os.path.join(MAIN, region, model, "seed0")
    if not os.path.exists(os.path.join(d, "test_metrics.json")):
        return None
    sel = [h for h in json.load(open(os.path.join(d, "history.json"))) if h["selected"]]
    return (sel[-1]["val_ADE"], sel[-1]["val_FDE"]) if sel else None


# (group, label, {region: getter}, selected?)
def rows():
    R = []
    g = r"\multicolumn{7}{l}{\textit{(a) RIGS-Net: decoder output and learning rate} ($R=2$~nmi, $\lambda_{\mathrm{safe}}=0.05$)} \\"
    R.append(g)
    for mode, mlab in [("direct", "Direct displacement"), ("cv", "Offset from CV")]:
        for lr, llab in [("0.001", r"$10^{-3}$"), ("0.0003", r"$3\times10^{-4}$")]:
            R.append((f"{mlab}, lr {llab}", {r: tuned(r, f"rigs-{mode}-lr{lr}") for r in ("RWA", "ESB")},
                      mode == "cv" and lr == "0.0003"))
    R.append(r"\midrule")
    R.append(r"\multicolumn{7}{l}{\textit{(b) RIGS-Net: candidate-neighbour radius $R$} (offset from CV, lr $3\times10^{-4}$, $\lambda_{\mathrm{safe}}=0$)} \\")
    for tag, lab in [("r0", "All vessels of the scene"), ("r3704", r"$R=2$~nmi"), ("r9260", r"$R=5$~nmi")]:
        R.append((lab, {r: tuned(r, f"rigs-{tag}") for r in REGIONS}, tag == "r3704"))
    R.append(r"\midrule")
    R.append(r"\multicolumn{7}{l}{\textit{(c) Baselines: learning rate} (published setting\textsuperscript{$\dagger$} vs.\ RIGS-Net grid)} \\")
    base = [("aisformer", "AISFormer", "0.0006", r"$6\times10^{-4}$"),
            ("mstformer", "MSTFormer", "0.000005", r"$5\times10^{-6}$"),
            ("dgcntf", "DGCN-Transformer", "0.001", r"$10^{-3}$")]
    for m, name, pub, publab in base:
        grid = [(pub, publab)] + [(lr, l) for lr, l in [("0.001", r"$10^{-3}$"), ("0.0003", r"$3\times10^{-4}$")]
                                  if lr != pub]                         # published lr may lie on the grid
        for lr, llab in grid:
            if lr == pub:
                if pub == "0.001":                                        # identical config to the tuning run
                    vals = {r: tuned(r, f"{m}-lr0.001") for r in ("RWA", "ESB")}
                else:
                    vals = {r: main_run(r, m) for r in ("RWA", "ESB")}
                R.append((f"{name}, lr {llab}\\textsuperscript{{$\\dagger$}}", vals, False))
            else:
                R.append((f"{name}, lr {llab}", {r: tuned(r, f"{m}-lr{lr}") for r in ("RWA", "ESB")}, False))
    R.append(r"\midrule")
    R.append(r"\multicolumn{7}{l}{\textit{(d) Baseline with the RIGS-Net output parameterization} (offset from CV)} \\")
    for lr, llab in [("0.001", r"$10^{-3}$"), ("0.0003", r"$3\times10^{-4}$")]:
        R.append((f"DGCN-Transformer + CV offset, lr {llab}",
                  {r: tuned(r, f"dgcntf-cv-lr{lr}") for r in ("RWA", "ESB")}, False))
    R.append(r"\midrule")
    R.append(("Constant velocity (reference)", {r: tuned(r, "cv") for r in ("RWA", "ESB")}, False))
    return R


def fmt(v, best):
    if v is None:
        return "--", "--"
    a, f = f"{v[0]:.2f}", f"{v[1]:.2f}"
    return (rf"\textbf{{{a}}}", rf"\textbf{{{f}}}") if best else (a, f)


def latex():
    out = [r"\begin{table*}[t]", r"\centering",
           r"\caption{Validation sensitivity of the tuned settings (seed 0, $K=1$, ADE/FDE in m). "
           r"RIGS-Net choices were fixed on the RWA and ESB validation splits (radius also on KIB) before any "
           r"test evaluation; bold marks the setting used in the main experiments. The three radii differ by "
           r"at most 0.08~m in ADE; $R=2$~nmi was retained. Baselines are reported in the main tables at "
           r"their published learning rate (\textsuperscript{$\dagger$}); the other rows retrain them on the "
           r"RIGS-Net learning-rate grid with all other settings unchanged; (d) gives DGCN-Transformer the "
           r"constant-velocity offset output selected for RIGS-Net. -- : not run.}",
           r"\label{tab:tuning_sensitivity}", r"\footnotesize", r"\setlength{\tabcolsep}{5pt}",
           r"\begin{tabular}{lcccccc}", r"\toprule",
           r"& \multicolumn{2}{c}{RWA} & \multicolumn{2}{c}{ESB} & \multicolumn{2}{c}{KIB} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
           r"Setting & ADE & FDE & ADE & FDE & ADE & FDE \\", r"\midrule"]
    for r in rows():
        if isinstance(r, str):
            out.append(r)
            continue
        lab, vals, sel = r
        cells = []
        for reg in REGIONS:
            cells += fmt(vals.get(reg), sel)
        out.append(f"{lab} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tab_tuning_sensitivity.tex"))
    a = ap.parse_args()
    s = latex()
    open(a.out, "w").write(s)
    print(s)
