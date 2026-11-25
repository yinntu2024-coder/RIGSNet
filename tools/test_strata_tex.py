#!/usr/bin/env python3
"""Rows of tab:test_strata from results/test_strata/<R>.json (tools/val_strata.py table --dump test_dump.npz,
ref = rigs-noint, run = val-selected rigs-lam*). Cell = mean ADE difference (full - no interaction) [95% paired
scene-bootstrap CI]; bold when the CI excludes 0.   python tools/test_strata_tex.py > results/tab_test_strata.tex"""
import json, sys
sys.path.insert(0, '.')
from rigs.config import DATASETS
sel = json.load(open('results/lambda_selected.json'))
lab = [("<1 (GT inside)", r"$<1$ (inside QSD)"), ("1-1.5", r"$[1,1.5)$"), ("1.5-2", r"$[1.5,2)$"), ("2-4", r"$[2,4)$"),
       (">=4", r"$\ge 4$"), ("single vessel", "single vessel"), ("ALL", "All receivers")]
J = {r: json.load(open(f'results/test_strata/{r}.json')) for r in DATASETS}
for k, t in lab:
    cells = []
    for r in DATASETS:
        e = J[r].get(k)
        if e is None:
            cells.append('--'); continue
        m, lo, hi = e[f"diff_rigs-lam{sel[r]:g}"]
        n = e['n']
        s = f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]".replace('+', '$+$').replace('-', '$-$')
        cells.append((rf"\textbf{{{s}}}" if (hi < 0 or lo > 0) else s) + rf" ({n})")
    if k == "ALL":
        print(r"\midrule")
    print(t + ' & ' + ' & '.join(cells) + r' \\')
