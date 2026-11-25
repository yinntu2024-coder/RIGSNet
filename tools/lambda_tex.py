#!/usr/bin/env python3
"""Rows of tab:lambda_sensitivity: per-region ADE/FDE (K=1) for each shared lambda_safe + validation-selected row,
Avg ADE/FDE, Avg QSD VR (results/qsd.json, mean over regions) and Avg mean min. separation. SEEDS=0 -> seed 0 only.
    SEEDS=0 python tools/lambda_tex.py > results/tab_lambda.tex"""
import importlib.util, json, os, sys
import numpy as np
sys.path.insert(0, '.')
s = importlib.util.spec_from_file_location('mt', 'tools/make_tables.py'); mt = importlib.util.module_from_spec(s); s.loader.exec_module(mt)
from rigs.config import DATASETS, LAMBDAS
SEEDS = os.environ.get('SEEDS')
if SEEDS:
    _ro = mt.runs_of; keep = {f'seed{x}' for x in SEEDS.split(',')}
    mt.runs_of = lambda runs, ds, name: [d for d in _ro(runs, ds, name) if os.path.basename(d) in keep]
R = 'runs_dma'; q = json.load(open('results/qsd.json'))
sel = {ds: mt.select_lambda(R, ds)[0] for ds in DATASETS}
def stats(name_of):
    out = {}
    for ds in DATASETS:
        rr = [d for d in mt.runs_of(R, ds, name_of(ds)) if os.path.exists(os.path.join(d, 'test_metrics.json'))]
        out[ds] = dict(ade=np.mean([mt.metric(d, 'meanADE') for d in rr]), fde=np.mean([mt.metric(d, 'meanFDE') for d in rr]),
                       vr=np.mean([q[os.path.relpath(d, R)]['pred']['VR'] for d in rr]), sep=np.mean([q[os.path.relpath(d, R)]['pred']['min_sep'] for d in rr]))
    out['Avg'] = {k: np.mean([out[ds][k] for ds in DATASETS]) for k in ('ade', 'fde', 'vr', 'sep')}
    return out
rows = [(f'{l:g}', stats(lambda ds, l=l: f'rigs-lam{l:g}')) for l in LAMBDAS] + [(r'Val.-selected$^\dagger$', stats(lambda ds: f'rigs-lam{sel[ds]:g}'))]
best = {c: min(round(r[c]['ade'], 2) for _, r in rows) for c in DATASETS + ['Avg']}
print('% selected:', sel)
for i, (lab, r) in enumerate(rows):
    if i == len(rows) - 1: print(r'\midrule')
    cells = []
    for c in DATASETS + ['Avg']:
        t = f"{r[c]['ade']:.2f}/{r[c]['fde']:.2f}"
        cells.append(rf'\textbf{{{t}}}' if round(r[c]['ade'], 2) == best[c] else t)
    print(lab + ' & ' + ' & '.join(cells) + f" & {r['Avg']['vr']:.3f}\\% & {r['Avg']['sep']:.1f}" + r' \\')
