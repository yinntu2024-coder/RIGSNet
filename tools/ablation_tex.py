#!/usr/bin/env python3
"""Rows of manuscript tab:ablation (K=1, ADE/FDE interleaved per region + Avg) from runs_dma.
SEEDS=0 restricts all rows to seed 0 (provisional).   SEEDS=0 python tools/ablation_tex.py > results/tab_ablation.tex"""
import importlib.util, os, sys
sys.path.insert(0, '.')
s = importlib.util.spec_from_file_location('mt', 'tools/make_tables.py'); mt = importlib.util.module_from_spec(s); s.loader.exec_module(mt)
from rigs.config import DATASETS
SEEDS = os.environ.get('SEEDS')
if SEEDS:
    _ro = mt.runs_of; keep = {f'seed{x}' for x in SEEDS.split(',')}
    mt.runs_of = lambda runs, ds, name: [d for d in _ro(runs, ds, name) if os.path.basename(d) in keep]
R = 'runs_dma'
sel = {ds: mt.select_lambda(R, ds)[0] for ds in DATASETS}
def row(name_of, key):
    out, per = {}, {}
    for ds in DATASETS:
        rr = mt.runs_of(R, ds, name_of(ds)); v = [mt.metric(d, key) for d in rr]; out[ds] = mt.agg(v)
        for d, x in zip(rr, v):
            if x is not None: per.setdefault(os.path.basename(d), {})[ds] = x
    out['Avg'] = mt.agg([sum(p.values()) / len(p) for p in per.values() if len(p) == len(DATASETS)]); return out
rows = [('No interaction', '& & &', lambda ds: 'rigs-noint'),
        ('Baseline', r'\checkmark & & &', lambda ds: 'rigs-base'),
        ('Baseline + Risk', r'\checkmark & \checkmark & &', lambda ds: 'rigs-risk'),
        (r'Baseline + Risk + QSD ($\lambda_{\mathrm{safe}}=0$)', r'\checkmark & \checkmark & \checkmark &', lambda ds: 'rigs-lam0'),
        (r'\textbf{RIGS-Net (Ours)}', r'\checkmark & \checkmark & \checkmark & \checkmark', lambda ds: f'rigs-lam{sel[ds]:g}')]
cols = DATASETS + ['Avg']
data = [{k: row(f, k) for k in ('meanADE', 'meanFDE')} for _, _, f in rows]
best = {(c, k): min(d[k][c][0] for d in data if d[k][c][0] is not None) for c in cols for k in ('meanADE', 'meanFDE')}
print('% selected lambda:', sel)
for (lab, marks, _), d in zip(rows, data):
    cells = [mt.cell(*d[k][c], d[k][c][0] is not None and round(d[k][c][0], 2) == round(best[(c, k)], 2)) for c in cols for k in ('meanADE', 'meanFDE')]
    print(f'{lab} & {marks} & ' + ' & '.join(cells) + r' \\')
