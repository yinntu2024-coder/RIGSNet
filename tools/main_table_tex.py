#!/usr/bin/env python3
"""Rows of manuscript tab:cie_mgcn_ade_fde (K=1, ADE/FDE interleaved per region) from runs_dma via make_tables helpers.\n    python tools/main_table_tex.py > results/tab_main_k1.tex"""
import importlib.util, os, sys
sys.path.insert(0, '.')
s=importlib.util.spec_from_file_location('mt','tools/make_tables.py'); mt=importlib.util.module_from_spec(s); s.loader.exec_module(mt)
from rigs.config import DATASETS, STOCHASTIC
SEEDS=os.environ.get('SEEDS')   # e.g. SEEDS=0 -> provisional seed-0-only table (all rows on the same seeds)
if SEEDS:
    _ro=mt.runs_of; keep={f'seed{x}' for x in SEEDS.split(',')}
    mt.runs_of=lambda runs,ds,name: [d for d in _ro(runs,ds,name) if os.path.basename(d) in keep]
R='runs_dma'
sel={ds: mt.select_lambda(R,ds)[0] for ds in DATASETS}
def rigs(key):
    out,per={},{}
    for ds in DATASETS:
        rr=mt.runs_of(R,ds,f"rigs-lam{sel[ds]:g}"); v=[mt.metric(d,key) for d in rr]; out[ds]=mt.agg(v)
        for d,x in zip(rr,v):
            if x is not None: per.setdefault(os.path.basename(d),{})[ds]=x
    out['Avg']=mt.agg([sum(p.values())/len(p) for p in per.values() if len(p)==len(DATASETS)]); return out
def model(m,key):
    return mt.per_model(R,m,key[4:],tag='test_mean') if m in STOCHASTIC else mt.per_model(R,m,key)
names=[('cv','CV (reference)'),('sstgcnn',r'S-STGCNN~\citep{mohamed2020socialstgcnn}'),('stmgcn',r'STMGCN~\citep{liu2022stmgcn}'),
('gstgan',r'G-STGAN~\citep{zhang2023gstgan}'),('mrgtraj',r'MRGTraj~\citep{peng2024mrgtraj}'),('aisformer',r'AISFormer~\citep{yu2025aisformer}'),
('mstformer',r'MSTFormer~\citep{qiang2025mstformer}'),('dgcntf',r'DGCN-Transformer~\citep{zhang2025dgcntransformer}')]
cols=DATASETS+['Avg']
data={}
for k,_ in names: data[k]={key:model(k,key) for key in ('meanADE','meanFDE')}
data['rigs']={key:rigs(key) for key in ('meanADE','meanFDE')}
names.append(('rigs',r'\textbf{Ours}'))
best={(c,key):min((data[k][key][c][0] for k,_ in names if data[k][key][c][0] is not None),default=None) for c in cols for key in ('meanADE','meanFDE')}
print('% selected lambda:',sel)
for k,lab in names:
    cells=[]
    for c in cols:
        for key in ('meanADE','meanFDE'):
            m,sd,n=data[k][key][c]
            cells.append(mt.cell(m,sd,n,m is not None and abs(m-best[(c,key)])<1e-12))
    print(lab+' & '+' & '.join(cells)+r' \\')
