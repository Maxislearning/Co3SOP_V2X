"""任务34b: run x training-seed re-analysis of the v4 occ->beam sweep.

Unit of analysis = source run (n=6). Each combo has up to 3 independent trainings
(s0 = original unseeded sweep, s1/s2 = tools/sweep_v4_seed_lane.sh). Seed indices are
NOT paired across combos (s1 of gt_target and s1 of gt_scene_target are unrelated
trainings), so contrasts use the seed-mean per run, then pair by run.

usage: python per_run_seed.py <meta_dir with meta_val.csv/meta_train.csv> <out_dir>
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

META, OUT = sys.argv[1], sys.argv[2]
LOG = '/home/admin0/Co3SOP/Baseline/work_dirs/sweep_v4'
WD = '/home/admin0/Co3SOP/Baseline/work_dirs'
COMBOS = ['gt_scene', 'gt_target', 'gt_scene_target', 'pred_scene', 'pred_target', 'pred_scene_target']
SEEDS = ['s0', 's1', 's2']
RUN = ['legacy', 'seed1', 'seed2', 'seed3', 'seed4', 'seed5']
K, C = 16, 64
rng = np.random.default_rng(0)
os.makedirs(f'{OUT}/data', exist_ok=True)

v = pd.read_csv(f'{META}/meta_val.csv'); t = pd.read_csv(f'{META}/meta_train.csv')
run = (v.sample_id // 1000).values
ytx, yrx = v.tx.values, v.rx.values
is_lr = np.isin(ytx // K, [0, 2]) | np.isin(yrx // K, [0, 2])
nlos = ~v.is_los.values.astype(bool)
pp = np.bincount((t.tx // K) * 4 + t.rx // K, minlength=16).argmax()
majority = ((ytx // K) * 4 + yrx // K) == pp


def softmax(x):
    x = x - x.max(1, keepdims=True); e = np.exp(x); return e / e.sum(1, keepdims=True)


def hits(z):
    ptx, prx = softmax(z['tx']), softmax(z['rx'])
    txp, rxp = ptx.argmax(1) // K, prx.argmax(1) // K
    pj = (ptx[:, :, None] * prx[:, None, :]).reshape(len(ptx), -1)
    oj = np.argsort(-pj, 1)[:, :10]; gj = ytx * C + yrx
    return {'panel_pair': (txp == ytx // K) & (rxp == yrx // K),
            'tx_top1': ptx.argmax(1) == ytx, 'rx_top1': prx.argmax(1) == yrx,
            'joint_top1': oj[:, 0] == gj, 'joint_top10': (oj == gj[:, None]).any(1),
            'lr_panel_pair': ((txp == ytx // K) & (rxp == yrx // K))}


def logits_path(c, s):
    return f'{LOG}/logits_{c}.npz' if s == 's0' else f'{LOG}/logits_{c}_{s}.npz'


def workdir(c, s):
    return f'{WD}/occ_beam_{c}_a2_merged_v4' + ('' if s == 's0' else f'_{s}')


# ---------- per (combo, seed, run) metrics ----------
H, avail = {}, {}
rows = []
for c in COMBOS:
    avail[c] = [s for s in SEEDS if os.path.exists(logits_path(c, s))]
    for s in avail[c]:
        h = hits(np.load(logits_path(c, s))); H[c, s] = h
        for met in ['panel_pair', 'tx_top1', 'rx_top1', 'joint_top1', 'joint_top10']:
            for r in range(6):
                m = run == r
                rows.append(dict(combo=c, seed=s, run=r, metric=met, acc=h[met][m].mean()))
            rows.append(dict(combo=c, seed=s, run='pooled', metric=met, acc=h[met].mean()))
        # subgroup: LR rows and NLOS rows (pooled only; event counts are tiny)
        rows.append(dict(combo=c, seed=s, run='pooled', metric='panel_pair_LR', acc=h['panel_pair'][is_lr].mean()))
        rows.append(dict(combo=c, seed=s, run='pooled', metric='panel_pair_NLOS', acc=h['panel_pair'][nlos].mean()))
        rows.append(dict(combo=c, seed=s, run='pooled', metric='panel_pair_LOS_nonLR',
                         acc=h['panel_pair'][~nlos & ~is_lr].mean()))
for r in range(6):
    rows.append(dict(combo='majority', seed='-', run=r, metric='panel_pair', acc=majority[run == r].mean()))
rows.append(dict(combo='majority', seed='-', run='pooled', metric='panel_pair', acc=majority.mean()))
L = pd.DataFrame(rows); L.to_csv(f'{OUT}/data/seed_long.csv', index=False)
json.dump(avail, open(f'{OUT}/data/seed_availability.json', 'w'), indent=1)


def mat(c, met='panel_pair'):
    """seeds x runs matrix of accuracies for combo c"""
    g = L[(L.combo == c) & (L.metric == met) & (L.run != 'pooled')]
    return np.array([g[g.seed == s].sort_values('run').acc.values for s in avail[c]])


# ---------- training dynamics per seed ----------
dyn = []
for c in COMBOS:
    for s in avail[c]:
        hp = f'{workdir(c, s)}/history.json'
        if not os.path.exists(hp):
            continue
        hist = json.load(open(hp))
        comb = [e['val_tx_top1'] + e['val_rx_top1'] for e in hist]
        b = int(np.argmax(comb))
        dyn.append(dict(combo=c, seed=s, epochs_run=len(hist), best_epoch=b + 1,
                        train_panel_at_best=(hist[b]['train_tx_panel_acc'] + hist[b]['train_rx_panel_acc']) / 2,
                        val_panel_at_best=(hist[b]['val_tx_panel_acc'] + hist[b]['val_rx_panel_acc']) / 2,
                        train_panel_last=(hist[-1]['train_tx_panel_acc'] + hist[-1]['train_rx_panel_acc']) / 2))
pd.DataFrame(dyn).to_csv(f'{OUT}/data/seed_training_dynamics.csv', index=False)

# ---------- variance decomposition ----------
vd = []
for c in COMBOS:
    M = mat(c)
    if len(M) < 2:
        continue
    pooled = L[(L.combo == c) & (L.metric == 'panel_pair') & (L.run == 'pooled')].acc.values
    vd.append(dict(combo=c, n_seeds=len(M),
                   seed_sd_pooled=pooled.std(ddof=1),                       # training noise on the headline number
                   seed_sd_within_run_mean=M.std(0, ddof=1).mean(),          # training noise inside one run
                   seed_sd_within_run_max=M.std(0, ddof=1).max(),
                   worst_run=RUN[int(M.std(0, ddof=1).argmax())],
                   run_sd_of_seedmean=M.mean(0).std(ddof=1)))                 # scenario heterogeneity
VD = pd.DataFrame(vd); VD.to_csv(f'{OUT}/data/seed_variance_decomposition.csv', index=False)


# ---------- contrasts on seed-mean per run ----------
def tci(x):
    x = np.asarray(x, float); m = x.mean(); h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def signflip_p(d):
    obs = abs(d.mean())
    return np.mean([abs((d * np.array(s)).mean()) >= obs - 1e-12 for s in itertools.product([1, -1], repeat=len(d))])


def hboot(MA, MB, n=10000):
    """hierarchical bootstrap: resample runs, then seeds within each combo (independently)."""
    out = np.empty(n)
    for i in range(n):
        rr = rng.integers(0, 6, 6)
        sa = rng.integers(0, len(MA), len(MA)); sb = rng.integers(0, len(MB), len(MB))
        out[i] = (MA[sa][:, rr].mean(0) - MB[sb][:, rr].mean(0)).mean()
    return np.percentile(out, [2.5, 97.5])


def contrast(name, MA, MB):
    d = MA.mean(0) - MB.mean(0); mu, lo, hi = tci(d)
    blo, bhi = hboot(MA, MB)
    # cell-level consistency: every seed of A vs every seed of B within a run
    cells = (MA[:, None, :] - MB[None, :, :])
    sd = d.std(ddof=1)
    return dict(contrast=name, seeds_A=len(MA), seeds_B=len(MB), diff_mean=mu, t_ci_lo=lo, t_ci_hi=hi,
                hboot_lo=blo, hboot_hi=bhi, dz=mu / sd if sd > 0 else np.nan,
                runs_pos=int((d > 0).sum()), p_signflip=signflip_p(d), p_t=stats.ttest_1samp(d, 0).pvalue,
                frac_seedpair_run_cells_pos=float((cells > 0).mean()),
                per_run=' '.join(f'{x:+.3f}' for x in d))


maj = np.array([majority[run == r].mean() for r in range(6)])[None, :]
PRIMARY = [('Q1 GT: scene_target - target', 'gt_scene_target', 'gt_target'),
           ('Q2 Pred: scene_target - target', 'pred_scene_target', 'pred_target'),
           ('Q3 scene_target: GT - Pred', 'gt_scene_target', 'pred_scene_target'),
           ('Q4 target: Pred - GT', 'pred_target', 'gt_target'),
           ('Q5 Pred target - majority', 'pred_target', None),
           ('Q6 Pred scene_target - majority', 'pred_scene_target', None)]
res = []
for name, a, b in PRIMARY:
    if not avail.get(a) or (b and not avail.get(b)):
        continue
    res.append(contrast(name, mat(a), mat(b) if b else maj))
if all(avail.get(c) for c in ['gt_scene_target', 'gt_target', 'pred_scene_target', 'pred_target']):
    GA, GB, PA, PB = mat('gt_scene_target'), mat('gt_target'), mat('pred_scene_target'), mat('pred_target')
    d = (GA.mean(0) - GB.mean(0)) - (PA.mean(0) - PB.mean(0)); mu, lo, hi = tci(d)
    res.append(dict(contrast='Q7 interaction (GT scene gain - Pred scene gain)', diff_mean=mu, t_ci_lo=lo, t_ci_hi=hi,
                    dz=mu / d.std(ddof=1), runs_pos=int((d > 0).sum()), p_signflip=signflip_p(d),
                    p_t=stats.ttest_1samp(d, 0).pvalue, per_run=' '.join(f'{x:+.3f}' for x in d)))
R = pd.DataFrame(res)
if len(R):
    order = np.argsort(R.p_signflip.values); adj = np.empty(len(R)); mx = 0
    for k, i in enumerate(order):
        mx = max(mx, min(1, (len(R) - k) * R.p_signflip.values[i])); adj[i] = mx
    R['p_signflip_holm'] = adj
    order = np.argsort(R.p_t.values); adj = np.empty(len(R)); mx = 0
    for k, i in enumerate(order):
        mx = max(mx, min(1, (len(R) - k) * R.p_t.values[i])); adj[i] = mx
    R['p_t_holm'] = adj
R.to_csv(f'{OUT}/data/seed_contrasts.csv', index=False)

# per-seed replication of the s0-only headline contrasts (does each single training reproduce the sign?)
rep = []
for name, a, b in PRIMARY:
    if not avail.get(a) or (b and not avail.get(b)):
        continue
    MA, MB = mat(a), (mat(b) if b else maj)
    for i, sa in enumerate(avail[a]):
        for j in range(len(MB)):
            d = MA[i] - MB[j]
            rep.append(dict(contrast=name, seed_A=sa, seed_B=(avail[b][j] if b else '-'),
                            pooled_like=d.mean(), runs_pos=int((d > 0).sum())))
pd.DataFrame(rep).to_csv(f'{OUT}/data/seed_contrast_replication.csv', index=False)

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print('availability:', {c: avail[c] for c in COMBOS})
print(L[(L.metric == 'panel_pair') & (L.run == 'pooled')].pivot(index='combo', columns='seed', values='acc').round(3))
print(VD.round(3).to_string(index=False))
print(pd.DataFrame(dyn).round(3).to_string(index=False))
if len(R):
    print(R.round(4).to_string(index=False))
