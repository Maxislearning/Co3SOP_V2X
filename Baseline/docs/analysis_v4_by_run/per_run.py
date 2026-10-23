"""Per-run (source run = unit of analysis) re-analysis of the v4 occ->beam sweep."""
import json, itertools, sys
import numpy as np, pandas as pd
from scipy import stats
S = sys.argv[1]
LOG = '/home/admin0/Co3SOP/Baseline/work_dirs/sweep_v4'
COMBOS = ['gt_scene', 'gt_target', 'gt_scene_target', 'pred_scene', 'pred_target', 'pred_scene_target']
K, C = 16, 64
RUNNAME = {0: 'legacy', 1: 'seed1', 2: 'seed2', 3: 'seed3', 4: 'seed4', 5: 'seed5'}

v = pd.read_csv(f'{S}/meta_val.csv'); t = pd.read_csv(f'{S}/meta_train.csv')
v['run'] = v.sample_id // 1000
runs = sorted(v.run.unique())
ytx, yrx = v.tx.values, v.rx.values

# train-label baselines (same as rerank_prior.py / majority baseline in full_eval)
eps = 1.0
pr_tx = np.bincount(t.tx, minlength=C) + eps / C; pr_tx /= pr_tx.sum()
pr_rx = np.bincount(t.rx, minlength=C) + eps / C; pr_rx /= pr_rx.sum()
pair = np.zeros((C, C)) + eps / C**2
np.add.at(pair, (t.tx.values, t.rx.values), 1); pr_pair = (pair / pair.sum()).ravel()
maj_txp = np.bincount(t.tx // K, minlength=4).argmax(); maj_rxp = np.bincount(t.rx // K, minlength=4).argmax()
pp = np.bincount((t.tx // K) * 4 + t.rx // K, minlength=16).argmax()

def softmax(x):
    x = x - x.max(1, keepdims=True); e = np.exp(x); return e / e.sum(1, keepdims=True)

def row_hits(ptx, prx):
    """per-row boolean hits for each metric"""
    otx = np.argsort(-ptx, 1)[:, :10]; orx = np.argsort(-prx, 1)[:, :10]
    pj = (ptx[:, :, None] * prx[:, None, :]).reshape(len(ptx), -1)
    oj = np.argsort(-pj, 1)[:, :10]; gj = ytx * C + yrx
    txp, rxp = otx[:, 0] // K, orx[:, 0] // K
    return {
        'panel_pair': (txp == ytx // K) & (rxp == yrx // K),
        'tx_panel': txp == ytx // K, 'rx_panel': rxp == yrx // K,
        'tx_top1': otx[:, 0] == ytx, 'rx_top1': orx[:, 0] == yrx,
        'tx_top5': (otx[:, :5] == ytx[:, None]).any(1), 'rx_top5': (orx[:, :5] == yrx[:, None]).any(1),
        'joint_top1': oj[:, 0] == gj, 'joint_top10': (oj == gj[:, None]).any(1),
        '_txp': txp, '_rxp': rxp,
    }

H = {}
for c in COMBOS:
    z = np.load(f'{LOG}/logits_{c}.npz'); H[c] = row_hits(softmax(z['tx']), softmax(z['rx']))
n = len(v)
H['prior'] = row_hits(np.tile(pr_tx, (n, 1)), np.tile(pr_rx, (n, 1)))
H['majority'] = {'panel_pair': ((ytx // K) * 4 + yrx // K) == pp,
                 'tx_panel': ytx // K == maj_txp, 'rx_panel': yrx // K == maj_rxp}
METRICS = ['panel_pair', 'tx_panel', 'rx_panel', 'tx_top1', 'rx_top1', 'tx_top5', 'rx_top5', 'joint_top1', 'joint_top10']

# --- per-run table (long) ---
rows = []
for m, h in H.items():
    for met in METRICS:
        if met not in h: continue
        for r in runs:
            msk = (v.run == r).values
            rows.append(dict(model=m, metric=met, run=r, acc=h[met][msk].mean(), n=msk.sum()))
        rows.append(dict(model=m, metric=met, run='pooled', acc=h[met].mean(), n=n))
L = pd.DataFrame(rows); L.to_csv(f'{S}/per_run_long.csv', index=False)

# minority-panel recall per run (LEFT/RIGHT), TX and RX
mrows = []
for c in COMBOS:
    for side, y, p in (('tx', ytx // K, H[c]['_txp']), ('rx', yrx // K, H[c]['_rxp'])):
        for r in runs:
            msk = (v.run == r).values & np.isin(y, [0, 2])
            mrows.append(dict(model=c, side=side, run=r, n_LR=int(msk.sum()),
                              recall_LR=(p[msk] == y[msk]).mean() if msk.any() else np.nan))
pd.DataFrame(mrows).to_csv(f'{S}/per_run_LR_recall.csv', index=False)

# composition per run
comp = v.groupby('run').apply(lambda g: pd.Series(dict(
    n_rows=len(g), n_points=g.sample_id.nunique(), nlos=int((~g.is_los).sum()),
    tx_LR=int(np.isin(g.tx // K, [0, 2]).sum()), rx_LR=int(np.isin(g.rx // K, [0, 2]).sum()))))
comp.to_csv(f'{S}/per_run_composition.csv')

# --- descriptive over runs ---
def tci(x):
    x = np.asarray(x, float); m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(.975, len(x) - 1) * se; return m, x.std(ddof=1), m - h, m + h
desc = []
for (m, met), g in L[L.run != 'pooled'].groupby(['model', 'metric']):
    x = g.sort_values('run').acc.values; mu, sd, lo, hi = tci(x)
    desc.append(dict(model=m, metric=met, runs_mean=mu, runs_sd=sd, ci_lo=lo, ci_hi=hi,
                     run_min=x.min(), run_max=x.max(),
                     pooled=L[(L.model == m) & (L.metric == met) & (L.run == 'pooled')].acc.item()))
D = pd.DataFrame(desc); D.to_csv(f'{S}/descriptive.csv', index=False)

# --- paired contrasts, unit = run ---
def signflip_p(d):
    d = np.asarray(d); obs = abs(d.mean())
    null = [abs((d * s).mean()) for s in itertools.product([1, -1], repeat=len(d))]
    return np.mean(np.array(null) >= obs - 1e-12)

def contrast(a, b, met):
    xa = L[(L.model == a) & (L.metric == met) & (L.run != 'pooled')].sort_values('run').acc.values
    xb = L[(L.model == b) & (L.metric == met) & (L.run != 'pooled')].sort_values('run').acc.values
    d = xa - xb; mu, sd, lo, hi = tci(d)
    ha, hb = H[a][met], H[b][met]
    b01, b10 = int((ha & ~hb).sum()), int((~ha & hb).sum())
    naive = stats.binomtest(b01, b01 + b10).pvalue if b01 + b10 else 1.0
    loo = [np.delete(d, i).mean() for i in range(len(d))]
    return dict(a=a, b=b, metric=met, diff_mean=mu, diff_sd=sd, ci_lo=lo, ci_hi=hi,
                dz=mu / sd if sd > 0 else np.nan, runs_pos=int((d > 0).sum()), runs_neg=int((d < 0).sum()),
                p_t=stats.ttest_rel(xa, xb).pvalue, p_signflip=signflip_p(d),
                p_wilcoxon=stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0,
                shapiro_p=stats.shapiro(d).pvalue, loo_min=min(loo), loo_max=max(loo),
                pooled_diff=ha.mean() - hb.mean(), naive_rowlevel_mcnemar_p=naive,
                per_run=' '.join(f'{x:+.3f}' for x in d))

PRIMARY = [('gt_scene_target', 'gt_target'), ('pred_scene_target', 'pred_target'),
           ('gt_scene_target', 'pred_scene_target'), ('pred_target', 'gt_target'),
           ('pred_target', 'majority'), ('pred_scene_target', 'majority')]
res = [contrast(a, b, 'panel_pair') for a, b in PRIMARY]
# interaction: scene gain under GT minus under Pred
def runvals(m, met): return L[(L.model == m) & (L.metric == met) & (L.run != 'pooled')].sort_values('run').acc.values
dint = (runvals('gt_scene_target', 'panel_pair') - runvals('gt_target', 'panel_pair')) - \
       (runvals('pred_scene_target', 'panel_pair') - runvals('pred_target', 'panel_pair'))
mu, sd, lo, hi = tci(dint)
res.append(dict(a='(gtST-gtT)', b='(predST-predT)', metric='panel_pair', diff_mean=mu, diff_sd=sd, ci_lo=lo, ci_hi=hi,
                dz=mu / sd, runs_pos=int((dint > 0).sum()), runs_neg=int((dint < 0).sum()),
                p_t=stats.ttest_1samp(dint, 0).pvalue, p_signflip=signflip_p(dint), p_wilcoxon=stats.wilcoxon(dint).pvalue,
                shapiro_p=stats.shapiro(dint).pvalue, loo_min=min(np.delete(dint, i).mean() for i in range(6)),
                loo_max=max(np.delete(dint, i).mean() for i in range(6)), pooled_diff=np.nan,
                naive_rowlevel_mcnemar_p=np.nan, per_run=' '.join(f'{x:+.3f}' for x in dint)))
R = pd.DataFrame(res)
# Holm over the primary family on the sign-flip p
order = np.argsort(R.p_signflip.values); m_ = len(R); adj = np.empty(m_); run_max = 0
for rank, i in enumerate(order):
    run_max = max(run_max, min(1, (m_ - rank) * R.p_signflip.values[i])); adj[i] = run_max
R['p_signflip_holm'] = adj
R.to_csv(f'{S}/contrasts_primary.csv', index=False)
# secondary metrics, same contrasts, no inference emphasis
sec = [contrast(a, b, met) for a, b in PRIMARY[:4] for met in ['tx_top1', 'rx_top1', 'joint_top1', 'joint_top10', 'tx_panel', 'rx_panel']]
pd.DataFrame(sec).to_csv(f'{S}/contrasts_secondary.csv', index=False)

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(comp)
print(L[L.run == 'pooled'].pivot(index='model', columns='metric', values='acc').round(3))
print(L[L.metric == 'panel_pair'].pivot(index='model', columns='run', values='acc').round(3))
print(R.drop(columns=['naive_rowlevel_mcnemar_p']).round(4).to_string())
print(R[['a', 'b', 'pooled_diff', 'naive_rowlevel_mcnemar_p', 'p_signflip']].round(5))
print(pd.DataFrame(sec)[['a', 'b', 'metric', 'diff_mean', 'ci_lo', 'ci_hi', 'runs_pos', 'p_signflip', 'per_run']].round(3).to_string())
