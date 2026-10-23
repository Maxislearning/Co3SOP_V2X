import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
O = sys.argv[1]
L = pd.read_csv(f'{O}/data/seed_long.csv'); R = pd.read_csv(f'{O}/data/seed_contrasts.csv')
RUN = ['legacy', 'seed1', 'seed2', 'seed3', 'seed4', 'seed5']
BLUE, ORANGE, AQUA, INK, INK2, GRID, SURF = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'savefig.facecolor': SURF})
P = L[(L.metric == 'panel_pair') & (L.run != 'pooled')].copy(); P['run'] = P.run.astype(int)

# Fig 4: per-run accuracy, every seed as a dot, seed-mean as bar tick; four main combos
combos = [('gt_target', 'GT target'), ('gt_scene_target', 'GT scene+target'),
          ('pred_target', 'Pred target'), ('pred_scene_target', 'Pred scene+target')]
fig, axes = plt.subplots(1, 4, figsize=(9.6, 3.2), sharey=True)
maj = P[P.combo == 'majority'].sort_values('run').acc.values
for ax, (c, t) in zip(axes, combos):
    g = P[P.combo == c]
    for r in range(6):
        v = g[g.run == r].acc.values
        ax.scatter(np.full(len(v), r), v, s=14, color=INK2, alpha=.75, zorder=2, edgecolor=SURF, linewidth=.8)
        ax.plot([r - .3, r + .3], [v.mean()] * 2, color=BLUE, lw=2.2, zorder=3)
        ax.plot([r - .18, r + .18], [maj[r]] * 2, color=ORANGE, lw=1, zorder=1)
    ax.set_xticks(range(6)); ax.set_xticklabels(RUN, rotation=45, fontsize=7.5); ax.set_title(t, fontsize=9.5, color=INK)
    ax.grid(axis='y', color=GRID, lw=.6); ax.set_axisbelow(True); ax.set_ylim(0, 1.02)
axes[0].set_ylabel('Panel-pair accuracy (val)')
axes[0].plot([], [], color=BLUE, lw=2.2, label='mean of 3 seeds'); axes[0].scatter([], [], s=14, color=INK2, label='one training seed')
axes[0].plot([], [], color=ORANGE, lw=1, label='majority baseline'); axes[0].legend(fontsize=7, frameon=False, loc='lower left')
fig.tight_layout(); [fig.savefig(f'{O}/figures/figure-04-per-run-by-seed.{e}', dpi=200) for e in ('pdf', 'png')]

# Fig 5: forest, seed-mean contrasts with hierarchical bootstrap
lab = {'Q1': 'GT: scene+target − target', 'Q2': 'Pred: scene+target − target', 'Q3': 'scene+target: GT − Pred',
       'Q4': 'target: Pred − GT', 'Q5': 'Pred target − majority', 'Q6': 'Pred scene+target − majority',
       'Q7': 'Interaction: GT scene gain − Pred scene gain'}
fig, ax = plt.subplots(figsize=(8.6, 3.8))
for i, r in R.iterrows():
    y = len(R) - 1 - i; d = np.array([float(s) for s in r.per_run.split()])
    ax.scatter(d, np.full(6, y) + np.linspace(-.18, .18, 6), s=10, color=INK2, alpha=.7, zorder=2)
    ax.plot([r.t_ci_lo, r.t_ci_hi], [y, y], color=BLUE, lw=2.2, zorder=3); ax.scatter(r.diff_mean, y, marker='D', s=36, color=BLUE, zorder=4)
    if not np.isnan(r.hboot_lo): ax.plot([r.hboot_lo, r.hboot_hi], [y + .28] * 2, color=AQUA, lw=1.4, zorder=3)
    ax.text(0.64, y, f'{r.runs_pos}/6 >0  p_perm={r.p_signflip:.3f}  Holm(t)={r.p_t_holm:.2f}', va='center', fontsize=7, color=INK2)
ax.axvline(0, color=INK, lw=.8); ax.set_yticks(range(len(R)))
ax.set_yticklabels([lab[c.split()[0]] for c in R.contrast][::-1], fontsize=8)
ax.set_xlim(-0.4, 1.12); ax.set_xticks(np.arange(-.4, .61, .2)); ax.grid(axis='x', color=GRID, lw=.6); ax.set_axisbelow(True)
ax.set_xlabel('Δ panel-pair accuracy (seed-mean per run, paired by run)')
ax.scatter([], [], s=10, color=INK2, label='per-run Δ (3-seed mean)'); ax.scatter([], [], marker='D', color=BLUE, label='mean ± 95% t-CI (df=5)')
ax.plot([], [], color=AQUA, lw=1.4, label='95% hierarchical bootstrap (runs, then seeds)')
ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.17), ncol=3, fontsize=7, frameon=False)
fig.tight_layout(); [fig.savefig(f'{O}/figures/figure-05-seed-mean-contrasts.{e}', dpi=200) for e in ('pdf', 'png')]

# Fig 6: power vs number of runs (3 seeds), alpha=.05
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
ns = np.arange(4, 81)
from statsmodels.stats.power import TTestPower
tp = TTestPower()
for ax, delta in zip(axes, (0.05, 0.10)):
    for tau, col in ((0.065, BLUE), (0.10, ORANGE), (0.14, AQUA)):
        dz = delta / np.sqrt(tau ** 2 + 2 * 0.05 ** 2 / 3)
        pw = [tp.power(dz, n, 0.05) for n in ns]; ax.plot(ns, pw, color=col, lw=2, label=f'τ = {tau}')
    ax.axhline(.8, color=INK2, lw=.8, ls='--'); ax.axvline(6, color=INK2, lw=.8, ls=':')
    ax.text(6.8, .05, 'now: 6 runs', fontsize=7, color=INK2)
    ax.set_title(f'detect Δ = {int(delta * 100)} pp', fontsize=9.5, color=INK); ax.set_xlabel('number of independent runs')
    ax.set_xlim(4, 80); ax.grid(color=GRID, lw=.6); ax.set_axisbelow(True)
axes[0].set_ylabel('power (paired t, α=.05, 3 seeds)'); axes[0].set_ylim(0, 1)
axes[1].legend(title='run-level SD of Δ', fontsize=7, title_fontsize=7, frameon=False, loc='lower right')
fig.tight_layout(); [fig.savefig(f'{O}/figures/figure-06-power-vs-runs.{e}', dpi=200) for e in ('pdf', 'png')]
print('ok')
