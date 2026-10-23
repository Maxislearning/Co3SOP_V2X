import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
O = sys.argv[1]; D = f'{O}/data'
L = pd.read_csv(f'{D}/per_run_long.csv'); R = pd.read_csv(f'{D}/contrasts_primary.csv')
comp = pd.read_csv(f'{D}/per_run_composition.csv')
RUN = ['legacy', 'seed1', 'seed2', 'seed3', 'seed4', 'seed5']
BLUE, ORANGE, INK, INK2, GRID, SURF = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})
def runvals(m, met='panel_pair'):
    g = L[(L.model == m) & (L.metric == met) & (L.run != 'pooled')].copy(); g['run'] = g.run.astype(int)
    return g.sort_values('run').acc.values
def tci(x):
    m = x.mean(); h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); return m, m - h, m + h

# Fig 1: slopegraph target -> scene_target, GT vs Pred facet
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
for ax, src, title in zip(axes, ['gt', 'pred'], ['GT occupancy input', 'Predicted occupancy input']):
    a, b = runvals(f'{src}_target'), runvals(f'{src}_scene_target'); maj = runvals('majority')
    for i in range(6):
        ax.plot([0, 1], [a[i], b[i]], color=INK2, lw=1, alpha=.55, zorder=1)
        ax.scatter([0, 1], [a[i], b[i]], s=14, color=INK2, zorder=2, edgecolor=SURF, linewidth=1)
        ax.plot([-0.12, -0.04], [maj[i]] * 2, color=INK2, lw=1)  # majority tick
    ys = b.copy(); order = np.argsort(ys); pos = ys[order].copy()
    for _ in range(50):  # dodge labels to >=0.04 apart
        for k in range(1, 6):
            if pos[k] - pos[k - 1] < .04: mid = (pos[k] + pos[k - 1]) / 2; pos[k - 1], pos[k] = mid - .02, mid + .02
    for k, i in enumerate(order): ax.text(1.06, pos[k], RUN[i], va='center', fontsize=7.5, color=INK2)
    for x, v in ((0, a), (1, b)):
        m, lo, hi = tci(v); ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt='D', color=BLUE, ms=6, lw=2, capsize=0, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['target', 'scene+target']); ax.set_xlim(-0.25, 1.35)
    ax.set_title(title, fontsize=9.5, color=INK); ax.grid(axis='y', color=GRID, lw=.6); ax.set_axisbelow(True)
axes[0].set_ylabel('Panel-pair accuracy (val, per run)'); axes[0].set_ylim(0, 1.02)
fig.text(0.01, 0.005, 'Gray: one source run (n=220–314 rows). Blue ◆: mean over 6 runs ± 95% t-CI (df=5).\n'
         'Short tick left of each axis: per-run majority-panel-pair baseline.', fontsize=7, color=INK2)
fig.tight_layout(rect=(0, .07, 1, 1)); [fig.savefig(f'{O}/figures/figure-01-per-run-scene-gain.{e}', dpi=200) for e in ('pdf', 'png')]

# Fig 2: forest plot, run-level vs pooled
lab = {0: 'GT: scene+target − target', 1: 'Pred: scene+target − target', 2: 'scene+target: GT − Pred',
       3: 'target: Pred − GT', 4: 'Pred target − majority', 5: 'Pred scene+target − majority',
       6: 'Interaction: GT scene gain − Pred scene gain'}
fig, ax = plt.subplots(figsize=(8.6, 3.8))
for i, r in R.iterrows():
    y = len(R) - 1 - i; d = np.array([float(s) for s in r.per_run.split()])
    ax.scatter(d, np.full(6, y) + np.linspace(-.18, .18, 6), s=10, color=INK2, alpha=.7, zorder=2)
    ax.plot([r.ci_lo, r.ci_hi], [y, y], color=BLUE, lw=2.2, zorder=3); ax.scatter(r.diff_mean, y, marker='D', s=36, color=BLUE, zorder=4)
    if not np.isnan(r.pooled_diff): ax.scatter(r.pooled_diff, y + .3, marker='v', s=26, color=ORANGE, zorder=4)
    ax.text(0.64, y, f'{r.runs_pos}/6 >0  p_perm={r.p_signflip:.3f}  Holm={r.p_signflip_holm:.2f}', va='center', fontsize=7, color=INK2)
ax.axvline(0, color=INK, lw=.8); ax.set_yticks(range(len(R))); ax.set_yticklabels([lab[i] for i in range(len(R))][::-1], fontsize=8)
ax.set_xlim(-0.4, 1.12); ax.set_xticks(np.arange(-.4, .61, .2)); ax.grid(axis='x', color=GRID, lw=.6); ax.set_axisbelow(True)
ax.set_xlabel('Δ panel-pair accuracy (paired by run)')
ax.scatter([], [], s=10, color=INK2, label='per-run Δ'); ax.scatter([], [], marker='D', color=BLUE, label='run-level mean ± 95% t-CI')
ax.scatter([], [], marker='v', color=ORANGE, label='pooled row-level Δ (for reference)')
ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.17), ncol=3, fontsize=7, frameon=False)
fig.tight_layout(); [fig.savefig(f'{O}/figures/figure-02-run-level-contrasts.{e}', dpi=200) for e in ('pdf', 'png')]

# Fig 3: heatmap model x run + composition
models = ['majority', 'gt_scene', 'gt_target', 'gt_scene_target', 'pred_scene', 'pred_target', 'pred_scene_target']
M = np.array([runvals(m) for m in models])
fig, ax = plt.subplots(figsize=(7.2, 4.0))
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('b', ['#eef4fc', '#2a78d6', '#0b3d7a'])
ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect='auto')
for i in range(M.shape[0]):
    for j in range(6):
        ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=8, color=INK if M[i, j] < .55 else '#ffffff')
ev = {0: (2, 2), 1: (1, 0), 2: (1, 1), 3: (1, 0), 4: (0, 0), 5: (2, 3)}
ax.set_xticks(range(6)); ax.set_xticklabels([f'{RUN[j]}\nn={comp.n_rows[j]}\nNLOS {comp.nlos[j]} ({ev[j][1]} ev)\nL/R ev {ev[j][0]}' for j in range(6)], fontsize=7)
ax.set_yticks(range(len(models))); ax.set_yticklabels([m.replace('_', ' / ', 1) for m in models], fontsize=8)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title('Panel-pair accuracy by source run (val)', fontsize=9.5, color=INK)
fig.tight_layout(); [fig.savefig(f'{O}/figures/figure-03-run-by-model-heatmap.{e}', dpi=200) for e in ('pdf', 'png')]
print('ok')
