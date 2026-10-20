import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

W = '/home/admin0/Co3SOP/Baseline/work_dirs/occ_beam_{}_a2_merged_v4/topk_best.json'
CH = [('scene', 'scene', '#2a78d6'), ('target', 'target', '#eb6834'), ('scene_target', 'scene + target', '#1baf7a')]
SRC = [('gt', 'GT occupancy', '-', 'o'), ('pred', 'Predicted occupancy', '--', 's')]
PANELS = [('tx', 'TX beam (64-way)'), ('rx', 'RX beam (64-way)'), ('joint', 'Joint TX-RX beam pair (4096 pairs)')]
K = list(range(1, 11))

INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#ffffff'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': INK2,
                     'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2})

fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True, facecolor=SURF)
for ax, (key, title) in zip(axes, PANELS):
    for src, _, ls, mk in SRC:
        for ch, _, col in CH:
            y = json.load(open(W.format(f'{src}_{ch}')))[key]['val']
            ax.plot(K, y, color=col, linestyle=ls, linewidth=2, marker=mk, markersize=6.5,
                    markeredgecolor=SURF, markeredgewidth=1.2, zorder=3)
    ax.set_title(title, color=INK, fontsize=11.5, loc='left', pad=8)
    ax.set_xticks(K); ax.set_xticklabels([f'Top{k}' for k in K], fontsize=8.5)
    ax.set_xlabel('Top-k')
    ax.set_ylim(0.28, 1.0); ax.set_xlim(0.6, 10.4)
    ax.grid(axis='y', color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
axes[0].set_ylabel('Accuracy (val)')

handles = [Line2D([], [], color=col, linewidth=2.5, label=lab) for _, lab, col in CH] + \
          [Line2D([], [], color=INK2, linestyle=ls, marker=mk, linewidth=2, markersize=6.5,
                  markeredgecolor=SURF, label=lab) for _, lab, ls, mk in SRC]
fig.legend(handles=handles, loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=10)
fig.text(0.5, -0.02, 'Note: the two "scene" curves are inflated - those models collapse to the majority class '
         '(scene input cannot tell the two links apart).', ha='center', color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0, 1, 0.93))
for ext in ('png', 'pdf'):
    fig.savefig(f'/home/admin0/figures_v4/topk_v4.{ext}', dpi=200, bbox_inches='tight', facecolor=SURF)
print('saved')
