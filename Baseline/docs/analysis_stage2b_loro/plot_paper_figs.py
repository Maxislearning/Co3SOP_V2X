#!/usr/bin/env python3
"""任务43/44/45: 论文配图（发送端/接收端单独的 top-k），只比较预测占用（按折重训 Stage2B）和 GT 占用。
读材料包里 paper_tables.py 生成的 topk_pooled.csv / topk_per_fold.csv，输出到论文目录 /home/admin0/Documents/siyuan/CO3SOP-V2V/figures（pdf + png）。
fig1 top-k 曲线（6 折测试集合并准确率）；fig2 各 run 的 top-1（GT 与预测按 run 连线，看波动）。
颜色：预测 = 蓝（类别色第 1 位）；GT 是上限参考线，用灰色虚线。"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PKG = '/home/admin0/Documents/siyuan/CO3SOP-V2V'
pooled = pd.read_csv(f'{PKG}/tables/topk_pooled.csv')
df = pd.read_csv(f'{PKG}/tables/topk_per_fold.csv')
FOLDS = ['legacy', 'seed1', 'seed2', 'seed3', 'seed4', 'seed5']
SIDES = [('tx', 'Transmitter (Tx)'), ('rx', 'Receiver (Rx)')]
S = {'GT 占用（target）': dict(label='Ground-truth occupancy (upper bound)', c='#52514e', ls='--', m='s'),
     '预测占用（target）': dict(label='Predicted occupancy', c='#2a78d6', ls='-', m='o')}
OUT = f'{PKG}/figures'  # 论文图目录
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'lines.linewidth': 2,
                     'legend.frameon': False, 'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.dpi': 300})


def save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(f'{OUT}/{name}.{ext}')
    plt.close(fig)


def top_legend(fig, ax, ncol):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=ncol, fontsize=8, bbox_to_anchor=(.5, 1.0))


# fig1: top-k 曲线（合并准确率）
fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
for ax, (side, title) in zip(axes, SIDES):
    m = pooled[pooled.metric == side].set_index('model')[[f'top{k}' for k in range(1, 11)]]
    m.columns = range(1, 11)
    for name, st in S.items():
        ax.plot(m.columns, m.loc[name], color=st['c'], ls=st['ls'], marker=st['m'], ms=4.5, label=st['label'])
    ax.set_title(title, color=INK, fontsize=10)
    ax.set_xlabel('k'); ax.set_xticks(range(1, 11)); ax.set_ylim(75, 100)
axes[0].set_ylabel('Top-k accuracy (%)')
top_legend(fig, axes[0], 2)
save(fig, 'fig1_topk_curves')

# fig2: 各折 top-1，GT 与预测按 run 连线
fig, axes = plt.subplots(1, 2, figsize=(7, 2.9), sharey=True)
x = np.arange(len(FOLDS))
for ax, (side, title) in zip(axes, SIDES):
    w = df[df.k == 1].pivot(index='fold', columns='model', values=side).loc[FOLDS] * 100
    ax.vlines(x, w['预测占用（target）'], w['GT 占用（target）'], color=GRID, lw=1.6, zorder=1)
    for name, st in S.items():
        ax.scatter(x, w[name], color=st['c'], marker=st['m'], s=34, zorder=3, label=st['label'],
                   edgecolors='#fcfcfb', linewidths=.8)
        ax.hlines(w[name].mean(), -.5, len(FOLDS) - .5, color=st['c'], ls=':', lw=1, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(FOLDS, fontsize=8); ax.set_xlim(-.5, len(FOLDS) - .5)
    ax.set_title(title, color=INK, fontsize=10); ax.set_xlabel('Held-out run'); ax.grid(axis='x', visible=False)
axes[0].set_ylabel('Top-1 accuracy (%)'); axes[0].set_ylim(50, 100)
top_legend(fig, axes[0], 2)
save(fig, 'fig2_per_fold_top1')

print('ok')
