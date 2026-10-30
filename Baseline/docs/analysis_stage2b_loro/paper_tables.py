#!/usr/bin/env python3
"""任务42/44/45: 论文用表——预测占用（按折重训 Stage2B）vs GT 占用，发送端/接收端单独的 top-k。
主表 = 合并准确率：6 折测试集合在一起（每个样本都由没见过它的那一折模型预测），不带 ±。
附表 = 各 run 的 top-1 和范围，以及以 run 为单位（n=6）的配对差值（精确符号翻转置换 + 配对 t）。
输出到论文材料包 /home/admin0/Documents/siyuan/CO3SOP-V2V/tables。
任务48：默认读 v5（28 GHz 标签）的结果目录 v5_gated / v5_s2b_<折>；TAG_PREFIX= PKG=<目录> 可重算 v4（24 GHz）。"""
import itertools
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = '/home/admin0/Co3SOP/Baseline/work_dirs/loro_cv'
PKG = os.environ.get('PKG', '/home/admin0/Documents/siyuan/CO3SOP-V2V/tables')
P = os.environ.get('TAG_PREFIX', 'v5_')
FOLDS = ['legacy', 'seed1', 'seed2', 'seed3', 'seed4', 'seed5']
MODELS = {'GT 占用（target）': (P + 'gated', 'gt_target'), '预测占用（target）': (P + 's2b_{f}', 'pred_target'),
          'GT 占用（scene+target）': (P + 'gated', 'gt_scene_target'), '预测占用（scene+target）': (P + 's2b_{f}', 'pred_scene_target')}
NAMES = list(MODELS)
K = 10


def softmax(x):
    x = x - x.max(1, keepdims=True); e = np.exp(x); return e / e.sum(1, keepdims=True)


def hits(tag, combo, f):
    """每个测试样本的累计命中 [N, K]：tx、rx、联合波束对；以及面板对命中 [N]"""
    z = np.load(f'{ROOT}/{tag.format(f=f)}/logits/{combo}_{f}_s0.npz')
    ptx, prx = softmax(z['tx']), softmax(z['rx'])
    otx, orx = np.argsort(-ptx, 1)[:, :K], np.argsort(-prx, 1)[:, :K]
    oj = np.argsort(-(ptx[:, :, None] * prx[:, None, :]).reshape(len(ptx), -1), 1)[:, :K]
    cum = lambda o, y: np.cumsum(o == y[:, None], 1) > 0
    panel = (otx[:, 0] // 16 == z['tx_gt'] // 16) & (orx[:, 0] // 16 == z['rx_gt'] // 16)
    lr = np.isin(z['tx_gt'] // 16, [0, 2]) | np.isin(z['rx_gt'] // 16, [0, 2])  # 至少一端最优面板是左/右
    return dict(tx=cum(otx, z['tx_gt']), rx=cum(orx, z['rx_gt']), joint=cum(oj, z['tx_gt'] * 64 + z['rx_gt']), panel=panel, lr=lr)


H = {m: {f: hits(tag, c, f) for f in FOLDS} for m, (tag, c) in MODELS.items()}
n_test = {f: len(H[NAMES[0]][f]['tx']) for f in FOLDS}

# 合并准确率（主表）
rows = []
for m in NAMES:
    for met in ['tx', 'rx', 'joint']:
        a = np.concatenate([H[m][f][met] for f in FOLDS]).mean(0) * 100
        rows.append(dict(model=m, metric=met, **{f'top{k}': a[k - 1] for k in range(1, K + 1)}))
pooled = pd.DataFrame(rows)
pooled.round(2).to_csv(f'{PKG}/topk_pooled.csv', index=False)
panel_pooled = {m: np.concatenate([H[m][f]['panel'] for f in FOLDS]).mean() * 100 for m in NAMES}

# 各折（附表、画图用）
per = pd.DataFrame([dict(model=m, fold=f, k=k, **{met: H[m][f][met][:, k - 1].mean() for met in ['joint', 'tx', 'rx']})
                    for m in NAMES for f in FOLDS for k in range(1, K + 1)])
per.to_csv(f'{PKG}/topk_per_fold.csv', index=False)


def run_level(a, b, side, k):
    d = np.array([(H[a][f][side][:, k - 1].mean() - H[b][f][side][:, k - 1].mean()) * 100 for f in FOLDS])
    flips = np.array(list(itertools.product([1, -1], repeat=len(d))))
    pp = np.mean(np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-12)
    ci = stats.t.interval(.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    return d, ci, pp, stats.ttest_1samp(d, 0).pvalue


out = ['# 论文用表：发送端 / 接收端单独 top-k，预测占用 vs GT 占用\n',
       f'评估：留一个 run 交叉验证（6 折）。**合并准确率** = 把 6 折的测试集合在一起计算（n = {sum(n_test.values())} 条链路，'
       '每条都由训练时没见过它所在 run 的模型预测）。各折测试链路数：' + '、'.join(f'{f} {n}' for f, n in n_test.items()) + '。\n']
KS = [1, 2, 3, 5, 10]
for side, lab in [('tx', '发送端'), ('rx', '接收端'), ('joint', '联合波束对（次要指标）')]:
    t = pooled[pooled.metric == side].set_index('model')[[f'top{k}' for k in KS]].loc[NAMES]
    t.columns = [f'top-{k}' for k in KS]
    out.append(f'\n## {lab} top-k 合并准确率（%）\n\n' + t.round(1).to_markdown())
out.append('\n## 面板对准确率（发送端、接收端面板都选对；次要指标，%）\n\n'
           + pd.Series(panel_pooled).round(1).to_frame('合并准确率').to_markdown())

out.append('\n---\n\n# 附表：各 run 的波动（以 run 为单位）\n')
for side, lab in [('tx', '发送端'), ('rx', '接收端')]:
    w = per[per.k == 1].pivot(index='model', columns='fold', values=side).loc[NAMES, FOLDS] * 100
    w['最低–最高'] = [f'{r.min():.1f}–{r.max():.1f}' for _, r in w[FOLDS].iterrows()]
    out.append(f'\n## {lab} top-1 各 run（%）\n\n' + w.round(1).to_markdown())
out.append('\n## GT − 预测的差距，以 run 为单位的配对检验（target，百分点，n = 6）\n\n'
           '| 指标 | 各 run 差值 | 均值 | 95% CI | p（置换） | p（t） |\n|---|---|---|---|---|---|')
for side, lab in [('tx', '发送端'), ('rx', '接收端')]:
    for k in [1, 5, 10]:
        d, ci, pp, pt = run_level(NAMES[0], NAMES[1], side, k)
        out.append(f'| {lab} top-{k} | {" / ".join(f"{x:.1f}" for x in d)} | {d.mean():.1f} | [{ci[0]:.1f}, {ci[1]:.1f}] | {pp:.3f} | {pt:.3f} |')
out.append('\n各 run 差值的顺序：' + ' / '.join(FOLDS) + '。n = 6 时精确置换检验最小 p = 2/64 = .031。')
out.append('\n## 预测占用：scene+target − target（百分点，以 run 为单位）\n\n| 指标 | 均值 | 95% CI | 变好的 run |\n|---|---|---|---|')
for side, lab in [('tx', '发送端'), ('rx', '接收端')]:
    d, ci, _, _ = run_level(NAMES[3], NAMES[1], side, 1)
    out.append(f'| {lab} top-1 | {d.mean():.1f} | [{ci[0]:.1f}, {ci[1]:.1f}] | {(d > 0).sum()}/6 |')

# 面板对准确率：各 run + 合并（全部 / 左右面板子集），以及 run 级对比
pp_all = pd.DataFrame({m: [H[m][f]['panel'].mean() * 100 for f in FOLDS] for m in NAMES}, index=FOLDS)
pp_all.loc['合并（按样本数加权）'] = [panel_pooled[m] for m in NAMES]
pp_all.round(1).rename_axis('fold').to_csv(f'{PKG}/panel_pair_acc_per_fold.csv')
lr_n = [int(H[NAMES[0]][f]['lr'].sum()) for f in FOLDS]
pp_lr = pd.DataFrame({m: [H[m][f]['panel'][H[m][f]['lr']].mean() * 100 if H[m][f]['lr'].any() else np.nan for f in FOLDS]
                      for m in NAMES}, index=FOLDS)
pp_lr.loc['合并（按样本数加权）'] = [np.concatenate([H[m][f]['panel'][H[m][f]['lr']] for f in FOLDS]).mean() * 100 for m in NAMES]
pp_lr.insert(0, '左右面板样本数', lr_n + [sum(lr_n)])
pp_lr.round(1).rename_axis('fold').to_csv(f'{PKG}/panel_pair_acc_LR_per_fold.csv')
crow = []
for lab, a, b in [('target: GT - 预测', 0, 1), ('scene_target: GT - 预测', 2, 3), ('预测: scene_target - target', 3, 1)]:
    d = (pp_all.loc[FOLDS, NAMES[a]] - pp_all.loc[FOLDS, NAMES[b]]).values
    flips = np.array(list(itertools.product([1, -1], repeat=len(d))))
    ci = stats.t.interval(.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    crow.append(dict(对比=lab, 均值=d.mean(), CI低=ci[0], CI高=ci[1], 正向run数=f'{(d > 0).sum()}/6',
                     p_perm=np.mean(np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-12), p_t=stats.ttest_1samp(d, 0).pvalue))
pd.DataFrame(crow).round(3).to_csv(f'{PKG}/panel_pair_contrasts.csv', index=False)
open(f'{PKG}/paper_tables.md', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
