#!/usr/bin/env python3
"""任务38: 按折重训 Stage2B 后的 6 折汇总。每折的测试 run 是分析单位（n=6），按 run 配对。
旧 pred = work_dirs/loro_cv/gated（原 Stage2B 占用），新 pred = work_dirs/loro_cv/s2b_<fold>（按折重训的 Stage2B）。
检验：符号翻转置换检验（精确，2^6 种）+ 配对 t；主要对比做 Holm。"""
import itertools
import numpy as np
import pandas as pd
from scipy import stats

ROOT = '/home/admin0/Co3SOP/Baseline/work_dirs/loro_cv'
FOLDS = ['legacy', 'seed1', 'seed2', 'seed3', 'seed4', 'seed5']
SRC = {'gt_target': 'gated', 'gt_scene_target': 'gated', 'old_pred_target': 'gated', 'old_pred_scene_target': 'gated',
       'new_pred_target': 's2b_{f}', 'new_pred_scene_target': 's2b_{f}'}
CONTRASTS = [  # 主要对比（Holm），最后一个是次要
    ('target: 新pred - 旧pred', 'new_pred_target', 'old_pred_target'),
    ('scene_target: 新pred - 旧pred', 'new_pred_scene_target', 'old_pred_scene_target'),
    ('target: GT - 新pred', 'gt_target', 'new_pred_target'),
    ('scene_target: GT - 新pred', 'gt_scene_target', 'new_pred_scene_target'),
]
SECONDARY = [('新pred: scene_target - target', 'new_pred_scene_target', 'new_pred_target')]


def acc(name, f):
    combo = name.replace('old_', '').replace('new_', '')
    z = np.load(f"{ROOT}/{SRC[name].format(f=f)}/logits/{combo}_{f}_s0.npz")
    tp, rp = z['tx_gt'] // 16, z['rx_gt'] // 16
    hit = (z['tx'].argmax(1) // 16 == tp) & (z['rx'].argmax(1) // 16 == rp)
    lr = np.isin(tp, [0, 2]) | np.isin(rp, [0, 2])
    return hit.mean(), hit[lr].mean() if lr.any() else np.nan


def test(d):
    d = np.asarray(d)
    flips = np.array(list(itertools.product([1, -1], repeat=len(d))))
    perm_p = np.mean(np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-12)
    t = stats.ttest_1samp(d, 0)
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    return d.mean(), ci, perm_p, t.pvalue, int((d > 0).sum())


def holm(p):
    p = np.asarray(p); o = np.argsort(p); out = np.empty_like(p); run = 0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i]); out[i] = min(run, 1)
    return out


per = pd.DataFrame({n: [acc(n, f)[0] for f in FOLDS] for n in SRC}, index=FOLDS)
per_lr = pd.DataFrame({n: [acc(n, f)[1] for f in FOLDS] for n in SRC}, index=FOLDS)
for lab, tab in [('全部', per), ('左右面板', per_lr)]:
    print(f'\n== 各折面板对准确率（{lab}）'); print((tab * 100).round(1).to_string())
    print('均值', (tab.mean() * 100).round(1).to_dict())
rows = []
for lab, a, b in CONTRASTS + SECONDARY:
    m, ci, pp, tp_, k = test((per[a] - per[b]) * 100)
    rows.append(dict(对比=lab, 均值=m, CI低=ci[0], CI高=ci[1], 正向run数=f'{k}/6', p_perm=pp, p_t=tp_))
res = pd.DataFrame(rows)
res.loc[:3, 'Holm_perm'] = holm(res.p_perm[:4]); res.loc[:3, 'Holm_t'] = holm(res.p_t[:4])
print('\n== 对比（百分点，run 配对，n=6）'); print(res.round(3).to_string(index=False))
per.to_csv('per_run_all.csv'); per_lr.to_csv('per_run_lr.csv'); res.to_csv('contrasts.csv', index=False)
