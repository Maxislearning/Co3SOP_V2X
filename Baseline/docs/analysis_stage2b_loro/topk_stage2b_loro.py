#!/usr/bin/env python3
"""任务41: 6 折的波束对 top-1..10 准确率（联合 = tx/rx softmax 外积后在 64×64 个波束对里排序，同 analysis_v4_by_run）。
旧 pred = gated（原 Stage2B），新 pred = s2b_<fold>（按折重训的 Stage2B）。每折测试 run 一个数，再对 6 折取均值。"""
import numpy as np
import pandas as pd

ROOT = '/home/admin0/Co3SOP/Baseline/work_dirs/loro_cv'
FOLDS = ['legacy', 'seed1', 'seed2', 'seed3', 'seed4', 'seed5']
MODELS = {'GT target': ('gated', 'gt_target'), '旧pred target': ('gated', 'pred_target'), '新pred target': ('s2b_{f}', 'pred_target'),
          'GT scene_target': ('gated', 'gt_scene_target'), '旧pred scene_target': ('gated', 'pred_scene_target'),
          '新pred scene_target': ('s2b_{f}', 'pred_scene_target')}
K = 10


def softmax(x):
    x = x - x.max(1, keepdims=True); e = np.exp(x); return e / e.sum(1, keepdims=True)


def topk(tag, combo, f):
    z = np.load(f'{ROOT}/{tag.format(f=f)}/logits/{combo}_{f}_s0.npz')
    ptx, prx = softmax(z['tx']), softmax(z['rx'])
    pj = (ptx[:, :, None] * prx[:, None, :]).reshape(len(ptx), -1)
    oj = np.argsort(-pj, 1)[:, :K]
    hit = oj == (z['tx_gt'] * 64 + z['rx_gt'])[:, None]
    otx, orx = np.argsort(-ptx, 1)[:, :K], np.argsort(-prx, 1)[:, :K]
    cum = lambda h: np.cumsum(h, 1).clip(0, 1).mean(0)
    return cum(hit), cum(otx == z['tx_gt'][:, None]), cum(orx == z['rx_gt'][:, None])


rows = []
for m, (tag, combo) in MODELS.items():
    for f in FOLDS:
        j, t, r = topk(tag, combo, f)
        for k in range(K):
            rows.append(dict(model=m, fold=f, k=k + 1, joint=j[k], tx=t[k], rx=r[k]))
df = pd.DataFrame(rows)
df.to_csv('topk_per_fold.csv', index=False)
pd.set_option('display.width', 250)
for met in ['joint', 'tx', 'rx']:
    print(f'\n== {met} top-k（6 折均值，%）')
    print((df.groupby(['model', 'k'])[met].mean().unstack() * 100).reindex(MODELS).round(1).to_string())
print('\n== 联合 top-1 / top-10 各折（%）')
for k in [1, 10]:
    print(f'-- top-{k}'); print((df[df.k == k].pivot(index='fold', columns='model', values='joint') * 100)[list(MODELS)].round(1).to_string())
