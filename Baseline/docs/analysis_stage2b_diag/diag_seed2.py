#!/usr/bin/env python3
"""任务39: seed2 折为什么提升小。读 diag_target.py 对各折新 Stage2B 的输出 data/diag_loro_<fold>.csv
（每折只取自己的测试 run），加上 GT 占用 beam 头的命中，按 link / 方位 / IoU 拆开 GT−新pred 的差距。"""
import numpy as np
import pandas as pd

FOLDS = ['legacy', 'seed1', 'seed2', 'seed3', 'seed4', 'seed5']
GATED = '/home/admin0/Co3SOP/Baseline/work_dirs/loro_cv/gated/logits'


def gt_hits(f):
    z = np.load(f'{GATED}/gt_target_{f}_s0.npz')
    ok = (z['tx'].argmax(1) // 16 == z['tx_gt'] // 16) & (z['rx'].argmax(1) // 16 == z['rx_gt'] // 16)
    return {(int(s), str(l)): bool(o) for s, l, o in zip(z['sample_id'], z['link'], ok)}


dfs = []
for f in FOLDS:
    d = pd.read_csv(f'data/diag_loro_{f}.csv')
    d = d[d.run == f].copy()
    g = gt_hits(f)
    d['hit_gt'] = [g.get((s, l), np.nan) for s, l in zip(d.sample_id, d.link)]
    dfs.append(d)
d = pd.concat(dfs)
d['iou0'] = d.iou3d.fillna(0)
d['sector'] = pd.cut(d.gt_bearing, [-180, -135, -45, 45, 135, 180], labels=['后', '右侧', '前', '左侧', '后2']) \
    .astype(str).replace({'后2': '后', 'nan': '无GT'})

print('== 各折测试 run × link：target 占用质量和 beam 命中')
print(d.groupby(['run', 'link']).agg(n=('iou0', 'size'), iou=('iou0', 'mean'), iou0_frac=('iou0', lambda x: (x < .01).mean()),
                                     hit_gt=('hit_gt', 'mean'), hit_pred=('hit_pred_target', 'mean')).round(3).to_string())
for link in ['TX_CAR1', 'TX_CAR2']:
    x = d[d.link == link]
    print(f'\n== {link} 相对接收车的方位分布'); print(pd.crosstab(x.run, x.sector, normalize='index').round(3).to_string())

s = d[d.run == 'seed2']
c1 = s[s.link == 'TX_CAR1']
print('\n== seed2 TX_CAR1 按 IoU 分箱')
print(c1.groupby(pd.cut(c1.iou0, [-.01, .01, .2, .4, .6, 1])).agg(n=('iou0', 'size'), hit_gt=('hit_gt', 'mean'),
                                                                  hit_pred=('hit_pred_target', 'mean')).round(3).to_string())
miss = c1[c1.iou0 < .01]
nz = miss[miss.pred_vox > 0]
print(f'\nIoU≈0: {len(miss)} 个（预测为空 {(miss.pred_vox == 0).sum()}，有块但错位 {len(nz)}）；真实方位 {miss.sector.value_counts().to_dict()}')
print('错位块方位（0=前）:', pd.cut(np.degrees(np.arctan2(nz.blob_y, nz.blob_x)), [-180, -135, -45, 45, 135, 180]).value_counts().sort_index().to_dict(),
      '距离中位 %.1f m' % np.hypot(nz.blob_x, nz.blob_y).median())
gap = s.hit_gt.astype(float) - s.hit_pred_target.astype(float)
print('seed2 GT−新pred 差距 %.1f 点，其中 TX_CAR1 IoU≈0 样本贡献 %.1f 点' % (gap.mean() * 100, gap[(s.link == 'TX_CAR1') & (s.iou0 < .01)].sum() / len(s) * 100))
