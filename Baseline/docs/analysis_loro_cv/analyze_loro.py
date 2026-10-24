#!/usr/bin/env python3
"""任务35: 留一个 run 交叉验证结果分析。每折的测试 run 是分析单位（n<=6），组合之间按 run 配对。
用法: python analyze_loro.py <tag> [seeds]   例: python analyze_loro.py cc 0
输出 data/<tag>_per_run.csv（run × 组合 × 子集）和 data/<tag>_contrasts.csv。"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = '/home/admin0/Co3SOP/Baseline/work_dirs/loro_cv'
FOLDS = ['legacy', 'seed1', 'seed2', 'seed3', 'seed4', 'seed5']
COMBOS = ['gt_target', 'gt_scene_target', 'pred_target', 'pred_scene_target']
LOS_CSV = '/home/admin0/carla_V2V/channel_analysis/dataset_full_merged_20260929_regen.csv'
CONTRASTS = [  # (名字, A, B) -> A - B
    ('GT: scene_target - target', 'gt_scene_target', 'gt_target'),
    ('Pred: scene_target - target', 'pred_scene_target', 'pred_target'),
    ('target: GT - Pred', 'gt_target', 'pred_target'),
    ('scene_target: GT - Pred', 'gt_scene_target', 'pred_scene_target'),
]


def load(tag, seeds):
    beam = pd.read_csv(LOS_CSV)
    los = {(int(r.sample_id), r.tx_name): bool(r.is_los) for r in beam.itertuples(index=False)
           if r.tx_name in ('TX_CAR1', 'TX_CAR2')}
    rows = []
    for c, f, s in itertools.product(COMBOS, FOLDS, seeds):
        p = os.path.join(ROOT, tag, 'logits', f'{c}_{f}_s{s}.npz')
        if not os.path.exists(p):
            continue
        z = np.load(p)
        tp, rp = z['tx_gt'] // 16, z['rx_gt'] // 16
        hit = (z['tx'].argmax(1) // 16 == tp) & (z['rx'].argmax(1) // 16 == rp)
        is_los = np.array([los[(int(a), str(b))] for a, b in zip(z['sample_id'], z['link'])])
        lr = np.isin(tp, [0, 2]) | np.isin(rp, [0, 2])  # RIGHT=0, LEFT=2
        maj_pair = pd.Series(list(zip(tp, rp))).value_counts().index[0]
        maj = np.array([(a, b) == maj_pair for a, b in zip(tp, rp)])
        for sub, mask in [('all', np.ones_like(hit)), ('LOS_nonLR', is_los & ~lr), ('LR', lr), ('NLOS', ~is_los)]:
            if mask.sum() == 0:
                continue
            rows.append(dict(combo=c, run=f, seed=s, subset=sub, n=int(mask.sum()),
                             acc=hit[mask].mean(), majority_in_test=maj[mask].mean()))
    return pd.DataFrame(rows)


def sign_flip_p(d):
    d = np.asarray(d)
    obs = abs(d.mean())
    flips = np.array(list(itertools.product([1, -1], repeat=len(d))))
    return float((np.abs((flips * d).mean(1)) >= obs - 1e-12).mean())


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'cc'
    seeds = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0]
    df = load(tag, seeds)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    df.to_csv(os.path.join(out, f'{tag}_per_run_long.csv'), index=False)
    m = df.groupby(['subset', 'combo', 'run']).agg(acc=('acc', 'mean'), n=('n', 'first')).reset_index()

    for sub in ['all', 'LOS_nonLR', 'LR', 'NLOS']:
        t = m[m.subset == sub].pivot(index='combo', columns='run', values='acc').reindex(COMBOS)
        n = m[m.subset == sub].groupby('run').n.first()
        print(f'\n== {sub}  (每个 run 的行数: {n.to_dict()})')
        t['mean_of_runs'] = t.mean(1)
        print(t.round(3).to_string())

    res = []
    for sub in ['all', 'LOS_nonLR', 'LR', 'NLOS']:
        t = m[m.subset == sub].pivot(index='run', columns='combo', values='acc')
        for name, a, b in CONTRASTS:
            if a not in t or b not in t:
                continue
            d = (t[a] - t[b]).dropna()
            if len(d) < 2:
                continue
            se = d.std(ddof=1) / np.sqrt(len(d))
            tc = stats.t.ppf(.975, len(d) - 1)
            res.append(dict(subset=sub, contrast=name, n_runs=len(d), mean=d.mean(), ci_lo=d.mean() - tc * se,
                            ci_hi=d.mean() + tc * se, n_pos=int((d > 0).sum()), p_perm=sign_flip_p(d),
                            p_t=stats.ttest_1samp(d, 0).pvalue if d.std() > 0 else np.nan,
                            per_run=' '.join(f'{r}:{v:+.3f}' for r, v in d.items())))
    r = pd.DataFrame(res)
    if r.empty:
        print('\n(还没有可以配对的组合)')
        return
    r.to_csv(os.path.join(out, f'{tag}_contrasts.csv'), index=False)
    print('\n== 对比（按 run 配对，均值 / 95% t-CI / 正的 run 数 / 置换 p）')
    print(r.drop(columns='per_run').round(3).to_string(index=False))
    print('\n' + r[r.subset == 'all'][['contrast', 'per_run']].to_string(index=False))


if __name__ == '__main__':
    main()
