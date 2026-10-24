#!/usr/bin/env python3
"""任务35: leave-one-run-out (LORO) CV split files for the occ->beam sweep.

One fold per source run (bucket = sample_id // 1000: legacy=0, route3 seed1..5 = 1..5):
  test  = every (sample_id, link) of the held-out run. Legacy fold: only
          sid >= LEGACY_STAGE2B_MAX+1 (627-783) -- Stage 2B was trained on legacy
          sid 2-626, so those frames would leak into pred-occupancy test inputs.
  val   = inner early-stopping split: the last VAL_FRAC of each TRAINING run's
          sample_ids, with GAP sample_ids dropped before it (consecutive frames
          are near-duplicates). The test run never takes part in model selection.
  train = the rest of the training runs.
--drop-stage2b-train additionally removes legacy sid 2-626 from train/val
(sensitivity check: those frames carry train-quality Stage 2B predictions).

Usage:
  python tools/loro_cv.py --out work_dirs/loro_cv/splits [--drop-stage2b-train]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from projects.mmdet3d_plugin_carla_v2v.datasets.occ_beam_dataset import _build_split_infos  # noqa: E402

RUN_NAMES = {0: 'legacy', 1: 'seed1', 2: 'seed2', 3: 'seed3', 4: 'seed4', 5: 'seed5'}
LEGACY_STAGE2B_MAX = 626


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--gap', type=int, default=10, help='sample_ids dropped between train and inner val, per run')
    ap.add_argument('--drop-stage2b-train', action='store_true')
    a = ap.parse_args()

    universe = sorted(set(_build_split_infos('train')) | set(_build_split_infos('val')))
    by_run = {}
    for sid, link in universe:
        by_run.setdefault(sid // 1000, []).append((sid, link))
    assert sorted(by_run) == sorted(RUN_NAMES), sorted(by_run)

    os.makedirs(a.out, exist_ok=True)
    summary = {}
    for test_b, test_name in RUN_NAMES.items():
        test = by_run[test_b]
        if test_b == 0:
            test = [k for k in test if k[0] > LEGACY_STAGE2B_MAX]
        train, val = [], []
        for b, rows in by_run.items():
            if b == test_b:
                continue
            if b == 0 and a.drop_stage2b_train:
                rows = [k for k in rows if k[0] > LEGACY_STAGE2B_MAX]
            sids = sorted({s for s, _ in rows})
            n_val = max(1, round(len(sids) * a.val_frac))
            val_start = sids[-n_val]
            gap_start = sids[max(0, len(sids) - n_val - a.gap)]
            train += [k for k in rows if k[0] < gap_start]
            val += [k for k in rows if k[0] >= val_start]
        assert not (set(train) & set(val)) and not (set(train) & set(test)) and not (set(val) & set(test))
        split = {'fold': test_name, 'test_bucket': test_b,
                 'train': [list(k) for k in train], 'val': [list(k) for k in val], 'test': [list(k) for k in test]}
        with open(os.path.join(a.out, f'fold_{test_name}.json'), 'w') as f:
            json.dump(split, f)
        summary[test_name] = {'train': len(train), 'val': len(val), 'test': len(test)}
        print(f'fold {test_name:7s} train={len(train):5d} val={len(val):4d} test={len(test):5d}')
    with open(os.path.join(a.out, 'summary.json'), 'w') as f:
        json.dump({'args': vars(a), 'folds': summary}, f, indent=2)


if __name__ == '__main__':
    main()
