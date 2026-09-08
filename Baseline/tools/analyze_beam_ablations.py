#!/usr/bin/env python3
"""Comparison report for the 2x2 TX-localization ablation (see
/home/admin0/.claude/plans/effervescent-shimmying-turing.md). Parses each
run's work_dirs/<name>/*.log.json for its final-epoch val metrics (mmcv logs
one JSON line per epoch with mode='val', see run_beam_ablations.sh's cells)
and prints one comparison table against the already-trained no-localization
baseline and the majority-class floor.

Usage: python tools/analyze_beam_ablations.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORK_DIRS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'work_dirs')

RUNS = [
    ('baseline (no localization)', 'carla_v2v_beam'),
    ('geo + vis (supplement, visual kept)', 'carla_v2v_beam_ablation_geo_vis'),
    ('geo + no-vis (supplement, visual dropped)', 'carla_v2v_beam_ablation_geo_novis'),
    ('no-geo + vis (replace, visual kept)', 'carla_v2v_beam_ablation_nogeo_vis'),
    ('no-geo + no-vis (replace, visual dropped)', 'carla_v2v_beam_ablation_nogeo_novis'),
]

METRIC_KEYS = ['tx_top1_acc', 'rx_top1_acc', 'tx_top3_acc', 'rx_top3_acc',
               'tx_top5_acc', 'rx_top5_acc']


def final_val_metrics(work_dir_name):
    work_dir = os.path.join(WORK_DIRS_ROOT, work_dir_name)
    val_entries = []
    for path in glob.glob(os.path.join(work_dir, '*.log.json')):
        with open(path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get('mode') == 'val':
                    val_entries.append(entry)
    if not val_entries:
        return None
    return max(val_entries, key=lambda e: e['epoch'])


def majority_baseline():
    """Same computation used ad hoc during phase-1 analysis -- always guess
    the most common TX/RX beam in the val split, no input used at all."""
    import projects.mmdet3d_plugin_carla_v2v  # noqa: F401 -- registers CarlaV2VBeamCo3SOP
    from mmcv import Config
    from mmdet.datasets import build_dataset

    cfg = Config.fromfile('projects/configs/co3sop_base/co3sop_base_carla_v2v_beam.py')
    ds = build_dataset(cfg.data.val)
    tx = np.array([ds.get_data_info(i)['gt_beam'][0] for i in range(len(ds))])
    rx = np.array([ds.get_data_info(i)['gt_beam'][1] for i in range(len(ds))])

    out = {}
    for name, arr in [('tx', tx), ('rx', rx)]:
        vals, counts = np.unique(arr, return_counts=True)
        order = np.argsort(-counts)
        out[f'{name}_top1_acc'] = counts[order[0]] / len(arr)
        top3 = vals[order[:3]]
        out[f'{name}_top3_acc'] = np.isin(arr, top3).mean()
        top5 = vals[order[:5]]
        out[f'{name}_top5_acc'] = np.isin(arr, top5).mean()
    return out


def main():
    rows = {}
    rows['majority-class baseline (no model at all)'] = majority_baseline()
    for label, work_dir_name in RUNS:
        metrics = final_val_metrics(work_dir_name)
        if metrics is None:
            rows[label] = {k: None for k in METRIC_KEYS}
            print(f'[warn] no val entries found for {work_dir_name} '
                  f'(work_dirs/{work_dir_name}/*.log.json) -- run may not have finished yet',
                  file=sys.stderr)
        else:
            rows[label] = {k: metrics.get(k) for k in METRIC_KEYS}

    df = pd.DataFrame(rows).T[METRIC_KEYS]
    pd.set_option('display.float_format', lambda x: f'{x:.1%}' if pd.notna(x) else 'N/A')
    print(df.to_string())
    print()
    print('Caveat: the localization module\'s disambiguation prior is centered exactly on the')
    print('ground-truth TX direction, so even the no-geo cells are not a strict no-GPS-at-all')
    print('test -- see beam_head.py BeamSelectionHead.loss() docstring.')


if __name__ == '__main__':
    main()
