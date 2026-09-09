#!/usr/bin/env python3
"""LOS vs NLOS breakdown of beam accuracy, comparing configs with and
without the exact-GPS geometry input (visual branch kept in both, per user
request -- the *_novis cells are out of scope here). Reuses already-trained
checkpoints, no retraining: just replays inference on the val set.

TX_CAR2 is 100% LOS in this data (see carla_V2V's channel_summary CSV); all
NLOS samples come from the TX_CAR1 link, so the NLOS subset is small (~13-14
val samples) -- read the NLOS numbers as directional, not statistically solid.

Usage: python tools/analyze_los_nlos.py
"""
import os
import sys

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import projects.mmdet3d_plugin_carla_v2v as plugin  # noqa: F401
from mmdet.datasets import build_dataset
from mmdet3d.models import build_model

RUNS = [
    ('baseline (geo+vis, no localization)', 'co3sop_base_carla_v2v_beam.py', 'carla_v2v_beam'),
    ('geo+vis (geo kept, +localization)', 'co3sop_base_carla_v2v_beam_ablation_geo_vis.py', 'carla_v2v_beam_ablation_geo_vis'),
    ('nogeo+vis (geo replaced by localization)', 'co3sop_base_carla_v2v_beam_ablation_nogeo_vis.py', 'carla_v2v_beam_ablation_nogeo_vis'),
]


def evaluate_run(config_name, work_dir_name):
    cfg = Config.fromfile(f'projects/configs/co3sop_base/{config_name}')
    ds = build_dataset(cfg.data.val)
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, f'work_dirs/{work_dir_name}/epoch_12.pth', map_location='cpu', strict=False)
    model = model.cuda().eval()

    occ_results = []
    with torch.no_grad():
        for i in range(len(ds)):
            batch = scatter(collate([ds[i]], samples_per_gpu=1), [0])[0]
            result = model(return_loss=False, rescale=True, **batch)
            occ_results.extend(result['evaluation'])
    return ds.evaluate(occ_results)


def main():
    rows = {}
    for label, config_name, work_dir_name in RUNS:
        print(f'evaluating: {label} ...', file=sys.stderr)
        rows[label] = evaluate_run(config_name, work_dir_name)

    keys = ['n_los', 'n_nlos',
            'los_tx_top1_acc', 'los_rx_top1_acc', 'los_tx_top3_acc', 'los_rx_top3_acc',
            'los_tx_top5_acc', 'los_rx_top5_acc',
            'nlos_tx_top1_acc', 'nlos_rx_top1_acc', 'nlos_tx_top3_acc', 'nlos_rx_top3_acc',
            'nlos_tx_top5_acc', 'nlos_rx_top5_acc']

    import pandas as pd
    df = pd.DataFrame(rows).T
    df = df[[k for k in keys if k in df.columns]]
    for c in df.columns:
        if c not in ('n_los', 'n_nlos'):
            df[c] = df[c].map(lambda x: f'{x:.1%}' if pd.notna(x) else 'N/A')
    print(df.to_string())


if __name__ == '__main__':
    main()
