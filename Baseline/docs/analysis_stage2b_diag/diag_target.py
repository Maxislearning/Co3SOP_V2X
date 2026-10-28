#!/usr/bin/env python3
"""任务36: Stage2B 预测 target occupancy 与 GT 的逐行对比，并和留一个 run 交叉验证（gated, s0）的 beam 结果对应。
输出 data/target_diag_rows.csv（每个 (sample_id, link) 一行）。
用法: python diag_target.py [occ_dir] [loro_logits_dir] [out_csv]（任务37：对比重训后的 Stage2B）"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import ndimage

OCC = sys.argv[1] if len(sys.argv) > 1 else '/home/admin0/carla_V2V/occupancy_prediction_beam/beam_occ_native'
BEAM_CSV = '/home/admin0/carla_V2V/channel_analysis/dataset_full_merged_20260929_regen.csv'
LORO = sys.argv[2] if len(sys.argv) > 2 else '/home/admin0/Co3SOP/Baseline/work_dirs/loro_cv/gated/logits'
OUT_CSV = sys.argv[3] if len(sys.argv) > 3 else 'data/target_diag_rows.csv'
VOX = 0.4          # m, point_cloud_range [-40,40] / 200
CENTER = 100       # 接收车（本车）在网格中心
NEAR_M = 2.0       # 预测概率落在 GT 框周围 2 m 以内算“在目标上”
PANELS = ['RIGHT', 'REAR', 'LEFT', 'FRONT']
FOLDS = ['legacy', 'seed1', 'seed2', 'seed3', 'seed4', 'seed5']


def centroid(w):
    idx = np.argwhere(w > 0)
    if len(idx) == 0:
        return np.nan, np.nan
    ww = w[idx[:, 0], idx[:, 1]]
    c = (idx * ww[:, None]).sum(0) / ww.sum()
    return (c[0] - CENTER + 0.5) * VOX, (c[1] - CENTER + 0.5) * VOX


def one(key):
    sid, link = key
    d = os.path.join(OCC, f'{sid:06d}')
    gt = np.load(os.path.join(d, f'{link}_gt_target.npz'))['mask'][0] > 0          # [X,Y,Z]
    pr = np.load(os.path.join(d, f'{link}_pred_target.npz'))['prob'][0].astype(np.float32)
    gt2, pr2 = gt.any(2), pr.max(2)                                                  # BEV
    pb = pr > 0.5
    near = ndimage.binary_dilation(gt2, iterations=int(NEAR_M / VOX)) if gt2.any() else gt2
    gx, gy = centroid(gt2.astype(np.float32))
    px, py = centroid(np.where(pr2 > 0.5, pr2, 0))
    # 预测里最大的连通块（排除散点）的质心
    lab, n = ndimage.label(pr2 > 0.5)
    if n:
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        big = lab == (1 + int(np.argmax(sizes)))
        bx, by = centroid(big.astype(np.float32))
    else:
        bx = by = np.nan
    inter = (pb & gt).sum()
    union = (pb | gt).sum()
    mass = pr2.sum()
    return dict(sample_id=sid, link=link, gt_vox=int(gt.sum()), pred_vox=int(pb.sum()), n_blobs=int(n),
                iou3d=inter / union if union else np.nan,
                recall_bev=(pr2[gt2] > 0.5).mean() if gt2.any() else np.nan,
                mass_near_frac=pr2[near].sum() / mass if mass > 0 else np.nan,
                gt_x=gx, gt_y=gy, pred_x=px, pred_y=py, blob_x=bx, blob_y=by)


def main():
    beam = pd.read_csv(BEAM_CSV)
    beam = beam[beam.tx_name.isin(['TX_CAR1', 'TX_CAR2'])]
    keys = [(int(s), l) for s, l in zip(beam.sample_id, beam.tx_name)
            if os.path.exists(os.path.join(OCC, f'{int(s):06d}', f'{l}_pred_target.npz'))]
    with Pool(6) as p:
        rows = p.map(one, keys, chunksize=32)
    df = pd.DataFrame(rows)
    b = beam.set_index(['sample_id', 'tx_name'])
    df['is_los'] = [bool(b.loc[(s, l), 'is_los']) for s, l in zip(df.sample_id, df.link)]
    df['tx_panel'] = [int(b.loc[(s, l), 'optimal_tx_beam_idx']) // 16 for s, l in zip(df.sample_id, df.link)]
    df['rx_panel'] = [int(b.loc[(s, l), 'optimal_rx_beam_idx']) // 16 for s, l in zip(df.sample_id, df.link)]
    df['run'] = [FOLDS[s // 1000] for s in df.sample_id]
    for pre, x, y in [('gt', 'gt_x', 'gt_y'), ('pred', 'pred_x', 'pred_y'), ('blob', 'blob_x', 'blob_y')]:
        df[f'{pre}_dist'] = np.hypot(df[x], df[y])
        df[f'{pre}_bearing'] = np.degrees(np.arctan2(df[y], df[x]))   # 0 = +x（前），90 = +y
    df['centroid_err_m'] = np.hypot(df.pred_x - df.gt_x, df.pred_y - df.gt_y)
    df['blob_err_m'] = np.hypot(df.blob_x - df.gt_x, df.blob_y - df.gt_y)
    df['bearing_err_deg'] = ((df.pred_bearing - df.gt_bearing + 180) % 360 - 180).abs()
    # 和交叉验证 beam 结果对应（每行只属于一个测试折）
    for c in ['gt_target', 'pred_target', 'pred_scene_target', 'gt_scene_target']:
        hit = {}
        for f in FOLDS:
            fp = os.path.join(LORO, f'{c}_{f}_s0.npz')
            if not os.path.exists(fp):
                continue
            z = np.load(fp)
            ok = (z['tx'].argmax(1) // 16 == z['tx_gt'] // 16) & (z['rx'].argmax(1) // 16 == z['rx_gt'] // 16)
            txp = z['tx'].argmax(1) // 16
            for s, l, o, t in zip(z['sample_id'], z['link'], ok, txp):
                hit[(int(s), str(l))] = (bool(o), int(t))
        df[f'hit_{c}'] = [hit.get((s, l), (np.nan, -1))[0] for s, l in zip(df.sample_id, df.link)]
        df[f'txpanel_{c}'] = [hit.get((s, l), (np.nan, -1))[1] for s, l in zip(df.sample_id, df.link)]
    os.makedirs(os.path.dirname(OUT_CSV) or '.', exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(len(df), 'rows ->', OUT_CSV)


if __name__ == '__main__':
    main()
