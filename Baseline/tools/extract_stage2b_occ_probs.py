#!/usr/bin/env python3
"""Stage 3 Step 1: offline extraction + canonicalization of GT and frozen-
Stage-2B-Predicted Scene/Target Occupancy for every (sample_id, link) Stage
2B/Beam already uses -- no camera images, no gradients, run once.

Locks onto Stage 2B's ACTUAL train/val split (Audit 1c) by literally
building CarlaV2VTargetOccCo3SOP's train/val dataset objects and reading
their .data_infos -- never re-derives the 80/20 cut. Runs the frozen
checkpoint once per (sample_id,link) (1564 total, NOT 783+1564 -- Scene and
Target both come out of a single TargetAwareOccHead.forward() call, see
target_occ_head.py:75-93), saving Scene from whichever of a sample_id's two
link-forwards runs first and using the OTHER link's Scene output only for an
equality assertion (confirms the Scene branch is genuinely role-independent,
never both saved).

GT and Predicted canonicalization happen in this same pass, from the same
per-sample ego_yaw_deg lookup (rear_vehicle_log.csv's vehicle_2_yaw at that
sample_id) -- structurally rules out the two ever using a different yaw.

Usage: python tools/extract_stage2b_occ_probs.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import projects.mmdet3d_plugin  # noqa: F401
import projects.mmdet3d_plugin_carla_v2v  # noqa: F401
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from projects.mmdet3d_plugin_carla_v2v.datasets.occ_canonicalize import canonicalize_to_world_axes

CONFIG = 'projects/configs/co3sop_base/co3sop_base_carla_v2v_target_occ.py'
CKPT = 'work_dirs/carla_v2v_target_occ_phaseB/epoch_12.pth'
VEHICLE_LOG = '/home/admin0/data/run_20260904_142344_bigcar80/rear_vehicle_log.csv'
BEAM_CSV = '/home/admin0/carla_V2V/channel_analysis/channel_summary_from_carla_20260904_145458.csv'
OUT_DIR = '/home/admin0/carla_V2V/occupancy_prediction_beam/beam_occ_canonical'


def one_hot_scene(gt_occ):
    """gt_occ: [X,Y,Z] uint8 in {0,1,2} (empty/road/vehicle). -> [3,X,Y,Z] float32 one-hot."""
    out = np.zeros((3,) + gt_occ.shape, dtype=np.float32)
    for c in range(3):
        out[c] = (gt_occ == c).astype(np.float32)
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='cap total (sample_id,link) pairs processed, for a quick smoke run')
    args = parser.parse_args()

    cfg = Config.fromfile(CONFIG)
    # IMPORTANT: cfg.data.train's pipeline includes PhotoMetricDistortionMultiViewImage
    # (random color jitter) -- fine for actual training, but this extraction
    # must be fully deterministic (the whole point is a frozen, reusable
    # feature dump). Force the deterministic test_pipeline for the train
    # split too, not just val. Caught by this script's own cross-link Scene
    # equality assertion failing on the first smoke run (max diff 15.07,
    # not ~0) -- looked like a role-leakage bug at first, was actually two
    # independently-randomly-augmented forward passes of the same sample_id.
    train_data_cfg = cfg.data.train.copy()
    train_data_cfg['pipeline'] = cfg.test_pipeline
    train_ds = build_dataset(train_data_cfg)
    val_ds = build_dataset(cfg.data.val)

    all_items = []  # (dataset, index, sample_id, link)
    seen = set()
    for ds, split_name in [(train_ds, 'train'), (val_ds, 'val')]:
        for i, info in enumerate(ds.data_infos):
            sid, link = int(info[2]), info[3]
            key = (sid, link)
            assert key not in seen, f'duplicate (sample_id,link) across train/val: {key}'
            seen.add(key)
            all_items.append((ds, i, sid, link, split_name))

    if args.limit:
        all_items = all_items[:args.limit]
    print(f'[audit 1c] total (sample_id,link) pairs: {len(all_items)}')
    train_sids = sorted(set(int(info[2]) for info in train_ds.data_infos))
    val_sids = sorted(set(int(info[2]) for info in val_ds.data_infos))
    print(f'[audit 1c] train sample_ids: {train_sids[0]}-{train_sids[-1]} (n={len(train_sids)}), '
          f'val: {val_sids[0]}-{val_sids[-1]} (n={len(val_sids)})')
    assert set(train_sids).isdisjoint(val_sids)
    missing = set(range(1, 784)) - set(train_sids) - set(val_sids)
    print(f'[audit 1c] sample_ids missing from both splits (1..783): {sorted(missing)}')

    veh_df = pd.read_csv(VEHICLE_LOG).set_index('sample_id')

    beam_df = pd.read_csv(BEAM_CSV)
    beam_df = beam_df[beam_df['tx_name'].isin(['TX_CAR1', 'TX_CAR2'])]

    def print_class_dist(sids, name):
        print(f'\n[audit 2] {name} beam class distribution:')
        for link in ('TX_CAR1', 'TX_CAR2'):
            sub = beam_df[(beam_df['tx_name'] == link) & (beam_df['sample_id'].isin(sids))]
            for col in ('optimal_tx_beam_idx', 'optimal_rx_beam_idx'):
                vc = sub[col].value_counts(normalize=True)
                print(f'  {link} {col}: n={len(sub)} n_unique={sub[col].nunique()} '
                      f'majority={vc.index[0]}({vc.iloc[0]:.3f}) top3={vc.head(3).sum():.3f}')
    print_class_dist(train_sids, 'TRAIN')
    print_class_dist(val_sids, 'VAL')

    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, CKPT, map_location='cpu', strict=False)
    model = model.cuda().eval()

    os.makedirs(OUT_DIR, exist_ok=True)
    scene_logits_by_sid = {}  # sid -> pre-softmax occ_preds[-1] logits, for the cross-link equality assertion
    n_written_scene = 0
    n_written_target = 0
    max_scene_diff = 0.0

    for idx, (ds, i, sid, link, split_name) in enumerate(all_items):
        sample = ds[i]
        gt_occ = np.asarray(sample['gt_occ'])       # [X,Y,Z] uint8 {0,1,2}
        gt_target = np.asarray(sample['gt_target'])  # [X,Y,Z] {0,1}
        ego_yaw = float(veh_df.loc[sid, 'vehicle_2_yaw'])

        sid_dir = os.path.join(OUT_DIR, f'{sid:06d}')
        os.makedirs(sid_dir, exist_ok=True)

        with torch.no_grad():
            batch = scatter(collate([sample], samples_per_gpu=1), [0])[0]
            img_feats = model.extract_feat(img=batch['img'], img_metas=batch['img_metas'])
            preds = model.pts_bbox_head(img_feats, batch['img_metas'])
            scene_logits = preds['occ_preds'][-1][0].cpu().numpy()      # [3,X,Y,Z]
            target_logits = preds['target_preds'][-1][0].cpu().numpy()  # [2,X,Y,Z]

        # cross-link Scene equality check (role-independence assertion)
        if sid in scene_logits_by_sid:
            diff = float(np.abs(scene_logits - scene_logits_by_sid[sid]).max())
            max_scene_diff = max(max_scene_diff, diff)
        else:
            scene_logits_by_sid[sid] = scene_logits
            # GT scene + Pred scene canonicalized and saved once per sample_id
            gt_scene_1h = one_hot_scene(gt_occ)
            gt_scene_canon = canonicalize_to_world_axes(gt_scene_1h, ego_yaw, order=0, is_scene=True)
            np.savez_compressed(os.path.join(sid_dir, 'gt_scene.npz'),
                                 onehot=gt_scene_canon.astype(np.uint8))

            pred_scene_probs = F.softmax(torch.from_numpy(scene_logits), dim=0).numpy()
            pred_scene_canon = canonicalize_to_world_axes(pred_scene_probs, ego_yaw, order=1, is_scene=True)
            np.savez_compressed(os.path.join(sid_dir, 'pred_scene.npz'),
                                 probs=pred_scene_canon.astype(np.float16))
            n_written_scene += 1

        # GT + Pred target, per link
        gt_target_1c = gt_target.astype(np.float32)[None]  # [1,X,Y,Z]
        gt_target_canon = canonicalize_to_world_axes(gt_target_1c, ego_yaw, order=0, is_scene=False)
        np.savez_compressed(os.path.join(sid_dir, f'{link}_gt_target.npz'),
                             mask=gt_target_canon.astype(np.uint8))

        pred_target_prob = F.softmax(torch.from_numpy(target_logits), dim=0).numpy()[1:2]  # [1,X,Y,Z]
        pred_target_canon = canonicalize_to_world_axes(pred_target_prob, ego_yaw, order=1, is_scene=False)
        np.savez_compressed(os.path.join(sid_dir, f'{link}_pred_target.npz'),
                             prob=pred_target_canon.astype(np.float16))
        n_written_target += 1

        if (idx + 1) % 200 == 0:
            print(f'  [{idx+1}/{len(all_items)}]', file=sys.stderr)

    print(f'\n[done] scene volumes written: {n_written_scene} (expect {len(train_sids)+len(val_sids)})')
    print(f'[done] target volumes written: {n_written_target} (expect {len(all_items)})')
    print(f'[audit] max |scene_logits diff| between the two links of the same sample_id: {max_scene_diff:.6f} '
          f'(expect ~0 -- confirms Scene branch is role-independent)')
    assert max_scene_diff < 1e-3, 'Scene branch output differs between links of the same sample -- role leakage?'


if __name__ == '__main__':
    main()
