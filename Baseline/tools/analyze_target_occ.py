#!/usr/bin/env python3
"""Stage 2B Stop Gate report. Replays the trained checkpoint on the val set
(same "load checkpoint, manual loop" pattern as analyze_los_nlos.py) and
reports, split by FRONT/REAR:
  - Target IoU/Dice/precision/recall
  - Predicted-mask centroid -> range/azimuth/XYZ error vs. the known TX
    position (tx_rx_geometry -- eval-only, never touches the model forward)
  - Scene mIoU/road IoU/vehicle IoU, with vehicle recall split into "core 3
    vehicles" vs "background vehicles" (recomputed at eval time from
    rear_vehicle_log.csv/background_vehicles_log.csv via fill_obb/
    world_to_ego -- the saved GT only has one merged Vehicle class)
  - Anti-shortcut checks: always-zero, always-average-training-mask-per-role,
    camera-shuffle (same checkpoint, img tensor's camera dim shuffled before
    the forward call)
  - A few BEV visualizations (pred vs GT)

Usage: python tools/analyze_target_occ.py [--checkpoint work_dirs/carla_v2v_target_occ_phaseB/epoch_12.pth]
"""
import argparse
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
sys.path.insert(0, '/home/admin0/carla_V2V/src')
import projects.mmdet3d_plugin_carla_v2v as plugin  # noqa: F401
from mmdet.datasets import build_dataset
from mmdet3d.models import build_model
from generate_occupancy_gt import (  # noqa: E402
    NX, NY, NZ, LABEL_VEHICLE, VOXEL, X_RANGE, Y_RANGE, Z_RANGE, fill_obb, world_to_ego,
)

CONFIG = 'projects/configs/co3sop_base/co3sop_base_carla_v2v_target_occ.py'
VEHICLE_LOG = '/home/admin0/data/run_20260904_142344_bigcar80/rear_vehicle_log.csv'
BACKGROUND_LOG = '/home/admin0/data/run_20260904_142344_bigcar80/background_vehicles_log.csv'
LINK_VEHICLE = {'TX_CAR1': 1, 'TX_CAR2': 3}


def iou(tp, p, g, eps=1e-6):
    return float(tp / (p + g - tp + eps))


def build_reference_masks(sample_id):
    """Ego(=vehicle_2)-frame reference masks for the 3 core vehicles vs
    background vehicles, recomputed from source metadata (not read off the
    saved GT, which merges everything into one Vehicle class)."""
    veh_row = VEH_DF.loc[sample_id]
    ego_origin = np.array([veh_row.vehicle_2_x, veh_row.vehicle_2_y, veh_row.vehicle_2_z], dtype=np.float32)
    ego_yaw = float(veh_row.vehicle_2_yaw)

    core = np.zeros((NX, NY, NZ), dtype=np.uint8)
    fill_obb(core, np.zeros(3, dtype=np.float32),
              veh_row.vehicle_2_length / 2, veh_row.vehicle_2_width / 2, veh_row.vehicle_2_height / 2,
              0.0, LABEL_VEHICLE)
    for vn in (1, 3):
        w = np.array([[veh_row[f'vehicle_{vn}_x'], veh_row[f'vehicle_{vn}_y'], veh_row[f'vehicle_{vn}_z']]], dtype=np.float32)
        e = world_to_ego(w, ego_origin, ego_yaw)[0]
        fill_obb(core, e, veh_row[f'vehicle_{vn}_length'] / 2, veh_row[f'vehicle_{vn}_width'] / 2,
                  veh_row[f'vehicle_{vn}_height'] / 2, veh_row[f'vehicle_{vn}_yaw'] - ego_yaw, LABEL_VEHICLE)

    background = np.zeros((NX, NY, NZ), dtype=np.uint8)
    if sample_id in BG_BY_SID:
        for _, b in BG_BY_SID[sample_id].iterrows():
            w = np.array([[b.x, b.y, b.z]], dtype=np.float32)
            e = world_to_ego(w, ego_origin, ego_yaw)[0]
            fill_obb(background, e, b.length / 2, b.width / 2, b.height / 2, b.yaw - ego_yaw, LABEL_VEHICLE)

    return core == LABEL_VEHICLE, background == LABEL_VEHICLE


def mask_centroid_xyz(mask):
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return None
    cx = (idx[:, 0].mean() + 0.5) * VOXEL + X_RANGE[0]
    cy = (idx[:, 1].mean() + 0.5) * VOXEL + Y_RANGE[0]
    cz = (idx[:, 2].mean() + 0.5) * VOXEL + Z_RANGE[0]
    return cx, cy, cz


def run_forward(model, batch, shuffle_cameras=False):
    """batch = scatter(collate([sample], samples_per_gpu=1), [0])[0] -- by
    this point img/img_metas are already plain (unwrapped) tensor/list, same
    convention every other analysis script this session relies on
    (analyze_los_nlos.py, the ablation smoke tests): no DataContainer
    unwrapping needed here, `**batch` already worked directly against
    Co3SOPBase-derived forward_train/forward_test everywhere else.
    """
    img = batch['img']
    if shuffle_cameras:
        perm = torch.randperm(img.shape[1])
        img = img[:, perm]
    img_metas = batch['img_metas']
    with torch.no_grad():
        img_feats = model.extract_feat(img=img, img_metas=img_metas)
        preds = model.pts_bbox_head(img_feats, img_metas)
    return preds, img_metas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='work_dirs/carla_v2v_target_occ_phaseB/epoch_12.pth')
    parser.add_argument('--n-viz', type=int, default=6)
    parser.add_argument('--limit', type=int, default=None, help='cap val samples processed, for a quick smoke test')
    args = parser.parse_args()

    global VEH_DF, BG_BY_SID
    VEH_DF = pd.read_csv(VEHICLE_LOG).set_index('sample_id')
    bg_df = pd.read_csv(BACKGROUND_LOG)
    BG_BY_SID = {sid: g for sid, g in bg_df.groupby('sample_id')}

    cfg = Config.fromfile(CONFIG)
    ds = build_dataset(cfg.data.val)
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    model = model.cuda().eval()

    per_role = {0: [], 1: []}  # role -> list of per-sample dicts
    scene_scores = []  # (tp,p,g) per class, accumulated
    core_recall_num, core_recall_den = 0, 0
    bg_recall_num, bg_recall_den = 0, 0
    shuffle_scores = {0: [], 1: []}
    avgmask_scores = {0: [], 1: []}
    role_train_masks = {0: [], 1: []}

    train_ds = build_dataset(cfg.data.train)
    for i in range(len(train_ds)):
        info = train_ds.data_infos[i]
        link = info[3]
        role = 0 if link == 'TX_CAR1' else 1
        if len(role_train_masks[role]) < 50:  # subsample for speed, enough for a stable average
            item = train_ds.get_data_info(i)
            role_train_masks[role].append(item['gt_target'])
    avg_mask = {r: (np.mean(np.stack(role_train_masks[r]), axis=0) > 0.5) for r in (0, 1)}

    n_samples = min(args.limit, len(ds)) if args.limit else len(ds)
    for i in range(n_samples):
        sample = ds[i]
        # plain numpy arrays already at this stage (not DataContainer-wrapped --
        # only 'img' goes through DC(stack=True) in this pipeline, confirmed
        # earlier this session for gt_occ/gt_beam/gt_target alike).
        gt_target = np.asarray(sample['gt_target'])
        gt_occ = np.asarray(sample['gt_occ'])

        batch = scatter(collate([sample], samples_per_gpu=1), [0])[0]
        preds, img_metas = run_forward(model, batch)
        role = int(img_metas[0]['target_role'])
        sample_id = int(img_metas[0]['occ_path'].split('/')[-2])

        target_pred = torch.argmax(preds['target_preds'][-1], dim=1)[0].cpu().numpy()
        scene_pred = torch.argmax(preds['occ_preds'][-1], dim=1)[0].cpu().numpy()

        tp = int(((target_pred == 1) & (gt_target == 1)).sum())
        p = int((target_pred == 1).sum())
        g = int((gt_target == 1).sum())
        rec = {'tp': tp, 'p': p, 'g': g}

        pred_c = mask_centroid_xyz(target_pred == 1)
        gt_geom = img_metas[0].get('tx_rx_geometry')
        if pred_c is not None and gt_geom is not None:
            dx, dy, dz = pred_c[0] - gt_geom[0], pred_c[1] - gt_geom[1], pred_c[2] - gt_geom[2]
            rec['xyz_err'] = float(np.sqrt(dx**2 + dy**2 + dz**2))
            gt_range, gt_az = gt_geom[3], np.arctan2(gt_geom[1], gt_geom[0])
            pred_range = float(np.sqrt(pred_c[0]**2 + pred_c[1]**2))
            pred_az = float(np.arctan2(pred_c[1], pred_c[0]))
            rec['range_err'] = abs(pred_range - gt_range)
            rec['az_err_deg'] = abs(np.degrees(np.arctan2(np.sin(pred_az - gt_az), np.cos(pred_az - gt_az))))
        per_role[role].append(rec)

        avg = avg_mask[role]
        avgmask_scores[role].append({
            'tp': int((avg & (gt_target == 1)).sum()),
            'p': int(avg.sum()),
            'g': g,
        })

        class_num = 3
        score = np.zeros((class_num, 3))
        for j in range(class_num):
            score[j, 0] += ((gt_occ == j) & (scene_pred == j)).sum()
            score[j, 1] += (gt_occ == j).sum()
            score[j, 2] += (scene_pred == j).sum()
        scene_scores.append(score)

        core_mask, bg_mask = build_reference_masks(sample_id)
        pred_vehicle = (scene_pred == 2)
        core_recall_num += int((pred_vehicle & core_mask).sum())
        core_recall_den += int(core_mask.sum())
        bg_recall_num += int((pred_vehicle & bg_mask).sum())
        bg_recall_den += int(bg_mask.sum())

        # camera-shuffle sanity check (same sample, perturbed input)
        preds_shuf, _ = run_forward(model, batch, shuffle_cameras=True)
        target_pred_shuf = torch.argmax(preds_shuf['target_preds'][-1], dim=1)[0].cpu().numpy()
        tp_s = int(((target_pred_shuf == 1) & (gt_target == 1)).sum())
        p_s = int((target_pred_shuf == 1).sum())
        shuffle_scores[role].append({'tp': tp_s, 'p': p_s, 'g': g})

        if (i + 1) % 50 == 0:
            print(f'  [{i+1}/{n_samples}]', file=sys.stderr)

    print('\n=== Target metrics (FRONT=TX_CAR1 role 0, REAR=TX_CAR2 role 1) ===')
    for role, name in [(0, 'FRONT'), (1, 'REAR')]:
        recs = per_role[role]
        tp = sum(r['tp'] for r in recs); p = sum(r['p'] for r in recs); g = sum(r['g'] for r in recs)
        dice = 2 * tp / (p + g + 1e-6)
        prec = tp / (p + 1e-6)
        rec_ = tp / (g + 1e-6)
        xyz_errs = [r['xyz_err'] for r in recs if 'xyz_err' in r]
        range_errs = [r['range_err'] for r in recs if 'range_err' in r]
        az_errs = [r['az_err_deg'] for r in recs if 'az_err_deg' in r]
        print(f'{name}: n={len(recs)} IoU={iou(tp,p,g):.4f} Dice={dice:.4f} Precision={prec:.4f} Recall={rec_:.4f}')
        print(f'  centroid XYZ err (m): mean={np.mean(xyz_errs):.2f} median={np.median(xyz_errs):.2f}  '
              f'range err (m): mean={np.mean(range_errs):.2f}  azimuth err (deg): mean={np.mean(az_errs):.2f}')

    print('\n=== Scene metrics ===')
    agg = np.stack(scene_scores, axis=0).mean(0)
    names = ['empty', 'road', 'vehicle']
    ious = []
    for j, name in enumerate(names):
        tp, p, g = agg[j]
        iou_j = iou(tp, p, g)
        ious.append(iou_j)
        print(f'  {name}_iou = {iou_j:.4f}')
    print(f'  scene mIoU (excl. empty) = {np.mean(ious[1:]):.4f}')
    print(f'  vehicle recall (core 3 vehicles)  = {core_recall_num/(core_recall_den+1e-6):.4f}  (n_voxels={core_recall_den})')
    print(f'  vehicle recall (background vehicles) = {bg_recall_num/(bg_recall_den+1e-6):.4f}  (n_voxels={bg_recall_den})')

    print('\n=== Anti-shortcut checks ===')
    for role, name in [(0, 'FRONT'), (1, 'REAR')]:
        recs = per_role[role]
        g_total = sum(r['g'] for r in recs)
        print(f'{name} always-zero-mask: IoU=0.0000 Dice=0.0000 (trivially, since predicted positive=0, '
              f'gt positive total={g_total})')

        am = avgmask_scores[role]
        tp_a = sum(r['tp'] for r in am); p_a = sum(r['p'] for r in am); g_a = sum(r['g'] for r in am)
        dice_a = 2 * tp_a / (p_a + g_a + 1e-6)
        print(f'{name} always-average-training-mask (constant mask, no per-sample input used): '
              f'IoU={iou(tp_a,p_a,g_a):.4f} Dice={dice_a:.4f}')

        s = shuffle_scores[role]
        tp = sum(r['tp'] for r in s); p = sum(r['p'] for r in s); g = sum(r['g'] for r in s)
        real_tp = sum(r['tp'] for r in per_role[role]); real_p = sum(r['p'] for r in per_role[role])
        print(f'{name} camera-shuffle: IoU {iou(tp,p,g):.4f} vs real {iou(real_tp,real_p,g):.4f} '
              f'({"DEGRADED (good, model uses vision)" if iou(tp,p,g) < iou(real_tp,real_p,g) - 0.02 else "NO CLEAR DEGRADATION"})')


if __name__ == '__main__':
    main()
