#!/usr/bin/env python3
"""Post-hoc diagnosis requested after the Stage 2B report: split FRONT
(TX_CAR1) val results by Sionna LOS/NLOS, and do an actual failure
inspection (not just "NLOS=occluded" by assumption) on the worst
centroid-error samples -- classify each into target_missing /
wrong_vehicle_selected / fixed_prior_location / scene_occupancy_failure /
boundary_range_issue / threshold_centroid_artifact / other, using both
quantifiable signals (computed here) and a rendered BEV visualization
(saved to disk for manual confirmation).

Usage: python tools/diagnose_front_failures.py [--checkpoint ...] [--top-k 10]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/home/admin0/carla_V2V/src')
import projects.mmdet3d_plugin_carla_v2v as plugin  # noqa: F401
from mmdet.datasets import build_dataset
from mmdet3d.models import build_model
from generate_occupancy_gt import NX, NY, NZ, LABEL_VEHICLE, VOXEL, X_RANGE, Y_RANGE, Z_RANGE, fill_obb, world_to_ego  # noqa: E402

CONFIG = 'projects/configs/co3sop_base/co3sop_base_carla_v2v_target_occ.py'
VEHICLE_LOG = '/home/admin0/data/run_20260904_142344_bigcar80/rear_vehicle_log.csv'
BACKGROUND_LOG = '/home/admin0/data/run_20260904_142344_bigcar80/background_vehicles_log.csv'
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'work_dirs', 'front_failure_viz')
BOUNDARY_MARGIN = 3.0  # metres from the +-40m edge to call "boundary/range issue"


def iou(tp, p, g, eps=1e-6):
    return float(tp / (p + g - tp + eps))


def centroid_xyz(mask):
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return None
    cx = (idx[:, 0].mean() + 0.5) * VOXEL + X_RANGE[0]
    cy = (idx[:, 1].mean() + 0.5) * VOXEL + Y_RANGE[0]
    cz = (idx[:, 2].mean() + 0.5) * VOXEL + Z_RANGE[0]
    return np.array([cx, cy, cz])


def grid_xy_to_idx(x, y):
    return int((x - X_RANGE[0]) / VOXEL), int((y - Y_RANGE[0]) / VOXEL)


def classify_failure(pred_mask, gt_mask, scene_pred, avg_mask_centroid, rear_pos_ego, bg_positions_ego, gt_xyz):
    p = int(pred_mask.sum())
    # p<5 ("target_missing") is a *symptom*, not a root cause -- boundary
    # clipping and scene-branch failure both also produce an empty
    # prediction, so check for those causes even when p<5 (visual inspection
    # on this session's val set found exactly this: some "target_missing"
    # cases were really boundary_range_issue, others were
    # scene_occupancy_failure during sharp turns -- see the experiment log).
    if abs(gt_xyz[0]) > (X_RANGE[1] - BOUNDARY_MARGIN) or abs(gt_xyz[1]) > (Y_RANGE[1] - BOUNDARY_MARGIN):
        return 'boundary_range_issue', {'gt_x': float(gt_xyz[0]), 'gt_y': float(gt_xyz[1])}

    gx0, gy0 = grid_xy_to_idx(gt_xyz[0], gt_xyz[1])
    r0 = 3  # voxels (~1.2m) around the GT location
    x00, x10 = max(0, gx0 - r0), min(NX, gx0 + r0 + 1)
    y00, y10 = max(0, gy0 - r0), min(NY, gy0 + r0 + 1)
    scene_has_vehicle_near_gt0 = (scene_pred[x00:x10, y00:y10, :] == 2).any()

    if p < 5:
        if not scene_has_vehicle_near_gt0:
            return 'scene_occupancy_failure', {}
        return 'target_missing', {}

    comps, n_comp = ndimage.label(pred_mask)
    sizes = ndimage.sum(pred_mask, comps, range(1, n_comp + 1)) if n_comp > 0 else np.array([])
    largest_frac = sizes.max() / p if len(sizes) else 0.0
    pred_c = centroid_xyz(pred_mask)

    if n_comp > 1 and largest_frac < 0.7:
        return 'threshold_centroid_artifact', {'n_components': int(n_comp), 'largest_frac': float(largest_frac)}

    # near a known OTHER vehicle (rear car or a background vehicle) instead of the true FRONT target?
    if rear_pos_ego is not None and np.linalg.norm(pred_c[:2] - rear_pos_ego[:2]) < 3.0:
        return 'wrong_vehicle_selected', {'matched': 'REAR_TX(vehicle_3)', 'dist_to_gt_m': float(np.linalg.norm(pred_c - gt_xyz))}
    for bg_pos in bg_positions_ego:
        if np.linalg.norm(pred_c[:2] - bg_pos[:2]) < 3.0:
            return 'wrong_vehicle_selected', {'matched': 'background_vehicle', 'dist_to_gt_m': float(np.linalg.norm(pred_c - gt_xyz))}

    if avg_mask_centroid is not None and np.linalg.norm(pred_c[:2] - avg_mask_centroid[:2]) < 2.0 \
            and np.linalg.norm(gt_xyz[:2] - avg_mask_centroid[:2]) > 4.0:
        return 'fixed_prior_location', {'dist_pred_to_prior_m': float(np.linalg.norm(pred_c - avg_mask_centroid))}

    gx, gy = grid_xy_to_idx(gt_xyz[0], gt_xyz[1])
    r = 3  # voxels (~1.2m) around the GT location
    x0, x1 = max(0, gx - r), min(NX, gx + r + 1)
    y0, y1 = max(0, gy - r), min(NY, gy + r + 1)
    scene_has_vehicle_near_gt = (scene_pred[x0:x1, y0:y1, :] == 2).any()
    if not scene_has_vehicle_near_gt:
        return 'scene_occupancy_failure', {}

    if abs(gt_xyz[0]) > (X_RANGE[1] - BOUNDARY_MARGIN) or abs(gt_xyz[1]) > (Y_RANGE[1] - BOUNDARY_MARGIN):
        return 'boundary_range_issue', {'gt_x': float(gt_xyz[0]), 'gt_y': float(gt_xyz[1])}

    return 'other', {}


def render_viz(sample_id, scene_gt, scene_pred, target_gt, target_pred, ego_pose_row, bg_df_sid, tag, xyz_err, is_los):
    ox, oy = ego_pose_row.vehicle_2_x, ego_pose_row.vehicle_2_y
    yaw = ego_pose_row.vehicle_2_yaw

    def w2e(wx, wy):
        import math
        a = math.radians(yaw)
        dx, dy = wx - ox, wy - oy
        return math.cos(a) * dx + math.sin(a) * dy, -math.sin(a) * dx + math.cos(a) * dy

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, scene, title in [(axes[0], scene_gt, 'Scene GT'), (axes[1], scene_pred, 'Scene Pred')]:
        ax.imshow(scene.max(axis=2).T, origin='lower', extent=[-40, 40, -40, 40], cmap='Greys', vmin=0, vmax=2, alpha=0.6)
        ax.set_xlim(-40, 40); ax.set_ylim(-40, 40)
        ax.set_title(f'{title} (sample {sample_id}, is_los={is_los})')

    for ax, tmask, label, color in [(axes[0], target_gt, 'target GT', 'lime'), (axes[1], target_pred, 'target pred', 'red')]:
        ys, xs = np.where(tmask.max(axis=2).T)
        if len(xs):
            ax.scatter(xs * VOXEL - 40, ys * VOXEL - 40, s=3, color=color, label=label)

    for ax in axes:
        ax.scatter([0], [0], marker='*', s=150, color='black', label='ego(RX)')
        ex1, ey1 = w2e(ego_pose_row.vehicle_1_x, ego_pose_row.vehicle_1_y)
        ex3, ey3 = w2e(ego_pose_row.vehicle_3_x, ego_pose_row.vehicle_3_y)
        ax.scatter([ex1], [ey1], marker='x', s=100, color='red', label='vehicle_1 (FRONT TX, true target)')
        ax.scatter([ex3], [ey3], marker='x', s=100, color='blue', label='vehicle_3 (REAR)')
        if bg_df_sid is not None:
            for _, b in bg_df_sid.iterrows():
                bx, by = w2e(b.x, b.y)
                if -40 <= bx <= 40 and -40 <= by <= 40:
                    ax.scatter([bx], [by], marker='s', s=12, color='orange')
        ax.legend(loc='upper right', fontsize=7)

    fig.suptitle(f'FRONT failure: sample {sample_id}, xyz_err={xyz_err:.1f}m, is_los={is_los}')
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f'{tag}_sample_{sample_id:06d}_err{xyz_err:.0f}m.png')
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='work_dirs/carla_v2v_target_occ_phaseB/epoch_12.pth')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--limit', type=int, default=None, help='cap FRONT samples processed, for a quick smoke test')
    args = parser.parse_args()

    veh_df = pd.read_csv(VEHICLE_LOG).set_index('sample_id')
    bg_df = pd.read_csv(BACKGROUND_LOG)
    bg_by_sid = {sid: g for sid, g in bg_df.groupby('sample_id')}

    cfg = Config.fromfile(CONFIG)
    ds = build_dataset(cfg.data.val)
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    model = model.cuda().eval()

    # avg training-mask centroid (FRONT role) for the "fixed prior" check -- same
    # 50-sample subsample the Stage 2B report used.
    train_ds = build_dataset(cfg.data.train)
    front_masks = []
    for i in range(len(train_ds)):
        if train_ds.data_infos[i][3] == 'TX_CAR1' and len(front_masks) < 50:
            front_masks.append(train_ds.get_data_info(i)['gt_target'])
    avg_mask = np.mean(np.stack(front_masks), axis=0) > 0.5
    avg_mask_centroid = centroid_xyz(avg_mask)

    records = []
    for i in range(len(ds)):
        link = ds.data_infos[i][3]
        if link != 'TX_CAR1':
            continue
        if args.limit and len(records) >= args.limit:
            break
        sample = ds[i]
        gt_target = np.asarray(sample['gt_target']) == 1
        gt_occ = np.asarray(sample['gt_occ'])

        batch = scatter(collate([sample], samples_per_gpu=1), [0])[0]
        with torch.no_grad():
            img_feats = model.extract_feat(img=batch['img'], img_metas=batch['img_metas'])
            preds = model.pts_bbox_head(img_feats, batch['img_metas'])
        target_pred = (torch.argmax(preds['target_preds'][-1], dim=1)[0].cpu().numpy() == 1)
        scene_pred = torch.argmax(preds['occ_preds'][-1], dim=1)[0].cpu().numpy()

        im = batch['img_metas'][0]
        is_los = bool(im['beam_eval_meta']['is_los'])
        sample_id = int(im['occ_path'].split('/')[-2])
        gt_geom = im['tx_rx_geometry']  # [dx,dy,dz,dist,rel_yaw] -- eval only
        gt_xyz = np.array([gt_geom[0], gt_geom[1], gt_geom[2]])

        tp = int((target_pred & gt_target).sum()); p = int(target_pred.sum()); g = int(gt_target.sum())
        pred_c = centroid_xyz(target_pred)
        xyz_err = float(np.linalg.norm(pred_c - gt_xyz)) if pred_c is not None else float('inf')
        az_gt = np.arctan2(gt_geom[1], gt_geom[0])
        az_pred = np.arctan2(pred_c[1], pred_c[0]) if pred_c is not None else np.nan
        az_err = abs(np.degrees(np.arctan2(np.sin(az_pred - az_gt), np.cos(az_pred - az_gt)))) if pred_c is not None else float('inf')

        records.append(dict(sample_id=sample_id, is_los=is_los, tp=tp, p=p, g=g,
                             iou=iou(tp, p, g), dice=2 * tp / (p + g + 1e-6),
                             xyz_err=xyz_err, az_err=az_err,
                             target_pred=target_pred, gt_target=gt_target,
                             scene_pred=scene_pred, gt_occ=gt_occ, gt_xyz=gt_xyz))
        if (len(records)) % 25 == 0:
            print(f'  [{len(records)}/157]', file=sys.stderr)

    df = pd.DataFrame([{k: v for k, v in r.items() if k not in
                         ('target_pred', 'gt_target', 'scene_pred', 'gt_occ', 'gt_xyz')} for r in records])

    print('\n=== FRONT LOS vs NLOS split ===')
    for los_val, name in [(True, 'LOS'), (False, 'NLOS')]:
        sub = df[df.is_los == los_val]
        tp, p, g = sub.tp.sum(), sub.p.sum(), sub.g.sum()
        print(f'{name}: n={len(sub)}  IoU={iou(tp,p,g):.4f}  Dice={2*tp/(p+g+1e-6):.4f}')
        print(f'  centroid err: mean={sub.xyz_err.mean():.2f}m median={sub.xyz_err.median():.2f}m  '
              f'azimuth err mean={sub.az_err.mean():.2f}deg')
        print(f'  fraction err>5m: {(sub.xyz_err>5).mean():.3f}   fraction err>10m: {(sub.xyz_err>10).mean():.3f}')

    # Failure inspection on the worst-K by xyz_err (across all FRONT, LOS+NLOS)
    worst = sorted(records, key=lambda r: -r['xyz_err'])[:args.top_k]
    print(f'\n=== Failure inspection: top {len(worst)} FRONT samples by centroid error ===')
    categories = []
    for r in worst:
        ego_row = veh_df.loc[r['sample_id']]
        ex1, ey1, ez1 = world_to_ego(np.array([[ego_row.vehicle_1_x, ego_row.vehicle_1_y, ego_row.vehicle_1_z]], dtype=np.float32),
                                  np.array([ego_row.vehicle_2_x, ego_row.vehicle_2_y, ego_row.vehicle_2_z], dtype=np.float32),
                                  float(ego_row.vehicle_2_yaw))[0]
        ex3, ey3, ez3 = world_to_ego(np.array([[ego_row.vehicle_3_x, ego_row.vehicle_3_y, ego_row.vehicle_3_z]], dtype=np.float32),
                                       np.array([ego_row.vehicle_2_x, ego_row.vehicle_2_y, ego_row.vehicle_2_z], dtype=np.float32),
                                       float(ego_row.vehicle_2_yaw))[0]
        rear_pos_ego = np.array([ex3, ey3, ez3])
        bg_positions_ego = []
        if r['sample_id'] in bg_by_sid:
            for _, b in bg_by_sid[r['sample_id']].iterrows():
                bp = world_to_ego(np.array([[b.x, b.y, b.z]], dtype=np.float32),
                                    np.array([ego_row.vehicle_2_x, ego_row.vehicle_2_y, ego_row.vehicle_2_z], dtype=np.float32),
                                    float(ego_row.vehicle_2_yaw))[0]
                bg_positions_ego.append(bp)

        cat, extra = classify_failure(r['target_pred'], r['gt_target'], r['scene_pred'],
                                        avg_mask_centroid, rear_pos_ego, bg_positions_ego, r['gt_xyz'])
        viz_path = render_viz(r['sample_id'], r['gt_occ'], r['scene_pred'], r['gt_target'], r['target_pred'],
                                ego_row, bg_by_sid.get(r['sample_id']), 'front_worst', r['xyz_err'], r['is_los'])
        categories.append(cat)
        print(f"sample={r['sample_id']:>4} is_los={r['is_los']!s:>5} xyz_err={r['xyz_err']:6.1f}m "
              f"az_err={r['az_err']:6.1f}deg iou={r['iou']:.3f}  category={cat}  {extra}")
        print(f'  viz: {viz_path}')

    print('\n=== Failure category summary (top-K worst) ===')
    for cat in sorted(set(categories)):
        print(f'  {cat}: {categories.count(cat)}/{len(categories)}')


if __name__ == '__main__':
    main()
