#!/usr/bin/env python3
"""Stage 2C post-training analysis: C0 (single-frame forecasting) vs C1
(K=3-frame temporal forecasting), replaying both trained checkpoints on
their (identical, by construction) val sets.

Per the plan (effervescent-shimmying-turing.md §7):
  - full metric table (Scene mIoU/road/vehicle IoU, background-vehicle
    recall recomputed from source CSVs at t+1 state, Target IoU/Dice/
    precision/recall, centroid error incl. fraction >5m/>10m) for C0 vs C1
  - subset breakdown (Overall/FRONT/REAR/future_LOS/future_NLOS (primary)/
    input_LOS/input_NLOS (secondary)/boundary/high-relative-yaw/
    dense-traffic) for both C0 and C1
  - C1-only temporal sanity checks (no-history/zero-history/
    shuffled-history/reversed-order) as eval-time input perturbations,
    reusing the SAME trained C1 checkpoint (no new training runs)

All subset-defining quantities (relative_yaw at t, background count at
t+1, boundary from target_center_raw, LOS at t vs t+1) are recomputed here
independently from source CSVs/npz files -- not read back off whatever a
given dataset's img_metas happens to carry (C0's img_metas has
tx_rx_geometry/beam_eval_meta from CarlaV2VBeamCo3SOP.get_data_info; C1's
frame dicts are built via _build_data_info directly and never go through
that method, so they don't have those fields at all) -- so both models are
scored against exactly the same subset definitions.

Usage: python tools/analyze_temporal_c0_c1.py [--limit N]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/home/admin0/carla_V2V/src')
import projects.mmdet3d_plugin  # noqa: F401
import projects.mmdet3d_plugin_carla_v2v  # noqa: F401
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from generate_occupancy_gt import (  # noqa: E402
    NX, NY, NZ, LABEL_VEHICLE, VOXEL, X_RANGE, Y_RANGE, Z_RANGE, fill_obb, world_to_ego,
)

C0_CONFIG = 'projects/configs/co3sop_base/co3sop_base_carla_v2v_temporal_c0.py'
C1_CONFIG = 'projects/configs/co3sop_base/co3sop_base_carla_v2v_temporal_c1.py'
C0_CKPT = 'work_dirs/carla_v2v_temporal_c0/epoch_12.pth'
C1_CKPT = 'work_dirs/carla_v2v_temporal_c1/epoch_12.pth'
VEHICLE_LOG = '/home/admin0/data/run_20260904_142344_bigcar80/rear_vehicle_log.csv'
BACKGROUND_LOG = '/home/admin0/data/run_20260904_142344_bigcar80/background_vehicles_log.csv'
BEAM_CSV = '/home/admin0/carla_V2V/channel_analysis/channel_summary_from_carla_20260904_145458.csv'
FUTURE_TARGET_ROOT = '/home/admin0/carla_V2V/occupancy_prediction_beam/target_occupancy_gt_future_t1'
LINK_VEHICLE = {'TX_CAR1': 1, 'TX_CAR2': 3}
RX_VEHICLE = 2
VEHICLES = (1, 2, 3)

# Data-driven thresholds, exactly as agreed in the approved plan (§1):
YAW_THRESH_DEG = 30.0       # high-relative-yaw, computed at frame t (not t+1)
DENSE_TRAFFIC_THRESH = 3    # background vehicles within 40m of RX_t, using t+1 background state
DENSE_TRAFFIC_RADIUS = 40.0
BOUNDARY_THRESH = 37.0      # max(|x|,|y|) of the continuous (pre-voxelization) target_center_raw


def iou(tp, p, g, eps=1e-6):
    return float(tp / (p + g - tp + eps))


def wrap_deg(d):
    return float(np.degrees(np.arctan2(np.sin(np.radians(d)), np.cos(np.radians(d)))))


def mask_centroid_xy(mask):
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return None
    cx = (idx[:, 0].mean() + 0.5) * VOXEL + X_RANGE[0]
    cy = (idx[:, 1].mean() + 0.5) * VOXEL + Y_RANGE[0]
    return cx, cy


def build_future_reference_masks(t, veh_df, bg_by_sid):
    """Core-3-vehicle vs background-vehicle reference masks for the t+1
    state, expressed in RX's t-frame -- mirrors generate_temporal_occupancy_gt.py's
    own scene-GT construction exactly (ego_origin/ego_yaw = RX pose AT t,
    objects voxelized AT t+1), just split into core-vs-background instead of
    merging into one class (generate_temporal_occupancy_gt.py's saved GT
    only has one Vehicle class, same limitation Stage 2A's occupancy_gt had)."""
    t1 = t + 1
    if t not in veh_df.index or t1 not in veh_df.index:
        return None, None
    ego_row = veh_df.loc[t]
    ego_origin = np.array([ego_row.vehicle_2_x, ego_row.vehicle_2_y, ego_row.vehicle_2_z], dtype=np.float32)
    ego_yaw = float(ego_row.vehicle_2_yaw)

    row1 = veh_df.loc[t1]
    core = np.zeros((NX, NY, NZ), dtype=np.uint8)
    for vn in VEHICLES:
        w = np.array([[row1[f'vehicle_{vn}_x'], row1[f'vehicle_{vn}_y'], row1[f'vehicle_{vn}_z']]], dtype=np.float32)
        e = world_to_ego(w, ego_origin, ego_yaw)[0]
        fill_obb(core, e, row1[f'vehicle_{vn}_length'] / 2, row1[f'vehicle_{vn}_width'] / 2,
                  row1[f'vehicle_{vn}_height'] / 2, row1[f'vehicle_{vn}_yaw'] - ego_yaw, LABEL_VEHICLE)

    background = np.zeros((NX, NY, NZ), dtype=np.uint8)
    if t1 in bg_by_sid:
        for _, b in bg_by_sid[t1].iterrows():
            w = np.array([[b.x, b.y, b.z]], dtype=np.float32)
            e = world_to_ego(w, ego_origin, ego_yaw)[0]
            fill_obb(background, e, b.length / 2, b.width / 2, b.height / 2, b.yaw - ego_yaw, LABEL_VEHICLE)

    return core == LABEL_VEHICLE, background == LABEL_VEHICLE


def compute_subsets(sample_ids_links, veh_df, bg_by_sid, beam_df):
    """Returns {(t, link): {subset_name: bool, ...}} recomputed independently
    from source CSVs/npz -- same for both C0 and C1's samples."""
    beam_by_key = {(int(r.sample_id), r.tx_name): r for r in beam_df.itertuples(index=False)}
    out = {}
    for t, link in sample_ids_links:
        tx_vn = LINK_VEHICLE[link]
        rec = {}

        # LOS: primary = future (t+1), secondary = input (t)
        row_future = beam_by_key.get((t + 1, link))
        row_input = beam_by_key.get((t, link))
        rec['future_is_los'] = bool(row_future.is_los) if row_future is not None else None
        rec['input_is_los'] = bool(row_input.is_los) if row_input is not None else None

        # high-relative-yaw: at frame t, tx vs rx pose
        if t in veh_df.index:
            row_t = veh_df.loc[t]
            rel_yaw = wrap_deg(float(row_t[f'vehicle_{tx_vn}_yaw']) - float(row_t.vehicle_2_yaw))
            rec['high_yaw'] = abs(rel_yaw) > YAW_THRESH_DEG
            rec['relative_yaw_t'] = rel_yaw
        else:
            rec['high_yaw'] = None

        # dense-traffic: background vehicles at t+1 within 40m of RX's t position
        t1 = t + 1
        if t in veh_df.index and t1 in bg_by_sid:
            rx_t = np.array([veh_df.loc[t].vehicle_2_x, veh_df.loc[t].vehicle_2_y], dtype=np.float64)
            bg = bg_by_sid[t1]
            d = np.sqrt((bg.x.values - rx_t[0]) ** 2 + (bg.y.values - rx_t[1]) ** 2)
            rec['dense_traffic'] = int((d < DENSE_TRAFFIC_RADIUS).sum()) >= DENSE_TRAFFIC_THRESH
        else:
            rec['dense_traffic'] = None

        # boundary: from the continuous target_center_raw (pre-voxelization,
        # pre-clipping), NOT a possibly-clipped mask centroid
        target_path = os.path.join(FUTURE_TARGET_ROOT, f'{t:06d}', f'{link}.npz')
        if os.path.exists(target_path):
            center = np.load(target_path)['target_center_raw']
            rec['boundary'] = bool(max(abs(center[0]), abs(center[1])) > BOUNDARY_THRESH)
            rec['target_center_raw'] = center
        else:
            rec['boundary'] = None
            rec['target_center_raw'] = None

        out[(t, link)] = rec
    return out


def run_c0_forward(model, batch):
    img = batch['img']
    img_metas = batch['img_metas']
    with torch.no_grad():
        img_feats = model.extract_feat(img=img, img_metas=img_metas)
        preds = model.pts_bbox_head(img_feats, img_metas)
    return preds


def run_c1_forward(model, img, img_metas, perturb=None):
    """img: [1,K,N,C,H,W] cuda tensor. img_metas: list len 1, each list len K.
    perturb: one of None/'no_history'/'zero_history'/'shuffled_history'/
    'reversed_order', or a dict {'shuffle_with': (other_img, other_img_metas)}
    for shuffled_history. Manually re-runs the head's own forward logic
    (calling its already-trained submodules directly, no model code changes)
    so these eval-time-only perturbations don't require touching
    TemporalTargetAwareOccHead itself."""
    head = model.pts_bbox_head
    K = img.shape[1]
    frame_metas = [[m[k] for m in img_metas] for k in range(K)]

    if perturb == 'reversed_order':
        img = img.flip(dims=[1])  # image content reversed, img_metas/warp matrices left as-is (mismatch is the point)

    with torch.no_grad():
        aligned_feats = []
        for k in range(K):
            img_feats_k = model.extract_feat(img=img[:, k], img_metas=frame_metas[k])
            if perturb == 'zero_history' and k < K - 1:
                # zero the RAW IMAGE for history frames, but still run the
                # full backbone/encoder on it (tests robustness to garbage-
                # but-present history, not the same as dropping it).
                img_feats_k = model.extract_feat(img=torch.zeros_like(img[:, k]), img_metas=frame_metas[k])
            fused_k, _, _ = head._encode_and_fuse(img_feats_k, frame_metas[k])
            if perturb == 'no_history' and k < K - 1:
                # zero the FUSED FEATURE (post-encode, pre-warp) for history
                # frames -- cleanest "history contributes nothing" ablation.
                fused_k = torch.zeros_like(fused_k)
            aligned_feats.append(head._warp_to_t(fused_k, frame_metas[k]))

        fused_cat = torch.cat(aligned_feats, dim=1)
        fused_temporal = head.temporal_fusion(fused_cat)
        outputs = head._run_deblocks(fused_temporal)
        occ_preds = [head.occ[i](outputs[i]) for i in range(len(outputs))]

        role = fused_temporal.new_tensor([m['target_role'] for m in frame_metas[-1]], dtype=torch.long)
        role_embed = head.role_embedding(role)
        target_preds = []
        for i in range(len(outputs)):
            scale, shift = head.film_proj[i](role_embed).chunk(2, dim=1)
            scale = scale[:, :, None, None, None]
            shift = shift[:, :, None, None, None]
            modulated = outputs[i] * (1 + scale) + shift
            target_preds.append(head.target_occ[i](modulated))

    return {'occ_preds': occ_preds, 'target_preds': target_preds}


SUBSET_DEFS = [
    ('Overall', lambda r: True),
    ('FRONT', lambda r: r['link'] == 'TX_CAR1'),
    ('REAR', lambda r: r['link'] == 'TX_CAR2'),
    ('future_LOS', lambda r: r['future_is_los'] is True),
    ('future_NLOS', lambda r: r['future_is_los'] is False),
    ('input_LOS', lambda r: r['input_is_los'] is True),
    ('input_NLOS', lambda r: r['input_is_los'] is False),
    ('boundary', lambda r: r['boundary'] is True),
    ('high_yaw', lambda r: r['high_yaw'] is True),
    ('dense_traffic', lambda r: r['dense_traffic'] is True),
]


def summarize(records):
    tp = sum(r['tp'] for r in records); p = sum(r['p'] for r in records); g = sum(r['g'] for r in records)
    dice = 2 * tp / (p + g + 1e-6)
    prec = tp / (p + 1e-6)
    rec = tp / (g + 1e-6)
    errs = [r['centroid_err'] for r in records]
    frac5 = float(np.mean([e > 5.0 for e in errs])) if errs else float('nan')
    frac10 = float(np.mean([e > 10.0 for e in errs])) if errs else float('nan')
    finite_errs = [e for e in errs if np.isfinite(e)]
    return dict(n=len(records), iou=iou(tp, p, g), dice=dice, precision=prec, recall=rec,
                centroid_median=np.median(finite_errs) if finite_errs else float('nan'),
                frac_gt_5m=frac5, frac_gt_10m=frac10)


def print_subset_table(name, records):
    print(f'\n--- {name}: subset breakdown ---')
    header = f'{"subset":<15}{"n":>5}  {"IoU":>7}{"Dice":>7}{"Prec":>7}{"Rec":>7}{"medErr":>8}{"frac>5m":>9}{"frac>10m":>10}'
    print(header)
    for subset_name, pred in SUBSET_DEFS:
        sub = [r for r in records if pred(r)]
        if not sub:
            print(f'{subset_name:<15}{0:>5}  (no samples)')
            continue
        s = summarize(sub)
        print(f'{subset_name:<15}{s["n"]:>5}  {s["iou"]:>7.4f}{s["dice"]:>7.4f}{s["precision"]:>7.4f}'
              f'{s["recall"]:>7.4f}{s["centroid_median"]:>8.2f}{s["frac_gt_5m"]:>9.3f}{s["frac_gt_10m"]:>10.3f}')


def replay(model_name, cfg_path, ckpt_path, forward_fn):
    cfg = Config.fromfile(cfg_path)
    ds = build_dataset(cfg.data.val)
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, ckpt_path, map_location='cpu', strict=False)
    model = model.cuda().eval()
    return cfg, ds, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    veh_df = pd.read_csv(VEHICLE_LOG).set_index('sample_id')
    bg_df = pd.read_csv(BACKGROUND_LOG)
    bg_by_sid = {sid: g for sid, g in bg_df.groupby('sample_id')}
    beam_df = pd.read_csv(BEAM_CSV)

    print('Loading C0...', file=sys.stderr)
    c0_cfg, c0_ds, c0_model = replay('C0', C0_CONFIG, C0_CKPT, run_c0_forward)
    print('Loading C1...', file=sys.stderr)
    c1_cfg, c1_ds, c1_model = replay('C1', C1_CONFIG, C1_CKPT, None)

    c0_infos = [(int(info[2]), info[3]) for info in c0_ds.data_infos]
    c1_infos = [(int(info[2]), info[3]) for info in c1_ds.data_infos]
    assert sorted(c0_infos) == sorted(c1_infos), \
        f'C0/C1 val sample sets differ! C0 n={len(c0_infos)} C1 n={len(c1_infos)} -- comparison would not be fair'
    print(f'[fairness check] C0 and C1 val sets match exactly: {len(c0_infos)} samples', file=sys.stderr)

    subsets = compute_subsets(c0_infos, veh_df, bg_by_sid, beam_df)

    n0 = min(args.limit, len(c0_ds)) if args.limit else len(c0_ds)
    c0_records = []
    core_recall = {'num': 0, 'den': 0}
    bg_recall = {'num': 0, 'den': 0}
    c0_scene_scores = []
    for i in range(n0):
        sample = c0_ds[i]
        gt_target = np.asarray(sample['gt_target'])
        gt_occ = np.asarray(sample['gt_occ'])
        info = c0_ds.data_infos[i]
        t, link = int(info[2]), info[3]

        batch = scatter(collate([sample], samples_per_gpu=1), [0])[0]
        preds = run_c0_forward(c0_model, batch)
        target_pred = torch.argmax(preds['target_preds'][-1], dim=1)[0].cpu().numpy()
        scene_pred = torch.argmax(preds['occ_preds'][-1], dim=1)[0].cpu().numpy()

        tp = int(((target_pred == 1) & (gt_target == 1)).sum())
        p = int((target_pred == 1).sum())
        g = int((gt_target == 1).sum())

        pred_c = mask_centroid_xy(target_pred == 1)
        gt_center = subsets[(t, link)]['target_center_raw']
        if pred_c is not None and gt_center is not None:
            err = float(np.hypot(pred_c[0] - gt_center[0], pred_c[1] - gt_center[1]))
        else:
            err = float('inf')

        rec = dict(subsets[(t, link)])
        rec.update(link=link, t=t, tp=tp, p=p, g=g, centroid_err=err)
        c0_records.append(rec)

        class_num = 3
        score = np.zeros((class_num, 3))
        for j in range(class_num):
            score[j, 0] += ((gt_occ == j) & (scene_pred == j)).sum()
            score[j, 1] += (gt_occ == j).sum()
            score[j, 2] += (scene_pred == j).sum()
        c0_scene_scores.append(score)

        core_mask, background_mask = build_future_reference_masks(t, veh_df, bg_by_sid)
        if core_mask is not None:
            pred_vehicle = scene_pred == 2
            core_recall['num'] += int((pred_vehicle & core_mask).sum()); core_recall['den'] += int(core_mask.sum())
            bg_recall['num'] += int((pred_vehicle & background_mask).sum()); bg_recall['den'] += int(background_mask.sum())

        if (i + 1) % 50 == 0:
            print(f'  C0 [{i+1}/{n0}]', file=sys.stderr)

    del c0_model
    torch.cuda.empty_cache()

    n1 = min(args.limit, len(c1_ds)) if args.limit else len(c1_ds)
    c1_records = []
    c1_perturb_records = {k: [] for k in ('no_history', 'zero_history', 'shuffled_history', 'reversed_order')}
    c1_scene_scores = []
    core_recall_c1 = {'num': 0, 'den': 0}
    bg_recall_c1 = {'num': 0, 'den': 0}

    shuffle_pool = []  # (img, img_metas) cache for shuffled-history, filled as we go
    for i in range(n1):
        sample = c1_ds[i]
        gt_target = np.asarray(sample['gt_target_future'])
        gt_occ = np.asarray(sample['gt_occ_future'])
        info = c1_ds.data_infos[i]
        t, link = int(info[2]), info[3]

        batch = scatter(collate([sample], samples_per_gpu=1), [0])[0]
        img = batch['img'].cuda()
        img_metas = batch['img_metas']

        preds = run_c1_forward(c1_model, img, img_metas, perturb=None)
        target_pred = torch.argmax(preds['target_preds'][-1], dim=1)[0].cpu().numpy()
        scene_pred = torch.argmax(preds['occ_preds'][-1], dim=1)[0].cpu().numpy()

        tp = int(((target_pred == 1) & (gt_target == 1)).sum())
        p = int((target_pred == 1).sum())
        g = int((gt_target == 1).sum())
        pred_c = mask_centroid_xy(target_pred == 1)
        gt_center = subsets[(t, link)]['target_center_raw']
        err = float(np.hypot(pred_c[0] - gt_center[0], pred_c[1] - gt_center[1])) if (pred_c is not None and gt_center is not None) else float('inf')

        rec = dict(subsets[(t, link)])
        rec.update(link=link, t=t, tp=tp, p=p, g=g, centroid_err=err)
        c1_records.append(rec)

        class_num = 3
        score = np.zeros((class_num, 3))
        for j in range(class_num):
            score[j, 0] += ((gt_occ == j) & (scene_pred == j)).sum()
            score[j, 1] += (gt_occ == j).sum()
            score[j, 2] += (scene_pred == j).sum()
        c1_scene_scores.append(score)

        core_mask, background_mask = build_future_reference_masks(t, veh_df, bg_by_sid)
        if core_mask is not None:
            pred_vehicle = scene_pred == 2
            core_recall_c1['num'] += int((pred_vehicle & core_mask).sum()); core_recall_c1['den'] += int(core_mask.sum())
            bg_recall_c1['num'] += int((pred_vehicle & background_mask).sum()); bg_recall_c1['den'] += int(background_mask.sum())

        # ---- temporal sanity checks (same sample, perturbed input) ----
        for perturb in ('no_history', 'zero_history', 'reversed_order'):
            p_preds = run_c1_forward(c1_model, img, img_metas, perturb=perturb)
            p_target_pred = torch.argmax(p_preds['target_preds'][-1], dim=1)[0].cpu().numpy()
            p_tp = int(((p_target_pred == 1) & (gt_target == 1)).sum())
            p_p = int((p_target_pred == 1).sum())
            c1_perturb_records[perturb].append({'tp': p_tp, 'p': p_p, 'g': g})

        if len(shuffle_pool) >= 1:
            other_img, other_img_metas = shuffle_pool[np.random.randint(len(shuffle_pool))]
            swapped_img = img.clone()
            swapped_img[:, :2] = other_img[:, :2]  # swap in another sample's history frames (k=0,1), keep current frame k=2
            s_preds = run_c1_forward(c1_model, swapped_img, img_metas, perturb=None)
            s_target_pred = torch.argmax(s_preds['target_preds'][-1], dim=1)[0].cpu().numpy()
            s_tp = int(((s_target_pred == 1) & (gt_target == 1)).sum())
            s_p = int((s_target_pred == 1).sum())
            c1_perturb_records['shuffled_history'].append({'tp': s_tp, 'p': s_p, 'g': g})
        shuffle_pool.append((img.clone(), img_metas))
        if len(shuffle_pool) > 20:
            shuffle_pool.pop(0)

        if (i + 1) % 50 == 0:
            print(f'  C1 [{i+1}/{n1}]', file=sys.stderr)

    print('\n' + '=' * 78)
    print('STAGE 2C: C0 (single-frame) vs C1 (K=3 temporal) FORECASTING COMPARISON')
    print('=' * 78)

    for name, records, scene_scores, core_r, bg_r in [
        ('C0', c0_records, c0_scene_scores, core_recall, bg_recall),
        ('C1', c1_records, c1_scene_scores, core_recall_c1, bg_recall_c1),
    ]:
        overall = summarize(records)
        agg = np.stack(scene_scores, axis=0).mean(0)
        names = ['empty', 'road', 'vehicle']
        ious = []
        print(f'\n=== {name}: overall metrics (n={overall["n"]}) ===')
        for j, cname in enumerate(names):
            tp, p, g = agg[j]
            iou_j = iou(tp, p, g)
            ious.append(iou_j)
            print(f'  scene_{cname}_iou = {iou_j:.4f}')
        print(f'  scene_mIoU (excl. empty) = {np.mean(ious[1:]):.4f}')
        print(f'  vehicle recall (core 3, t+1 state)        = {core_r["num"]/(core_r["den"]+1e-6):.4f}  (n_voxels={core_r["den"]})')
        print(f'  vehicle recall (background, t+1 state)    = {bg_r["num"]/(bg_r["den"]+1e-6):.4f}  (n_voxels={bg_r["den"]})')
        print(f'  target IoU={overall["iou"]:.4f} Dice={overall["dice"]:.4f} Precision={overall["precision"]:.4f} Recall={overall["recall"]:.4f}')
        print(f'  target centroid median err={overall["centroid_median"]:.2f}m  frac>5m={overall["frac_gt_5m"]:.3f}  frac>10m={overall["frac_gt_10m"]:.3f}')

    print_subset_table('C0', c0_records)
    print_subset_table('C1', c1_records)

    print('\n=== C1 temporal sanity checks (same checkpoint, perturbed eval-time input) ===')
    real_summary = summarize(c1_records)
    print(f'real (unperturbed): IoU={real_summary["iou"]:.4f}')
    for perturb_name, recs in c1_perturb_records.items():
        tp = sum(r['tp'] for r in recs); p = sum(r['p'] for r in recs); g = sum(r['g'] for r in recs)
        perturb_iou = iou(tp, p, g)
        degraded = perturb_iou < real_summary['iou'] - 0.02
        print(f'{perturb_name:<20} n={len(recs):<5} IoU={perturb_iou:.4f}  '
              f'{"DEGRADED (temporal info appears used)" if degraded else "NO CLEAR DEGRADATION (temporal info may not be used)"}')

    print('\n=== C0 vs C1 headline comparison ===')
    c0_summary = summarize(c0_records)
    c1_summary = summarize(c1_records)
    for metric in ('iou', 'dice', 'precision', 'recall', 'frac_gt_5m', 'frac_gt_10m'):
        c0v, c1v = c0_summary[metric], c1_summary[metric]
        delta = c1v - c0v
        print(f'  target_{metric}: C0={c0v:.4f}  C1={c1v:.4f}  delta(C1-C0)={delta:+.4f}')


if __name__ == '__main__':
    main()
