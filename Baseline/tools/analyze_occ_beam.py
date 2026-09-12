#!/usr/bin/env python3
"""Stage 3 Step 6: full analysis. B0 majority (global + per-link) / old
geometry+visual reference / GT-{scene,target,scene_target} /
Pred-{scene,target,scene_target}, subset breakdown, and dependency/shortcut
checks on the trained Pred-scene_target model.

Current-frame only (t, no t+1 anywhere) -- LOS/high-yaw/boundary/dense-
traffic thresholds are the same VALUES as analyze_temporal_c0_c1.py's
(30deg / 3 vehicles / 37m) but computed from frame t itself, not a future
state (Stage 3 has no forecasting concept at all).
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/home/admin0/carla_V2V/src')

from projects.mmdet3d_plugin_carla_v2v.datasets.occ_beam_dataset import OccBeamDataset, BEAM_CSV
from projects.mmdet3d_plugin.co3sop_base.dense_heads.occ_beam_encoder import OccupancyBeamEncoder

VEHICLE_LOG = '/home/admin0/data/run_20260904_142344_bigcar80/rear_vehicle_log.csv'
BACKGROUND_LOG = '/home/admin0/data/run_20260904_142344_bigcar80/background_vehicles_log.csv'
LINK_VEHICLE = {'TX_CAR1': 1, 'TX_CAR2': 3}
RX_VEHICLE = 2
YAW_THRESH_DEG = 30.0
DENSE_TRAFFIC_THRESH = 3
DENSE_TRAFFIC_RADIUS = 40.0
BOUNDARY_THRESH = 37.0
CHANNELS_TO_CIN = {'scene': 3, 'target': 1, 'scene_target': 4}


def wrap_deg(d):
    return float(np.degrees(np.arctan2(np.sin(np.radians(d)), np.cos(np.radians(d)))))


def topk_accs(tx_gt, rx_gt, tx_top5, rx_top5, prefix=''):
    out = {}
    for k in (1, 3, 5):
        out[f'{prefix}tx_top{k}_acc'] = float((tx_top5[:, :k] == tx_gt[:, None]).any(1).mean())
        out[f'{prefix}rx_top{k}_acc'] = float((rx_top5[:, :k] == rx_gt[:, None]).any(1).mean())
    return out


def compute_subsets(infos, veh_df, bg_by_sid):
    out = {}
    for sid, link in infos:
        tx_vn = LINK_VEHICLE[link]
        rec = {}
        if sid in veh_df.index:
            row = veh_df.loc[sid]
            rel_yaw = wrap_deg(float(row[f'vehicle_{tx_vn}_yaw']) - float(row.vehicle_2_yaw))
            rec['high_yaw'] = abs(rel_yaw) > YAW_THRESH_DEG
            rx_xy = np.array([row.vehicle_2_x, row.vehicle_2_y])
            tx_xy = np.array([row[f'vehicle_{tx_vn}_x'], row[f'vehicle_{tx_vn}_y']])
            dist = float(np.linalg.norm(tx_xy - rx_xy))
            rec['boundary'] = dist > BOUNDARY_THRESH  # distance from RX as a proxy (current-frame, RX-centered)
        else:
            rec['high_yaw'] = None
            rec['boundary'] = None
        if sid in veh_df.index and sid in bg_by_sid:
            rx_xy = np.array([veh_df.loc[sid].vehicle_2_x, veh_df.loc[sid].vehicle_2_y])
            bg = bg_by_sid[sid]
            d = np.sqrt((bg.x.values - rx_xy[0]) ** 2 + (bg.y.values - rx_xy[1]) ** 2)
            rec['dense_traffic'] = int((d < DENSE_TRAFFIC_RADIUS).sum()) >= DENSE_TRAFFIC_THRESH
        else:
            rec['dense_traffic'] = None
        out[(sid, link)] = rec
    return out


def evaluate_occ_model(work_dir, source, channels, ckpt_name, subsets, beam_df):
    ds = OccBeamDataset(split='val', source=source, channels=channels)
    model = OccupancyBeamEncoder(in_channels=CHANNELS_TO_CIN[channels])
    state = torch.load(os.path.join(work_dir, ckpt_name), map_location='cpu')
    model.load_state_dict(state)
    model.eval()

    records = []
    with torch.no_grad():
        for i in range(len(ds)):
            vol, tx, rx, meta = ds[i]
            tx_logits, rx_logits = model(vol.unsqueeze(0))
            k = min(5, tx_logits.shape[1])
            tx_top5 = tx_logits.topk(k, dim=1).indices[0].numpy()
            rx_top5 = rx_logits.topk(k, dim=1).indices[0].numpy()
            rec = dict(subsets[(meta['sample_id'], meta['link'])])
            rec.update(link=meta['link'], tx_gt=int(tx), rx_gt=int(rx), tx_top5=tx_top5, rx_top5=rx_top5)
            row = beam_df[(beam_df.sample_id == meta['sample_id']) & (beam_df.tx_name == meta['link'])].iloc[0]
            rec['is_los'] = bool(row.is_los)
            records.append(rec)
    return records


def summarize_records(records, pred_fn=lambda r: True):
    sub = [r for r in records if pred_fn(r)]
    if not sub:
        return None
    tx_gt = np.array([r['tx_gt'] for r in sub])
    rx_gt = np.array([r['rx_gt'] for r in sub])
    tx_top5 = np.stack([np.pad(r['tx_top5'], (0, 5 - len(r['tx_top5'])), constant_values=-1) for r in sub])
    rx_top5 = np.stack([np.pad(r['rx_top5'], (0, 5 - len(r['rx_top5'])), constant_values=-1) for r in sub])
    out = topk_accs(tx_gt, rx_gt, tx_top5, rx_top5)
    out['n'] = len(sub)
    return out


SUBSET_DEFS = [
    ('Overall', lambda r: True),
    ('FRONT', lambda r: r['link'] == 'TX_CAR1'),
    ('REAR', lambda r: r['link'] == 'TX_CAR2'),
    ('LOS', lambda r: r['is_los'] is True),
    ('NLOS', lambda r: r['is_los'] is False),
    ('boundary', lambda r: r['boundary'] is True),
    ('high_yaw', lambda r: r['high_yaw'] is True),
    ('dense_traffic', lambda r: r['dense_traffic'] is True),
]


def print_table(name, records):
    print(f'\n--- {name} ---')
    for subset_name, pred_fn in SUBSET_DEFS:
        s = summarize_records(records, pred_fn)
        if s is None:
            print(f'{subset_name:<15} (no samples)')
            continue
        print(f'{subset_name:<15} n={s["n"]:<5} '
              f'tx_top1={s["tx_top1_acc"]:.3f} tx_top3={s["tx_top3_acc"]:.3f} tx_top5={s["tx_top5_acc"]:.3f}  '
              f'rx_top1={s["rx_top1_acc"]:.3f} rx_top3={s["rx_top3_acc"]:.3f} rx_top5={s["rx_top5_acc"]:.3f}')


def majority_baseline(beam_df, train_infos, val_infos):
    print('\n=== B0: majority baseline (fit train, eval val) ===')
    train_by_link = {'TX_CAR1': [], 'TX_CAR2': []}
    for sid, link in train_infos:
        row = beam_df[(beam_df.sample_id == sid) & (beam_df.tx_name == link)].iloc[0]
        train_by_link[link].append((row.optimal_tx_beam_idx, row.optimal_rx_beam_idx))

    global_tx = pd.Series([x[0] for v in train_by_link.values() for x in v]).value_counts().index[0]
    global_rx = pd.Series([x[1] for v in train_by_link.values() for x in v]).value_counts().index[0]
    per_link_majority = {
        link: (pd.Series([x[0] for x in v]).value_counts().index[0],
               pd.Series([x[1] for x in v]).value_counts().index[0])
        for link, v in train_by_link.items()
    }

    for name, get_pred in [
        ('global majority', lambda link: (global_tx, global_rx)),
        ('per-link majority', lambda link: per_link_majority[link]),
    ]:
        correct_tx = correct_rx = 0
        for sid, link in val_infos:
            row = beam_df[(beam_df.sample_id == sid) & (beam_df.tx_name == link)].iloc[0]
            pred_tx, pred_rx = get_pred(link)
            correct_tx += int(pred_tx == row.optimal_tx_beam_idx)
            correct_rx += int(pred_rx == row.optimal_rx_beam_idx)
        n = len(val_infos)
        print(f'{name}: tx_top1={correct_tx/n:.4f}  rx_top1={correct_rx/n:.4f}')


def old_reference_model(subsets):
    print('\n=== Old geometry+visual reference model (work_dirs/carla_v2v_beam/epoch_12.pth) ===')
    try:
        from mmcv import Config
        from mmcv.parallel import collate, scatter
        from mmcv.runner import load_checkpoint
        from mmdet3d.datasets import build_dataset
        from mmdet3d.models import build_model
        import projects.mmdet3d_plugin  # noqa: F401

        cfg = Config.fromfile('projects/configs/co3sop_base/co3sop_base_carla_v2v_beam.py')
        ds = build_dataset(cfg.data.val)
        model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        load_checkpoint(model, 'work_dirs/carla_v2v_beam/epoch_12.pth', map_location='cpu', strict=False)
        model = model.cuda().eval()

        results = []
        with torch.no_grad():
            for i in range(len(ds)):
                sample = ds[i]
                batch = scatter(collate([sample], samples_per_gpu=1), [0])[0]
                out = model.forward_test(img_metas=batch['img_metas'], img=batch['img'], gt_beam=batch['gt_beam'].unsqueeze(0)
                                          if batch['gt_beam'].dim() == 1 else batch['gt_beam'])
                results.extend(out['evaluation'])
        metrics = ds.evaluate(results)
        for k, v in metrics.items():
            print(f'  {k}: {v}')
    except Exception as e:
        print(f'  [skipped] could not evaluate old reference model: {e}')


def dependency_checks(work_dir_b3_source, source, subsets):
    print(f'\n=== Dependency/shortcut checks on trained Pred-scene_target (source={source}) ===')
    ds = OccBeamDataset(split='val', source=source, channels='scene_target')
    model = OccupancyBeamEncoder(in_channels=4)
    state = torch.load(os.path.join(work_dir_b3_source, 'last.pth'), map_location='cpu')
    model.load_state_dict(state)
    model.eval()

    def score(perturb):
        rng = np.random.default_rng(0)
        n_correct_tx = n_correct_rx = 0
        donor_pool = []
        with torch.no_grad():
            for i in range(len(ds)):
                vol, tx, rx, meta = ds[i]
                v = vol.clone()
                if perturb == 'scene_zero':
                    v[:3] = 0
                elif perturb == 'target_zero':
                    v[3:] = 0
                elif perturb == 'scene_shuffle' and donor_pool:
                    donor = donor_pool[rng.integers(len(donor_pool))]
                    v[:3] = donor[:3]
                elif perturb == 'target_shuffle' and donor_pool:
                    donor = donor_pool[rng.integers(len(donor_pool))]
                    v[3:] = donor[3:]
                donor_pool.append(vol.clone())
                if len(donor_pool) > 20:
                    donor_pool.pop(0)
                tx_logits, rx_logits = model(v.unsqueeze(0))
                n_correct_tx += int(tx_logits.argmax(1).item() == int(tx))
                n_correct_rx += int(rx_logits.argmax(1).item() == int(rx))
        return n_correct_tx / len(ds), n_correct_rx / len(ds)

    for perturb in ('none', 'scene_zero', 'target_zero', 'scene_shuffle', 'target_shuffle'):
        tx_acc, rx_acc = score(None if perturb == 'none' else perturb)
        print(f'  {perturb:<15} tx_top1={tx_acc:.4f}  rx_top1={rx_acc:.4f}')


def main():
    veh_df = pd.read_csv(VEHICLE_LOG).set_index('sample_id')
    bg_df = pd.read_csv(BACKGROUND_LOG)
    bg_by_sid = {sid: g for sid, g in bg_df.groupby('sample_id')}
    beam_df = pd.read_csv(BEAM_CSV)
    beam_df = beam_df[beam_df['tx_name'].isin(['TX_CAR1', 'TX_CAR2'])]

    train_ds_probe = OccBeamDataset(split='train', source='gt', channels='target')
    val_ds_probe = OccBeamDataset(split='val', source='gt', channels='target')
    train_infos = train_ds_probe.infos
    val_infos = val_ds_probe.infos

    majority_baseline(beam_df, train_infos, val_infos)
    old_reference_model(None)

    subsets = compute_subsets(train_infos + val_infos, veh_df, bg_by_sid)

    print('\n' + '=' * 78)
    print('GT / Predicted Occupancy Beam Encoder comparison (last.pth = fully-trained 30-epoch checkpoint)')
    print('=' * 78)
    for source in ('gt', 'pred'):
        for channels in ('scene', 'target', 'scene_target'):
            work_dir = f'work_dirs/occ_beam_{source}_{channels}'
            for ckpt in ('last.pth', 'best.pth'):
                records = evaluate_occ_model(work_dir, source, channels, ckpt, subsets, beam_df)
                print_table(f'{source}/{channels} [{ckpt}]', records)

    dependency_checks('work_dirs/occ_beam_pred_scene_target', 'pred', subsets)


if __name__ == '__main__':
    main()
