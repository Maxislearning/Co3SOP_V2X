#!/usr/bin/env python3
"""Stage 3 Step 1: offline extraction of GT and frozen-Stage-2B-Predicted
Scene/Target Occupancy for every (sample_id, link) Stage 2B/Beam already
uses -- no camera images, no gradients, run once.

Locks onto Stage 2B's ACTUAL train/val split (Audit 1c) by literally
building CarlaV2VTargetOccCo3SOP's train/val dataset objects and reading
their .data_infos -- never re-derives the 80/20 cut. Runs the frozen
checkpoint once per (sample_id,link) (1564 total, NOT 783+1564 -- Scene and
Target both come out of a single TargetAwareOccHead.forward() call, see
target_occ_head.py:75-93), saving Scene from whichever of a sample_id's two
link-forwards runs first and using the OTHER link's Scene output only for an
equality assertion (confirms the Scene branch is genuinely role-independent,
never both saved).

**2026-09-16 (Phase 2): world-axis canonicalization removed.** This step
used to rotate GT/Pred volumes to world axes via occ_canonicalize.py (see
that file's retirement note for why) -- the 4-panel beam redesign
(carla_V2V 任务13-17) makes the new beam label itself vehicle-ego-relative,
the same frame Stage 2B's Occupancy is already natively in, so no rotation
is needed or correct anymore. Volumes below are saved AS-IS from Stage 2B
(native RX-ego, RX-yaw-aligned), and the per-sample ego_yaw lookup that used
to feed the rotation is gone too.

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

CONFIG = 'projects/configs/co3sop_base/co3sop_base_carla_v2v_target_occ.py'
CKPT = 'work_dirs/carla_v2v_target_occ_phaseB/epoch_12.pth'
BEAM_CSV = '/home/admin0/carla_V2V/channel_analysis/channel_summary_from_carla_20260916_004652.csv'
OUT_DIR = '/home/admin0/carla_V2V/occupancy_prediction_beam/beam_occ_native'
# 任务25：--config/--beam-csv/--out-dir 可选覆盖（不传时完全等同旧行为），
# 用来跑合并数据集(co3sop_base_carla_v2v_target_occ_merged_v3.py)而不改
# 默认值——CKPT暂不做成可覆盖，这一步只是把Stage2B已训练好的checkpoint
# 拿去跑新场景的推理，Stage2B本身要不要用新数据重训是单独决定的事
# （新场景的Pred质量可能不佳，但GT不受影响，这一步的直接目的就是GT）。


def one_hot_scene(gt_occ):
    """gt_occ: [X,Y,Z] uint8 in {0,1,2} (empty/road/vehicle). -> [3,X,Y,Z] float32 one-hot."""
    out = np.zeros((3,) + gt_occ.shape, dtype=np.float32)
    for c in range(3):
        out[c] = (gt_occ == c).astype(np.float32)
    return out


def main():
    global BEAM_CSV, OUT_DIR, CONFIG, CKPT
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='cap total (sample_id,link) pairs processed, for a quick smoke run')
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--beam-csv', default=BEAM_CSV)
    parser.add_argument('--out-dir', default=OUT_DIR)
    parser.add_argument('--ckpt', default=CKPT, help='任务37: Stage2B checkpoint (e.g. a LORO-fold retrain)')
    args = parser.parse_args()
    CKPT = args.ckpt
    BEAM_CSV = args.beam_csv
    OUT_DIR = args.out_dir
    CONFIG = args.config

    cfg = Config.fromfile(args.config)
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
    # 任务25：合并后sample_id不再是1..783连续区间(legacy+多个offset过的
    # 新run)，改成按各来源自己的1000-block独立核对缺口，而不是硬编码
    # range(1,784)——原逻辑在只有一个来源(legacy)时给出完全相同的结果。
    covered = set(train_sids) | set(val_sids)
    for block in sorted(set(sid // 1000 for sid in covered)):
        lo, hi = (block * 1000 + 1, block * 1000 + 999) if block > 0 else (1, 999)
        block_covered = {sid for sid in covered if sid // 1000 == block}
        expected_range = range(min(block_covered), max(block_covered) + 1)
        missing = set(expected_range) - block_covered
        print(f'[audit 1c] block {block} ({min(block_covered)}-{max(block_covered)}): '
              f'missing from both splits: {sorted(missing)}')

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

    # 任务25：跑合并数据集时这个后台任务被系统内存压力连续kill了两次
    # （跟本脚本逻辑无关，是共享服务器上的瞬时事件，free -h在两次kill后
    # 都很快恢复正常）——原循环没有断点续跑，每次被kill都要从头全部重算
    # (forward pass是这里最贵的部分)。加一个基于输出文件是否已存在的
    # skip：两个target npz都在就跳过，连`ds[i]`(读图)都不用做。跳过的
    # sample不参与cross-link scene一致性assert(那个check是一次性验证，
    # smoke test里已全跑通过，resume时跳过不影响正确性)。最终计数改成
    # 跑完后直接扫盘统计，不再依赖跳过分支里维护一套增量计数器。
    already_skipped = 0
    for idx, (ds, i, sid, link, split_name) in enumerate(all_items):
        sid_dir_check = os.path.join(OUT_DIR, f'{sid:06d}')
        gt_p = os.path.join(sid_dir_check, f'{link}_gt_target.npz')
        pred_p = os.path.join(sid_dir_check, f'{link}_pred_target.npz')
        if os.path.exists(gt_p) and os.path.exists(pred_p):
            already_skipped += 1
            continue
        sample = ds[i]
        gt_occ = np.asarray(sample['gt_occ'])       # [X,Y,Z] uint8 {0,1,2}
        gt_target = np.asarray(sample['gt_target'])  # [X,Y,Z] {0,1}

        sid_dir = os.path.join(OUT_DIR, f'{sid:06d}')
        os.makedirs(sid_dir, exist_ok=True)

        with torch.no_grad():
            batch = scatter(collate([sample], samples_per_gpu=1), [0])[0]
            img_feats = model.extract_feat(img=batch['img'], img_metas=batch['img_metas'])
            preds = model.pts_bbox_head(img_feats, batch['img_metas'])
            scene_logits = preds['occ_preds'][-1][0].cpu().numpy()      # [3,X,Y,Z]
            target_logits = preds['target_preds'][-1][0].cpu().numpy()  # [2,X,Y,Z]

        # cross-link Scene equality check (role-independence assertion)
        # 任务37：比完就pop——原来每个sid的[3,X,Y,Z] float32 logits(~7.7MB)
        # 一直留在dict里，3875个sid累积到~30GB，被oomd杀(峰值39.9G)。
        if sid in scene_logits_by_sid:
            diff = float(np.abs(scene_logits - scene_logits_by_sid.pop(sid)).max())
            max_scene_diff = max(max_scene_diff, diff)
        else:
            scene_logits_by_sid[sid] = scene_logits
            # GT scene + Pred scene saved as-is, native RX-ego frame (no
            # canonicalization -- see occ_canonicalize.py's retirement note)
            gt_scene_1h = one_hot_scene(gt_occ)
            np.savez_compressed(os.path.join(sid_dir, 'gt_scene.npz'),
                                 onehot=gt_scene_1h.astype(np.uint8))

            pred_scene_probs = F.softmax(torch.from_numpy(scene_logits), dim=0).numpy()
            np.savez_compressed(os.path.join(sid_dir, 'pred_scene.npz'),
                                 probs=pred_scene_probs.astype(np.float16))
            n_written_scene += 1

        # GT + Pred target, per link -- native RX-ego frame, as-is
        gt_target_1c = gt_target.astype(np.float32)[None]  # [1,X,Y,Z]
        np.savez_compressed(os.path.join(sid_dir, f'{link}_gt_target.npz'),
                             mask=gt_target_1c.astype(np.uint8))

        pred_target_prob = F.softmax(torch.from_numpy(target_logits), dim=0).numpy()[1:2]  # [1,X,Y,Z]
        np.savez_compressed(os.path.join(sid_dir, f'{link}_pred_target.npz'),
                             prob=pred_target_prob.astype(np.float16))
        n_written_target += 1
        del sample, gt_occ, gt_target, batch, img_feats, preds, scene_logits, target_logits

        if (idx + 1) % 50 == 0:
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        if (idx + 1) % 200 == 0:
            print(f'  [{idx+1}/{len(all_items)}]  (skipped {already_skipped} already-done)', file=sys.stderr)

    # 断点续跑后，最终计数直接扫盘统计（覆盖本次运行里的增量计数器，
    # 后者在resume时只反映本次新写的部分，不是全量）。
    all_target_files = 0
    all_scene_files = 0
    for sid_dir_name in os.listdir(OUT_DIR):
        p = os.path.join(OUT_DIR, sid_dir_name)
        if not os.path.isdir(p):
            continue
        if os.path.exists(os.path.join(p, 'gt_scene.npz')):
            all_scene_files += 1
        all_target_files += sum(1 for f in os.listdir(p) if f.endswith('_gt_target.npz'))
    print(f'\n[done] scene volumes on disk: {all_scene_files} (expect {len(train_sids)+len(val_sids)})')
    print(f'[done] target volumes on disk: {all_target_files} (expect {len(all_items)})')
    print(f'[done] this run: newly wrote {n_written_scene} scene / {n_written_target} target, '
          f'skipped {already_skipped} already-done')
    print(f'[audit] max |scene_logits diff| between the two links of the same sample_id: {max_scene_diff:.6f} '
          f'(expect ~0 -- confirms Scene branch is role-independent; only checked for items actually '
          f'(re)computed this run, not skipped ones)')
    assert max_scene_diff < 1e-3, 'Scene branch output differs between links of the same sample -- role leakage?'

    write_version_manifest()


def write_version_manifest():
    """So a future reader of OUT_DIR knows which beam CSV/checkpoint/panel
    config it corresponds to without re-deriving it from git history."""
    import hashlib
    import json
    from datetime import datetime

    def sha256_of(path):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()

    sys.path.insert(0, '/home/admin0/carla_V2V/src')
    import panel_beamforming as pb

    manifest = {
        'generated_at': datetime.now().isoformat(),
        'out_dir': OUT_DIR,
        'beam_csv': BEAM_CSV,
        'beam_csv_sha256': sha256_of(BEAM_CSV),
        'stage2b_config': CONFIG,
        'stage2b_checkpoint': CKPT,
        'stage2b_checkpoint_sha256': sha256_of(CKPT),
        'native_ego_frame': True,
        'canonicalization': False,
        'num_panels': pb.NUM_PANELS,
        'beams_per_panel': pb.BEAMS_PER_PANEL,
        'num_classes': pb.NUM_PANELS * pb.BEAMS_PER_PANEL,
        'panel_enum': {name: idx for idx, name in pb.PANEL_NAMES.items()},
    }
    manifest_path = os.path.join(OUT_DIR, 'dataset_version.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f'[done] wrote version manifest: {manifest_path}')


if __name__ == '__main__':
    main()
