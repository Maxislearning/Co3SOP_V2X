#!/usr/bin/env python3
"""Stage 2C-v2 full analysis: replays C0, C1-v1, and C1-v2 on the same
matched val set, reports the standard metric table + subset breakdown for
all three, then runs C1-v2's temporal sanity checks with the two corrections
from the approved v2 review:

1. Feature-level shuffled-history (not v1's flawed image-tensor-swap-with-
   original-img_metas approach, which mixed up donor content with a
   temporal_trans_to_t matrix that was neither donor's nor original's own
   relationship): the donor sample's k=0/1 frames go through their OWN full
   pipeline (own backbone, own intra-frame Co3SOP/trans2ego fusion, own
   temporal_trans_to_t alignment into the DONOR's own current frame) to
   produce donor_aligned = [H2_donor, H1_donor]. Only the temporal_fusion
   INPUT gets spliced: [H2_donor, H1_donor, C_original] -- current-frame
   feature C, target_role, and GT all stay the original sample's own.

2. misaligned-history (new, distinct from shuffled): the ORIGINAL sample's
   own real H2/H1 (real content, real per-frame calibration), but warped
   with an identity matrix instead of the real temporal_trans_to_t
   (warp_fused_to_t(..., identity=True)) -- isolates "does the model care
   about correct geometric alignment of otherwise-correct content", as
   distinct from "does it care whether the content belongs to this sample".

3. Paired per-sample metrics + paired bootstrap 95% CI for
   correct vs {shuffled, misaligned, zero_history, no_history}, not just a
   comparison of aggregate IoU numbers -- a tiny aggregate gap (e.g. 0.681
   vs 0.680) must not be read as a pass.

4. Gate (g1/g2) activation diagnostics: mean/std/saturation fraction, and
   whether they differ between correct-history and shuffled-history inputs.

Usage: python tools/analyze_temporal_c1_v2_sanity.py [--limit N] [--n-boot 2000]
"""
import argparse
import os
import sys

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tools/ itself, for the analyze_temporal_c0_c1 import below
sys.path.insert(0, '/home/admin0/carla_V2V/src')
import projects.mmdet3d_plugin  # noqa: F401
import projects.mmdet3d_plugin_carla_v2v  # noqa: F401
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from analyze_temporal_c0_c1 import (  # noqa: E402 -- reuse Stage 2C's already-verified shared logic, don't duplicate
    C0_CONFIG, C0_CKPT, C1_CONFIG, C1_CKPT, VEHICLE_LOG, BACKGROUND_LOG, BEAM_CSV,
    iou, mask_centroid_xy, build_future_reference_masks, compute_subsets,
    run_c0_forward, summarize, print_subset_table, SUBSET_DEFS,
)
from projects.mmdet3d_plugin.co3sop_base.dense_heads.temporal_target_occ_head import warp_fused_to_t
import pandas as pd

C1_V2_CONFIG = 'projects/configs/co3sop_base/co3sop_base_carla_v2v_temporal_c1_v2.py'
C1_V2_CKPT = 'work_dirs/carla_v2v_temporal_c1_v2/epoch_12.pth'


def paired_bootstrap_ci(x, y, n_boot=2000, seed=0):
    """x, y: paired per-sample arrays (e.g. per-sample IoU under 'correct'
    and under some perturbation). Returns (mean_diff, ci_lo, ci_hi) for
    mean(x-y) via percentile bootstrap over sample indices."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x); y = np.asarray(y)
    n = len(x)
    diffs = x - y
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = diffs[idx].mean()
    return float(diffs.mean()), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def per_sample_iou(tp, p, g, eps=1e-6):
    return float(tp / (p + g - tp + eps))


def encode_and_warp_all(model, img, img_metas, identity_history=False):
    """Returns (aligned_feats=[H2,H1,C], role) for one sample's queue.
    identity_history=True -> warp_fused_to_t(..., identity=True) for the
    history frames (k=0,1) only -- used for the misaligned-history check."""
    head = model.pts_bbox_head
    K = img.shape[1]
    frame_metas = [[m[k] for m in img_metas] for k in range(K)]
    aligned = []
    with torch.no_grad():
        for k in range(K):
            feats_k = model.extract_feat(img=img[:, k], img_metas=frame_metas[k])
            fused_k, _, _ = head._encode_and_fuse(feats_k, frame_metas[k])
            use_identity = identity_history and k < K - 1
            aligned.append(warp_fused_to_t(fused_k, frame_metas[k], identity=use_identity))
    role = int(frame_metas[-1][0]['target_role'])
    return aligned, role, frame_metas


def decode_from_aligned(model, aligned_feats, frame_metas_last):
    head = model.pts_bbox_head
    with torch.no_grad():
        fused_temporal, (g1, g2) = head.fuse_aligned(aligned_feats)
        outputs = head._run_deblocks(fused_temporal)
        occ_preds = [head.occ[i](outputs[i]) for i in range(len(outputs))]
        role = fused_temporal.new_tensor([m['target_role'] for m in frame_metas_last], dtype=torch.long)
        role_embed = head.role_embedding(role)
        target_preds = []
        for i in range(len(outputs)):
            scale, shift = head.film_proj[i](role_embed).chunk(2, dim=1)
            scale = scale[:, :, None, None, None]; shift = shift[:, :, None, None, None]
            modulated = outputs[i] * (1 + scale) + shift
            target_preds.append(head.target_occ[i](modulated))
    return {'occ_preds': occ_preds, 'target_preds': target_preds}, (g1, g2)


def zero_raw_history(img, k_range):
    img = img.clone()
    for k in k_range:
        img[:, k] = 0
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--n-boot', type=int, default=2000)
    parser.add_argument('--donor-pool', type=int, default=20)
    args = parser.parse_args()

    veh_df = pd.read_csv(VEHICLE_LOG).set_index('sample_id')
    bg_df = pd.read_csv(BACKGROUND_LOG)
    bg_by_sid = {sid: g for sid, g in bg_df.groupby('sample_id')}
    beam_df = pd.read_csv(BEAM_CSV)

    print('Loading C0...', file=sys.stderr)
    c0_cfg = Config.fromfile(C0_CONFIG)
    c0_ds = build_dataset(c0_cfg.data.val)
    c0_model = build_model(c0_cfg.model, train_cfg=c0_cfg.get('train_cfg'), test_cfg=c0_cfg.get('test_cfg'))
    load_checkpoint(c0_model, C0_CKPT, map_location='cpu', strict=False)
    c0_model = c0_model.cuda().eval()

    print('Loading C1-v1...', file=sys.stderr)
    c1_cfg = Config.fromfile(C1_CONFIG)
    c1_ds = build_dataset(c1_cfg.data.val)
    c1_model = build_model(c1_cfg.model, train_cfg=c1_cfg.get('train_cfg'), test_cfg=c1_cfg.get('test_cfg'))
    load_checkpoint(c1_model, C1_CKPT, map_location='cpu', strict=False)
    c1_model = c1_model.cuda().eval()

    print('Loading C1-v2...', file=sys.stderr)
    c1v2_cfg = Config.fromfile(C1_V2_CONFIG)
    c1v2_ds = build_dataset(c1v2_cfg.data.val)
    c1v2_model = build_model(c1v2_cfg.model, train_cfg=c1v2_cfg.get('train_cfg'), test_cfg=c1v2_cfg.get('test_cfg'))
    load_checkpoint(c1v2_model, C1_V2_CKPT, map_location='cpu', strict=False)
    c1v2_model = c1v2_model.cuda().eval()

    c0_infos = [(int(info[2]), info[3]) for info in c0_ds.data_infos]
    c1_infos = [(int(info[2]), info[3]) for info in c1_ds.data_infos]
    c1v2_infos = [(int(info[2]), info[3]) for info in c1v2_ds.data_infos]
    assert sorted(c0_infos) == sorted(c1_infos) == sorted(c1v2_infos), \
        'C0/C1-v1/C1-v2 val sample sets differ -- comparison would not be fair'
    print(f'[fairness check] C0/C1-v1/C1-v2 val sets match exactly: {len(c0_infos)} samples', file=sys.stderr)

    subsets = compute_subsets(c0_infos, veh_df, bg_by_sid, beam_df)

    def score_records(ds, model, forward_fn, tag):
        n = min(args.limit, len(ds)) if args.limit else len(ds)
        records, scene_scores = [], []
        core_r, bg_r = {'num': 0, 'den': 0}, {'num': 0, 'den': 0}
        for i in range(n):
            sample = ds[i]
            key_gt_target = 'gt_target' if 'gt_target' in sample else 'gt_target_future'
            key_gt_occ = 'gt_occ' if 'gt_occ' in sample else 'gt_occ_future'
            gt_target = np.asarray(sample[key_gt_target])
            gt_occ = np.asarray(sample[key_gt_occ])
            info = ds.data_infos[i]
            t, link = int(info[2]), info[3]

            preds = forward_fn(sample)
            target_pred = torch.argmax(preds['target_preds'][-1], dim=1)[0].cpu().numpy()
            scene_pred = torch.argmax(preds['occ_preds'][-1], dim=1)[0].cpu().numpy()

            tp = int(((target_pred == 1) & (gt_target == 1)).sum())
            p = int((target_pred == 1).sum()); g = int((gt_target == 1).sum())
            pred_c = mask_centroid_xy(target_pred == 1)
            gt_center = subsets[(t, link)]['target_center_raw']
            err = float(np.hypot(pred_c[0] - gt_center[0], pred_c[1] - gt_center[1])) \
                if (pred_c is not None and gt_center is not None) else float('inf')

            rec = dict(subsets[(t, link)]); rec.update(link=link, t=t, tp=tp, p=p, g=g, centroid_err=err)
            records.append(rec)

            score = np.zeros((3, 3))
            for j in range(3):
                score[j, 0] += ((gt_occ == j) & (scene_pred == j)).sum()
                score[j, 1] += (gt_occ == j).sum(); score[j, 2] += (scene_pred == j).sum()
            scene_scores.append(score)

            core_mask, background_mask = build_future_reference_masks(t, veh_df, bg_by_sid)
            if core_mask is not None:
                pred_vehicle = scene_pred == 2
                core_r['num'] += int((pred_vehicle & core_mask).sum()); core_r['den'] += int(core_mask.sum())
                bg_r['num'] += int((pred_vehicle & background_mask).sum()); bg_r['den'] += int(background_mask.sum())

            if (i + 1) % 50 == 0:
                print(f'  {tag} [{i+1}/{n}]', file=sys.stderr)
        return records, scene_scores, core_r, bg_r

    def c0_fwd(sample):
        batch = scatter(collate([sample], samples_per_gpu=1), [0])[0]
        return run_c0_forward(c0_model, batch)

    def c1_fwd(sample):
        batch = scatter(collate([sample], samples_per_gpu=1), [0])[0]
        img = batch['img'].cuda(); img_metas = batch['img_metas']
        mcar_feats_list, img_metas_list = c1_model._extract_queue_feats(img, img_metas)
        with torch.no_grad():
            return c1_model.pts_bbox_head(mcar_feats_list, img_metas_list)

    c0_records, c0_scene, c0_core, c0_bg = score_records(c0_ds, c0_model, c0_fwd, 'C0')
    del c0_model; torch.cuda.empty_cache()
    c1_records, c1_scene, c1_core, c1_bg = score_records(c1_ds, c1_model, c1_fwd, 'C1-v1')
    del c1_model; torch.cuda.empty_cache()

    # ---- C1-v2: correct + all perturbations, per-sample paired records ----
    # IMPORTANT: keyed by sample index i (dict), NOT appended to a plain
    # list. 'shuffled' has no donor available for i=0 (empty pool) and would
    # otherwise be skipped there while every other perturbation still
    # appends -- with plain lists that silently shifts every later index out
    # of alignment with 'correct' for the paired-bootstrap comparison (a
    # real bug caught before ever running this, not after). Dict-keyed
    # records + explicit common-key intersection when pairing makes this
    # structurally impossible regardless of which perturbations skip which
    # indices.
    n2 = min(args.limit, len(c1v2_ds)) if args.limit else len(c1v2_ds)
    perturb_names = ['correct', 'zero_history', 'no_history', 'shuffled', 'misaligned', 'reversed_order']
    v2_records = {name: {} for name in perturb_names}
    v2_scene_correct, v2_core_correct, v2_bg_correct = [], {'num': 0, 'den': 0}, {'num': 0, 'den': 0}
    gate_stats = {'correct': [], 'shuffled': []}
    donor_pool = []  # list of (donor_i, aligned_H2, aligned_H1) from previous samples, for shuffled-history

    for i in range(n2):
        sample = c1v2_ds[i]
        gt_target = np.asarray(sample['gt_target_future'])
        gt_occ = np.asarray(sample['gt_occ_future'])
        info = c1v2_ds.data_infos[i]
        t, link = int(info[2]), info[3]
        rec_base = dict(subsets[(t, link)]); rec_base.update(link=link, t=t)

        batch = scatter(collate([sample], samples_per_gpu=1), [0])[0]
        img = batch['img'].cuda(); img_metas = batch['img_metas']

        aligned_correct, role, frame_metas = encode_and_warp_all(c1v2_model, img, img_metas)
        preds_correct, (g1c, g2c) = decode_from_aligned(c1v2_model, aligned_correct, frame_metas[-1])
        gate_stats['correct'].append((g1c.mean().item(), g1c.std().item(), g2c.mean().item(), g2c.std().item()))

        gt_center = subsets[(t, link)]['target_center_raw']

        def score(preds):
            tp_ = torch.argmax(preds['target_preds'][-1], dim=1)[0].cpu().numpy()
            sc_ = torch.argmax(preds['occ_preds'][-1], dim=1)[0].cpu().numpy()
            tp = int(((tp_ == 1) & (gt_target == 1)).sum())
            p = int((tp_ == 1).sum()); g = int((gt_target == 1).sum())
            pred_c = mask_centroid_xy(tp_ == 1)
            err = float(np.hypot(pred_c[0] - gt_center[0], pred_c[1] - gt_center[1])) \
                if (pred_c is not None and gt_center is not None) else float('inf')
            return dict(rec_base, tp=tp, p=p, g=g, centroid_err=err), sc_

        rec_correct, scene_pred_correct = score(preds_correct)
        v2_records['correct'][i] = rec_correct
        score_arr = np.zeros((3, 3))
        for j in range(3):
            score_arr[j, 0] += ((gt_occ == j) & (scene_pred_correct == j)).sum()
            score_arr[j, 1] += (gt_occ == j).sum(); score_arr[j, 2] += (scene_pred_correct == j).sum()
        v2_scene_correct.append(score_arr)
        core_mask, background_mask = build_future_reference_masks(t, veh_df, bg_by_sid)
        if core_mask is not None:
            pv = scene_pred_correct == 2
            v2_core_correct['num'] += int((pv & core_mask).sum()); v2_core_correct['den'] += int(core_mask.sum())
            v2_bg_correct['num'] += int((pv & background_mask).sum()); v2_bg_correct['den'] += int(background_mask.sum())

        # zero_history: raw images for k=0,1 zeroed, full pipeline rerun
        img_zeroed = zero_raw_history(img, [0, 1])
        aligned_zero, _, _ = encode_and_warp_all(c1v2_model, img_zeroed, img_metas)
        preds_zero, _ = decode_from_aligned(c1v2_model, aligned_zero, frame_metas[-1])
        v2_records['zero_history'][i] = score(preds_zero)[0]

        # no_history: aligned history features zeroed post-warp
        aligned_no = [torch.zeros_like(aligned_correct[0]), torch.zeros_like(aligned_correct[1]), aligned_correct[2]]
        preds_no, _ = decode_from_aligned(c1v2_model, aligned_no, frame_metas[-1])
        v2_records['no_history'][i] = score(preds_no)[0]

        # misaligned: own real content, identity warp for history frames
        aligned_mis, _, _ = encode_and_warp_all(c1v2_model, img, img_metas, identity_history=True)
        preds_mis, _ = decode_from_aligned(c1v2_model, aligned_mis, frame_metas[-1])
        v2_records['misaligned'][i] = score(preds_mis)[0]

        # reversed_order: swap which aligned feature plays H2 vs H1 (both still real, own content)
        aligned_rev = [aligned_correct[1], aligned_correct[0], aligned_correct[2]]
        preds_rev, _ = decode_from_aligned(c1v2_model, aligned_rev, frame_metas[-1])
        v2_records['reversed_order'][i] = score(preds_rev)[0]

        # shuffled: donor's own aligned H2/H1 (from an earlier sample in this
        # loop, already fully encoded+fused+warped through ITS OWN pipeline)
        # + this sample's own C. No donor available for i==0 -> that index
        # simply has no 'shuffled' entry at all (handled by common-key
        # intersection at pairing time, not a silent index shift).
        if donor_pool:
            _donor_i, donor_h2, donor_h1 = donor_pool[np.random.randint(len(donor_pool))]
            aligned_shuf = [donor_h2, donor_h1, aligned_correct[2]]
            preds_shuf, (g1s, g2s) = decode_from_aligned(c1v2_model, aligned_shuf, frame_metas[-1])
            v2_records['shuffled'][i] = score(preds_shuf)[0]
            gate_stats['shuffled'].append((g1s.mean().item(), g1s.std().item(), g2s.mean().item(), g2s.std().item()))
        donor_pool.append((i, aligned_correct[0].clone(), aligned_correct[1].clone()))
        if len(donor_pool) > args.donor_pool:
            donor_pool.pop(0)

        if (i + 1) % 50 == 0:
            print(f'  C1-v2 [{i+1}/{n2}]', file=sys.stderr)

    print('\n' + '=' * 78)
    print('STAGE 2C-v2: C0 vs C1-v1 vs C1-v2 (gated residual) FORECASTING COMPARISON')
    print('=' * 78)

    v2_correct_list = list(v2_records['correct'].values())
    for name, records, scene_scores, core_r, bg_r in [
        ('C0', c0_records, c0_scene, c0_core, c0_bg),
        ('C1-v1', c1_records, c1_scene, c1_core, c1_bg),
        ('C1-v2 (correct)', v2_correct_list, v2_scene_correct, v2_core_correct, v2_bg_correct),
    ]:
        overall = summarize(records)
        agg = np.stack(scene_scores, axis=0).mean(0)
        names = ['empty', 'road', 'vehicle']
        ious = []
        print(f'\n=== {name}: overall metrics (n={overall["n"]}) ===')
        for j, cname in enumerate(names):
            tp, p, g = agg[j]
            iou_j = iou(tp, p, g); ious.append(iou_j)
            print(f'  scene_{cname}_iou = {iou_j:.4f}')
        print(f'  scene_mIoU (excl. empty) = {np.mean(ious[1:]):.4f}')
        print(f'  vehicle recall (core 3, t+1 state)     = {core_r["num"]/(core_r["den"]+1e-6):.4f}')
        print(f'  vehicle recall (background, t+1 state) = {bg_r["num"]/(bg_r["den"]+1e-6):.4f}')
        print(f'  target IoU={overall["iou"]:.4f} Dice={overall["dice"]:.4f} Precision={overall["precision"]:.4f} Recall={overall["recall"]:.4f}')
        print(f'  target centroid median err={overall["centroid_median"]:.2f}m  frac>5m={overall["frac_gt_5m"]:.3f}  frac>10m={overall["frac_gt_10m"]:.3f}')

    print_subset_table('C1-v2 (correct)', v2_correct_list)

    print('\n=== C1-v2 temporal sanity checks: aggregate IoU ===')
    correct_summary = summarize(v2_correct_list)
    print(f'correct (real, aligned, own-sample history): IoU={correct_summary["iou"]:.4f}  n={correct_summary["n"]}')
    for name in ('zero_history', 'no_history', 'shuffled', 'misaligned', 'reversed_order'):
        s = summarize(list(v2_records[name].values()))
        print(f'{name:<16} n={s["n"]:<5} IoU={s["iou"]:.4f}  delta(correct-this)={correct_summary["iou"]-s["iou"]:+.4f}')

    def paired_arrays(name):
        """Common-key intersection between 'correct' and `name` -- robust to
        either dict having missing indices (e.g. 'shuffled' has none for
        i==0), never a silent positional shift."""
        common = sorted(set(v2_records['correct']) & set(v2_records[name]))
        x = [per_sample_iou(v2_records['correct'][k]['tp'], v2_records['correct'][k]['p'], v2_records['correct'][k]['g']) for k in common]
        y = [per_sample_iou(v2_records[name][k]['tp'], v2_records[name][k]['p'], v2_records[name][k]['g']) for k in common]
        return x, y

    print('\n=== C1-v2 PAIRED per-sample sanity gate (this is the actual pass/fail test) ===')
    for name in ('zero_history', 'no_history', 'shuffled', 'misaligned', 'reversed_order'):
        x, y = paired_arrays(name)
        mean_diff, lo, hi = paired_bootstrap_ci(x, y, n_boot=args.n_boot)
        verdict = 'PASS (correct significantly better)' if lo > 0 else (
            'FAIL (correct significantly worse)' if hi < 0 else 'NO SIGNIFICANT DIFFERENCE (CI includes 0)')
        print(f'correct vs {name:<16} n={len(x):<5} mean_diff={mean_diff:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  {verdict}')

    print('\n=== C1-v2 gate activation diagnostics ===')
    for name in ('correct', 'shuffled'):
        stats = np.array(gate_stats[name])
        if len(stats) == 0:
            continue
        g1_mean, g1_std, g2_mean, g2_std = stats.mean(axis=0)
        print(f'{name}: g1 mean={g1_mean:.4f} std={g1_std:.4f}  g2 mean={g2_mean:.4f} std={g2_std:.4f}  n={len(stats)}')

    print('\n=== Case classification (per approved plan §12) ===')
    v2_overall = correct_summary
    c0_overall = summarize(c0_records)
    gate_signal = all(
        paired_bootstrap_ci(*paired_arrays(n), n_boot=args.n_boot)[1] > 0
        for n in ('shuffled', 'misaligned')
    )
    print(f'C1-v2 overall IoU vs C0: {v2_overall["iou"]:.4f} vs {c0_overall["iou"]:.4f}')
    print(f'correct > shuffled AND correct > misaligned (both CIs > 0): {gate_signal}')
    if gate_signal and v2_overall['iou'] > c0_overall['iou']:
        print('=> Case A: temporal history has a real, statistically supported contribution.')
    elif gate_signal:
        print('=> Case B (if a subset shows clear improvement): temporal benefit limited/subset-specific -- '
              'check the subset table above (sharp-turn/high_yaw in particular) by hand.')
    else:
        print('=> Case C: correct-history does NOT show a statistically robust advantage over shuffled/misaligned '
              'history. Per the approved plan, this means Stage 2C should formally end here -- '
              'no further temporal architecture escalation.')


if __name__ == '__main__':
    main()
