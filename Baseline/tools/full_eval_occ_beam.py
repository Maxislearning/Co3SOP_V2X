#!/usr/bin/env python3
"""Stage 3A full clean-evaluation report for one (source, channels) combo's
trained checkpoint -- the diagnostic the user asked for before deciding on
weighted CE / dropout / a hierarchical 4x16 head.

Computes, for both train and val splits:
  - 64-way majority baseline (fit on train, eval on both splits)
  - 4-way majority-panel baseline (same)
  - Top1/3/5 (combined 64-way)
  - panel accuracy (pred_idx//16 == gt_idx//16)
  - per-panel recall (for each true panel, what fraction get the panel right)
  - GT-panel-conditioned in-panel Top1/3/5 (slice logits to the 16 belonging
    to the TRUE panel, rank within that -- isolates fine-steering ability
    from panel selection; conditioning on GT panel, never predicted panel,
    per the user's explicit requirement)
  - prediction diversity (n unique classes predicted)
  - GT class/panel distribution (train vs val)

Usage: python tools/full_eval_occ_beam.py --source gt --channels scene_target --ckpt best.pth
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/home/admin0/carla_V2V/src')
from projects.mmdet3d_plugin_carla_v2v.datasets.occ_beam_dataset import OccBeamDataset
from projects.mmdet3d_plugin.co3sop_base.dense_heads.occ_beam_encoder import (
    OccupancyBeamEncoder, HierarchicalOccupancyBeamEncoder)
import panel_beamforming as pb

CHANNELS_TO_CIN = {'scene': 3, 'target': 1, 'scene_target': 4}
K = pb.BEAMS_PER_PANEL  # 16
NUM_CLASSES = pb.NUM_PANELS * K  # 64


def collect_logits(model, ds, architecture):
    """Returns combined 64-way logits regardless of architecture -- for
    'hierarchical', combines panel+in-panel logits via log-prob addition
    (HierarchicalOccupancyBeamEncoder.combine_to_flat_logits) so every
    metric below stays architecture-agnostic."""
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
    tx_logits_all, rx_logits_all, tx_gt_all, rx_gt_all = [], [], [], []
    with torch.no_grad():
        for vol, tx, rx, _ in loader:
            if architecture == 'flat':
                tx_logits, rx_logits = model(vol)
            else:
                tx_panel, tx_inpanel, rx_panel, rx_inpanel = model(vol)
                tx_logits = HierarchicalOccupancyBeamEncoder.combine_to_flat_logits(
                    tx_panel, tx_inpanel, pb.NUM_PANELS, pb.BEAMS_PER_PANEL)
                rx_logits = HierarchicalOccupancyBeamEncoder.combine_to_flat_logits(
                    rx_panel, rx_inpanel, pb.NUM_PANELS, pb.BEAMS_PER_PANEL)
            tx_logits_all.append(tx_logits.numpy())
            rx_logits_all.append(rx_logits.numpy())
            tx_gt_all.append(tx.numpy())
            rx_gt_all.append(rx.numpy())
    return (np.concatenate(tx_logits_all), np.concatenate(rx_logits_all),
            np.concatenate(tx_gt_all), np.concatenate(rx_gt_all))


def topk_acc(logits, gt, k):
    topk = np.argsort(-logits, axis=1)[:, :k]
    return (topk == gt[:, None]).any(1).mean()


def panel_conditioned_inpanel_topk(logits, gt, k):
    """Slice each sample's logits to its OWN GT panel's 16 entries, rank
    within that 16-way sub-problem. Conditioned on GT panel, not predicted
    panel -- isolates fine-steering difficulty from panel selection."""
    gt_panel = gt // K
    gt_local = gt % K
    n = logits.shape[0]
    hits = np.zeros(n, dtype=bool)
    for p in range(pb.NUM_PANELS):
        mask = gt_panel == p
        if not mask.any():
            continue
        sub_logits = logits[mask, p * K:(p + 1) * K]  # [n_p, 16]
        sub_gt = gt_local[mask]
        topk = np.argsort(-sub_logits, axis=1)[:, :k]
        hits[mask] = (topk == sub_gt[:, None]).any(1)
    return hits.mean()


def per_panel_recall(pred, gt):
    gt_panel = gt // K
    pred_panel = pred // K
    out = {}
    for p in range(pb.NUM_PANELS):
        mask = gt_panel == p
        n = int(mask.sum())
        recall = float((pred_panel[mask] == p).mean()) if n > 0 else float('nan')
        out[pb.PANEL_NAMES[p]] = (recall, n)
    return out


def majority_baselines(train_gt):
    """Returns (majority_class, majority_panel) fit on train_gt."""
    vals, counts = np.unique(train_gt, return_counts=True)
    majority_class = int(vals[np.argmax(counts)])
    panels = train_gt // K
    pvals, pcounts = np.unique(panels, return_counts=True)
    majority_panel = int(pvals[np.argmax(pcounts)])
    return majority_class, majority_panel


def report_side(name, logits, gt, train_gt_for_baseline):
    pred = logits.argmax(1)
    maj_class, maj_panel = majority_baselines(train_gt_for_baseline)

    print(f'--- {name} (n={len(gt)}) ---')
    print(f'  64-way majority baseline (train majority={maj_class}): '
          f'acc={float((gt == maj_class).mean()):.3f}')
    print(f'  4-way panel majority baseline (train majority panel={pb.PANEL_NAMES[maj_panel]}): '
          f'acc={float((gt // K == maj_panel).mean()):.3f}')
    print(f'  Top1/3/5 (64-way): {topk_acc(logits, gt, 1):.3f} / {topk_acc(logits, gt, 3):.3f} / {topk_acc(logits, gt, 5):.3f}')
    topk_list = [float(topk_acc(logits, gt, k)) for k in range(1, 11)]
    print(f'  Top1-10 (64-way): ' + ' / '.join(f'{a:.3f}' for a in topk_list))
    panel_acc = float((pred // K == gt // K).mean())
    print(f'  Panel accuracy: {panel_acc:.3f}')
    print(f'  Per-panel recall (GT-panel conditioned):')
    for pname, (recall, n) in per_panel_recall(pred, gt).items():
        print(f'    {pname:6}: recall={recall:.3f}  (n={n})')
    ip1 = panel_conditioned_inpanel_topk(logits, gt, 1)
    ip3 = panel_conditioned_inpanel_topk(logits, gt, 3)
    ip5 = panel_conditioned_inpanel_topk(logits, gt, 5)
    print(f'  In-panel Top1/3/5 (conditioned on GT panel, 16-way sub-problem): {ip1:.3f} / {ip3:.3f} / {ip5:.3f}')
    n_unique = len(set(pred.tolist()))
    print(f'  Prediction diversity: {n_unique}/{NUM_CLASSES} classes used')
    import collections
    print(f'  pred class distribution (top5): {collections.Counter(pred.tolist()).most_common(5)}')
    print(f'  GT class distribution (top5):   {collections.Counter(gt.tolist()).most_common(5)}')
    gt_panel_dist = collections.Counter((gt // K).tolist())
    print(f'  GT panel distribution: {[(pb.PANEL_NAMES[p], c) for p, c in sorted(gt_panel_dist.items())]}')
    return {'panel_acc': panel_acc, 'top1': topk_acc(logits, gt, 1), 'inpanel_top1': ip1, 'topk_1_10': topk_list}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['gt', 'pred'], required=True)
    parser.add_argument('--channels', choices=list(CHANNELS_TO_CIN), required=True)
    parser.add_argument('--ckpt', default='best.pth')
    parser.add_argument('--work-dir', default=None)
    parser.add_argument('--architecture', choices=['flat', 'hierarchical'], default='flat')
    parser.add_argument('--coord-conv', nargs='?', const='dense', default=False, choices=['dense', 'gated'], help='任务35: model was trained with --coord-conv')
    args = parser.parse_args()

    work_dir = args.work_dir or f'work_dirs/occ_beam_{args.source}_{args.channels}'
    train_ds = OccBeamDataset(split='train', source=args.source, channels=args.channels)
    val_ds = OccBeamDataset(split='val', source=args.source, channels=args.channels)

    if args.architecture == 'flat':
        model = OccupancyBeamEncoder(in_channels=CHANNELS_TO_CIN[args.channels], coord_channels=args.coord_conv)
    else:
        model = HierarchicalOccupancyBeamEncoder(
            in_channels=CHANNELS_TO_CIN[args.channels],
            num_panels=pb.NUM_PANELS, beams_per_panel=pb.BEAMS_PER_PANEL, coord_channels=args.coord_conv)
    state = torch.load(os.path.join(work_dir, args.ckpt), map_location='cpu')
    model.load_state_dict(state)
    model.eval()

    tx_logits_tr, rx_logits_tr, tx_gt_tr, rx_gt_tr = collect_logits(model, train_ds, args.architecture)
    tx_logits_va, rx_logits_va, tx_gt_va, rx_gt_va = collect_logits(model, val_ds, args.architecture)

    print(f'\n========== {args.source}/{args.channels} [{args.ckpt}, {args.architecture}] -- TX ==========')
    tx_tr = report_side('train', tx_logits_tr, tx_gt_tr, tx_gt_tr)
    tx_va = report_side('val', tx_logits_va, tx_gt_va, tx_gt_tr)

    print(f'\n========== {args.source}/{args.channels} [{args.ckpt}, {args.architecture}] -- RX ==========')
    rx_tr = report_side('train', rx_logits_tr, rx_gt_tr, rx_gt_tr)
    rx_va = report_side('val', rx_logits_va, rx_gt_va, rx_gt_tr)

    # 联合beam对Top-k：把TX/RX各自的log-softmax相加，对全部64x64=4096个(tx,rx)
    # beam对排序，GT的(tx,rx)对落在前k个里算命中——对应"实际要扫描k个beam对"
    # 的场景，比TX/RX各自单独的Top-k更贴近真实的beam训练开销。
    def joint_topk(tx_logits, rx_logits, tx_gt, rx_gt, ks=range(1, 11)):
        def lsm(x):
            x = x - x.max(1, keepdims=True)
            return x - np.log(np.exp(x).sum(1, keepdims=True))
        joint = lsm(tx_logits)[:, :, None] + lsm(rx_logits)[:, None, :]  # [N,64tx,64rx]
        flat = joint.reshape(len(joint), -1)
        gt_idx = tx_gt * rx_logits.shape[1] + rx_gt
        order = np.argsort(-flat, axis=1)[:, :max(ks)]
        return [float((order[:, :k] == gt_idx[:, None]).any(1).mean()) for k in ks]

    print(f'\n========== {args.source}/{args.channels} -- JOINT (TX,RX) beam-pair Top-k over 4096 pairs ==========')
    jt_tr = joint_topk(tx_logits_tr, rx_logits_tr, tx_gt_tr, rx_gt_tr)
    jt_va = joint_topk(tx_logits_va, rx_logits_va, tx_gt_va, rx_gt_va)
    print('  train Top1-10: ' + ' / '.join(f'{a:.3f}' for a in jt_tr))
    print('  val   Top1-10: ' + ' / '.join(f'{a:.3f}' for a in jt_va))

    import json
    with open(os.path.join(work_dir, f'topk_{os.path.splitext(args.ckpt)[0]}.json'), 'w') as f:
        json.dump({'tx': {'train': tx_tr['topk_1_10'], 'val': tx_va['topk_1_10']},
                   'rx': {'train': rx_tr['topk_1_10'], 'val': rx_va['topk_1_10']},
                   'joint': {'train': jt_tr, 'val': jt_va},
                   'panel_acc': {'tx_val': tx_va['panel_acc'], 'rx_val': rx_va['panel_acc']}}, f, indent=1)


if __name__ == '__main__':
    main()
