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
from projects.mmdet3d_plugin.co3sop_base.dense_heads.occ_beam_encoder import OccupancyBeamEncoder
import panel_beamforming as pb

CHANNELS_TO_CIN = {'scene': 3, 'target': 1, 'scene_target': 4}
K = pb.BEAMS_PER_PANEL  # 16
NUM_CLASSES = pb.NUM_PANELS * K  # 64


def collect_logits(model, ds):
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
    tx_logits_all, rx_logits_all, tx_gt_all, rx_gt_all = [], [], [], []
    with torch.no_grad():
        for vol, tx, rx, _ in loader:
            tx_logits, rx_logits = model(vol)
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
    return {'panel_acc': panel_acc, 'top1': topk_acc(logits, gt, 1), 'inpanel_top1': ip1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['gt', 'pred'], required=True)
    parser.add_argument('--channels', choices=list(CHANNELS_TO_CIN), required=True)
    parser.add_argument('--ckpt', default='best.pth')
    parser.add_argument('--work-dir', default=None)
    args = parser.parse_args()

    work_dir = args.work_dir or f'work_dirs/occ_beam_{args.source}_{args.channels}'
    train_ds = OccBeamDataset(split='train', source=args.source, channels=args.channels)
    val_ds = OccBeamDataset(split='val', source=args.source, channels=args.channels)

    model = OccupancyBeamEncoder(in_channels=CHANNELS_TO_CIN[args.channels])
    state = torch.load(os.path.join(work_dir, args.ckpt), map_location='cpu')
    model.load_state_dict(state)
    model.eval()

    tx_logits_tr, rx_logits_tr, tx_gt_tr, rx_gt_tr = collect_logits(model, train_ds)
    tx_logits_va, rx_logits_va, tx_gt_va, rx_gt_va = collect_logits(model, val_ds)

    print(f'\n========== {args.source}/{args.channels} [{args.ckpt}] -- TX ==========')
    report_side('train', tx_logits_tr, tx_gt_tr, tx_gt_tr)
    report_side('val', tx_logits_va, tx_gt_va, tx_gt_tr)

    print(f'\n========== {args.source}/{args.channels} [{args.ckpt}] -- RX ==========')
    report_side('train', rx_logits_tr, rx_gt_tr, rx_gt_tr)
    report_side('val', rx_logits_va, rx_gt_va, rx_gt_tr)


if __name__ == '__main__':
    main()
