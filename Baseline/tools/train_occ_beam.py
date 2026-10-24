#!/usr/bin/env python3
"""Stage 3 Step 4: plain PyTorch training loop for OccupancyBeamEncoder --
no mmcv Runner, no camera images, no gradient into the frozen Stage 2B
perception network (this script never even loads that checkpoint; it only
reads Step 1's precomputed volumes off disk).

Usage:
  python tools/train_occ_beam.py --source gt --channels scene_target --epochs 30
  python tools/train_occ_beam.py --source pred --channels target
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/home/admin0/carla_V2V/src')
from projects.mmdet3d_plugin_carla_v2v.datasets.occ_beam_dataset import OccBeamDataset
from projects.mmdet3d_plugin.co3sop_base.dense_heads.occ_beam_encoder import (
    OccupancyBeamEncoder, HierarchicalOccupancyBeamEncoder)
import panel_beamforming as pb

CHANNELS_TO_CIN = {'scene': 3, 'target': 1, 'scene_target': 4}
NUM_CLASSES = pb.NUM_PANELS * pb.BEAMS_PER_PANEL  # 64


def panel_sqrt_inv_freq_weights(train_ds):
    """任务22 A2: panel-level (4-way) analogue of sqrt_inv_freq_weights,
    used for the hierarchical model's panel head only -- the in-panel head
    stays unweighted since the diagnosis found no in-panel imbalance
    problem, only a panel-level one."""
    tx_counts = np.zeros(pb.NUM_PANELS, dtype=np.float64)
    rx_counts = np.zeros(pb.NUM_PANELS, dtype=np.float64)
    for i in range(len(train_ds)):
        _, tx, rx, _ = train_ds[i]
        tx_counts[int(tx) // pb.BEAMS_PER_PANEL] += 1
        rx_counts[int(rx) // pb.BEAMS_PER_PANEL] += 1

    def to_weight(counts):
        w = 1.0 / np.sqrt(np.maximum(counts, 1.0))
        w = w / w.mean()
        return torch.tensor(w, dtype=torch.float32)

    tx_w, rx_w = to_weight(tx_counts), to_weight(rx_counts)
    print(f'[panel loss weighting] tx panel counts={tx_counts.astype(int).tolist()} '
          f'({[pb.PANEL_NAMES[p] for p in range(pb.NUM_PANELS)]}) weight={np.round(tx_w.numpy(),2).tolist()}')
    print(f'[panel loss weighting] rx panel counts={rx_counts.astype(int).tolist()} '
          f'weight={np.round(rx_w.numpy(),2).tolist()}')
    return tx_w, rx_w


def sqrt_inv_freq_weights(train_ds):
    """任务22 A1: minimal-intervention fix for the diagnosed panel-level
    class-imbalance collapse (flat-64 head unchanged, only the loss is
    reweighted). weight[c] = 1/sqrt(count[c]), normalized so the mean over
    OBSERVED classes is 1 (keeps overall loss scale comparable to
    unweighted CE). Classes with 0 train samples never appear as a target,
    so their weight value is irrelevant to training -- given count=1 only
    to avoid a division by zero."""
    tx_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    rx_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for i in range(len(train_ds)):
        _, tx, rx, _ = train_ds[i]
        tx_counts[int(tx)] += 1
        rx_counts[int(rx)] += 1

    def to_weight(counts):
        safe = np.maximum(counts, 1.0)
        w = 1.0 / np.sqrt(safe)
        observed = counts > 0
        w = w / w[observed].mean()
        return torch.tensor(w, dtype=torch.float32)

    tx_w, rx_w = to_weight(tx_counts), to_weight(rx_counts)
    print(f'[loss weighting] tx: {int((tx_counts>0).sum())}/{NUM_CLASSES} classes observed in train, '
          f'weight range [{tx_w.min():.2f}, {tx_w.max():.2f}]')
    print(f'[loss weighting] rx: {int((rx_counts>0).sum())}/{NUM_CLASSES} classes observed in train, '
          f'weight range [{rx_w.min():.2f}, {rx_w.max():.2f}]')
    return tx_w, rx_w


def forward_combined(model, vol, architecture):
    """Returns (tx_logits[B,64], rx_logits[B,64]) regardless of architecture
    -- for 'hierarchical', combines panel+in-panel logits via log-prob
    addition (see HierarchicalOccupancyBeamEncoder's docstring) so all
    downstream eval code stays architecture-agnostic."""
    if architecture == 'flat':
        return model(vol)
    tx_panel, tx_inpanel, rx_panel, rx_inpanel = model(vol)
    tx_logits = HierarchicalOccupancyBeamEncoder.combine_to_flat_logits(
        tx_panel, tx_inpanel, pb.NUM_PANELS, pb.BEAMS_PER_PANEL)
    rx_logits = HierarchicalOccupancyBeamEncoder.combine_to_flat_logits(
        rx_panel, rx_inpanel, pb.NUM_PANELS, pb.BEAMS_PER_PANEL)
    return tx_logits, rx_logits


def evaluate(model, loader, device, architecture):
    """Combined (64-way) top1 AND panel-level (4-way, idx//BEAMS_PER_PANEL)
    top1 -- panel accuracy is the cheap sanity signal for "does the model
    know front/rear/left/right at all", tracked every epoch so a training
    run's sanity can be judged without a separate post-hoc pass."""
    model.eval()
    correct_tx = correct_rx = correct_tx_panel = correct_rx_panel = n = 0
    with torch.no_grad():
        for vol, tx, rx, _ in loader:
            vol, tx, rx = vol.to(device), tx.to(device), rx.to(device)
            tx_logits, rx_logits = forward_combined(model, vol, architecture)
            tx_pred, rx_pred = tx_logits.argmax(1), rx_logits.argmax(1)
            correct_tx += (tx_pred == tx).sum().item()
            correct_rx += (rx_pred == rx).sum().item()
            correct_tx_panel += (tx_pred // pb.BEAMS_PER_PANEL == tx // pb.BEAMS_PER_PANEL).sum().item()
            correct_rx_panel += (rx_pred // pb.BEAMS_PER_PANEL == rx // pb.BEAMS_PER_PANEL).sum().item()
            n += vol.size(0)
    model.train()
    return correct_tx / n, correct_rx / n, correct_tx_panel / n, correct_rx_panel / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['gt', 'pred'], required=True)
    parser.add_argument('--channels', choices=list(CHANNELS_TO_CIN), required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--work-dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=10,
                        help='early stopping: stop if val combined top1 (tx_acc+rx_acc) does not improve '
                             'for this many consecutive epochs (任务22 -- the GT/scene_target sanity run '
                             'showed val performance peaking around epoch 13-17 then degrading, with the '
                             'RX head fully collapsing to a constant prediction by epoch 30; best.pth was '
                             'already tracked, this just stops wasting epochs training past the peak)')
    parser.add_argument('--loss-weighting', choices=['none', 'sqrt_inv_freq'], default='none',
                        help='任务22 A1: sqrt_inv_freq reweights CrossEntropyLoss by '
                             '1/sqrt(train class count) (flat: full 64-way count; hierarchical: '
                             'panel-head only, by 4-way panel count -- in-panel head stays unweighted).')
    parser.add_argument('--architecture', choices=['flat', 'hierarchical'], default='flat',
                        help='任务22 A2: hierarchical = separate 4-way panel + 16-way (shared) '
                             'in-panel heads per side, so panel loss can be reweighted independently. '
                             'flat (default) = OccupancyBeamEncoder, single 64-way head per side (A0/A1).')
    parser.add_argument('--seed', type=int, default=None,
                        help='任务34: seeds torch/numpy/random and the train-loader shuffle for multi-seed '
                             'repeats. Default None = unseeded, identical to the original v4 sweep.')
    parser.add_argument('--coord-conv', nargs='?', const='dense', default=False, choices=['dense', 'gated'],
                        help='任务35: append normalized (z,y,x) coordinate channels to the input (CoordConv) -- '
                             'fixes gt/target underfitting from the translation-invariant conv+GAP trunk. '
                             'dense (bare flag) = everywhere; gated = multiplied by occupancy (no init plateau). '
                             'Default off = identical to the v4 sweep.')
    parser.add_argument('--split-file', type=str, default=None,
                        help='任务35: JSON with "train"/"val" lists of [sample_id, link] (tools/loro_cv.py); '
                             'val is then the inner early-stopping split. Default = Stage 2B config split.')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='train loader workers (eval loaders use min(2, this)); lower under memory pressure')
    args = parser.parse_args()

    loader_gen = None
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        loader_gen = torch.Generator().manual_seed(args.seed)

    work_dir = args.work_dir or f'work_dirs/occ_beam_{args.source}_{args.channels}'
    os.makedirs(work_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.split_file:
        with open(args.split_file) as f:
            split = json.load(f)
        train_ds = OccBeamDataset('train', args.source, args.channels, infos=split['train'])
        val_ds = OccBeamDataset('val', args.source, args.channels, infos=split['val'])
    else:
        train_ds = OccBeamDataset(split='train', source=args.source, channels=args.channels)
        val_ds = OccBeamDataset(split='val', source=args.source, channels=args.channels)
    print(f'[{args.source}/{args.channels}] train n={len(train_ds)} val n={len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False,
                              generator=loader_gen)
    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=min(2, args.num_workers))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=min(2, args.num_workers))

    if args.architecture == 'flat':
        model = OccupancyBeamEncoder(in_channels=CHANNELS_TO_CIN[args.channels],
                                     coord_channels=args.coord_conv).to(device)
    else:
        model = HierarchicalOccupancyBeamEncoder(
            in_channels=CHANNELS_TO_CIN[args.channels],
            num_panels=pb.NUM_PANELS, beams_per_panel=pb.BEAMS_PER_PANEL,
            coord_channels=args.coord_conv).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.architecture == 'flat':
        if args.loss_weighting == 'sqrt_inv_freq':
            tx_weight, rx_weight = sqrt_inv_freq_weights(train_ds)
            tx_criterion = nn.CrossEntropyLoss(weight=tx_weight.to(device))
            rx_criterion = nn.CrossEntropyLoss(weight=rx_weight.to(device))
        else:
            tx_criterion = rx_criterion = nn.CrossEntropyLoss()
    else:
        if args.loss_weighting == 'sqrt_inv_freq':
            tx_panel_w, rx_panel_w = panel_sqrt_inv_freq_weights(train_ds)
            tx_panel_criterion = nn.CrossEntropyLoss(weight=tx_panel_w.to(device))
            rx_panel_criterion = nn.CrossEntropyLoss(weight=rx_panel_w.to(device))
        else:
            tx_panel_criterion = rx_panel_criterion = nn.CrossEntropyLoss()
        tx_inpanel_criterion = rx_inpanel_criterion = nn.CrossEntropyLoss()  # always unweighted, per diagnosis

    best_val = -1.0
    epochs_since_improvement = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for vol, tx, rx, _ in train_loader:
            vol, tx, rx = vol.to(device), tx.to(device), rx.to(device)
            if args.architecture == 'flat':
                tx_logits, rx_logits = model(vol)
                loss = tx_criterion(tx_logits, tx) + rx_criterion(rx_logits, rx)
            else:
                tx_panel_logits, tx_inpanel_logits, rx_panel_logits, rx_inpanel_logits = model(vol)
                tx_panel_gt, tx_local_gt = tx // pb.BEAMS_PER_PANEL, tx % pb.BEAMS_PER_PANEL
                rx_panel_gt, rx_local_gt = rx // pb.BEAMS_PER_PANEL, rx % pb.BEAMS_PER_PANEL
                loss = (tx_panel_criterion(tx_panel_logits, tx_panel_gt) +
                        tx_inpanel_criterion(tx_inpanel_logits, tx_local_gt) +
                        rx_panel_criterion(rx_panel_logits, rx_panel_gt) +
                        rx_inpanel_criterion(rx_inpanel_logits, rx_local_gt))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        tx_acc, rx_acc, tx_panel_acc, rx_panel_acc = evaluate(model, val_loader, device, args.architecture)
        tr_tx_acc, tr_rx_acc, tr_tx_panel_acc, tr_rx_panel_acc = evaluate(model, train_eval_loader, device, args.architecture)
        avg_loss = total_loss / n_batches
        history.append({
            'epoch': epoch, 'loss': avg_loss,
            'train_tx_top1': tr_tx_acc, 'train_rx_top1': tr_rx_acc,
            'train_tx_panel_acc': tr_tx_panel_acc, 'train_rx_panel_acc': tr_rx_panel_acc,
            'val_tx_top1': tx_acc, 'val_rx_top1': rx_acc,
            'val_tx_panel_acc': tx_panel_acc, 'val_rx_panel_acc': rx_panel_acc,
        })
        print(f'epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}  '
              f'train_tx/rx_top1={tr_tx_acc:.3f}/{tr_rx_acc:.3f} panel={tr_tx_panel_acc:.3f}/{tr_rx_panel_acc:.3f}  '
              f'val_tx/rx_top1={tx_acc:.3f}/{rx_acc:.3f} panel={tx_panel_acc:.3f}/{rx_panel_acc:.3f}')

        combined = tx_acc + rx_acc
        if combined > best_val:
            best_val = combined
            epochs_since_improvement = 0
            torch.save(model.state_dict(), os.path.join(work_dir, 'best.pth'))
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= args.patience:
                print(f'[early stop] val combined top1 has not improved for {args.patience} epochs '
                      f'(best={best_val:.4f} at an earlier epoch) -- stopping at epoch {epoch+1}/{args.epochs}')
                break

    torch.save(model.state_dict(), os.path.join(work_dir, 'last.pth'))
    with open(os.path.join(work_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f'[done] {args.source}/{args.channels} -> {work_dir} (best combined val top1 = {best_val:.4f})')


if __name__ == '__main__':
    main()
