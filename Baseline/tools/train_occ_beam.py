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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/home/admin0/carla_V2V/src')
from projects.mmdet3d_plugin_carla_v2v.datasets.occ_beam_dataset import OccBeamDataset
from projects.mmdet3d_plugin.co3sop_base.dense_heads.occ_beam_encoder import OccupancyBeamEncoder
import panel_beamforming as pb

CHANNELS_TO_CIN = {'scene': 3, 'target': 1, 'scene_target': 4}


def evaluate(model, loader, device):
    """Combined (64-way) top1 AND panel-level (4-way, idx//BEAMS_PER_PANEL)
    top1 -- panel accuracy is the cheap sanity signal for "does the model
    know front/rear/left/right at all", tracked every epoch so a training
    run's sanity can be judged without a separate post-hoc pass."""
    model.eval()
    correct_tx = correct_rx = correct_tx_panel = correct_rx_panel = n = 0
    with torch.no_grad():
        for vol, tx, rx, _ in loader:
            vol, tx, rx = vol.to(device), tx.to(device), rx.to(device)
            tx_logits, rx_logits = model(vol)
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
    args = parser.parse_args()

    work_dir = args.work_dir or f'work_dirs/occ_beam_{args.source}_{args.channels}'
    os.makedirs(work_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = OccBeamDataset(split='train', source=args.source, channels=args.channels)
    val_ds = OccBeamDataset(split='val', source=args.source, channels=args.channels)
    print(f'[{args.source}/{args.channels}] train n={len(train_ds)} val n={len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = OccupancyBeamEncoder(in_channels=CHANNELS_TO_CIN[args.channels]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    epochs_since_improvement = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for vol, tx, rx, _ in train_loader:
            vol, tx, rx = vol.to(device), tx.to(device), rx.to(device)
            tx_logits, rx_logits = model(vol)
            loss = criterion(tx_logits, tx) + criterion(rx_logits, rx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        tx_acc, rx_acc, tx_panel_acc, rx_panel_acc = evaluate(model, val_loader, device)
        tr_tx_acc, tr_rx_acc, tr_tx_panel_acc, tr_rx_panel_acc = evaluate(model, train_eval_loader, device)
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
