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
from projects.mmdet3d_plugin_carla_v2v.datasets.occ_beam_dataset import OccBeamDataset
from projects.mmdet3d_plugin.co3sop_base.dense_heads.occ_beam_encoder import OccupancyBeamEncoder

CHANNELS_TO_CIN = {'scene': 3, 'target': 1, 'scene_target': 4}


def evaluate(model, loader, device):
    model.eval()
    correct_tx = correct_rx = n = 0
    with torch.no_grad():
        for vol, tx, rx, _ in loader:
            vol, tx, rx = vol.to(device), tx.to(device), rx.to(device)
            tx_logits, rx_logits = model(vol)
            correct_tx += (tx_logits.argmax(1) == tx).sum().item()
            correct_rx += (rx_logits.argmax(1) == rx).sum().item()
            n += vol.size(0)
    model.train()
    return correct_tx / n, correct_rx / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['gt', 'pred'], required=True)
    parser.add_argument('--channels', choices=list(CHANNELS_TO_CIN), required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--work-dir', type=str, default=None)
    args = parser.parse_args()

    work_dir = args.work_dir or f'work_dirs/occ_beam_{args.source}_{args.channels}'
    os.makedirs(work_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = OccBeamDataset(split='train', source=args.source, channels=args.channels)
    val_ds = OccBeamDataset(split='val', source=args.source, channels=args.channels)
    print(f'[{args.source}/{args.channels}] train n={len(train_ds)} val n={len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = OccupancyBeamEncoder(in_channels=CHANNELS_TO_CIN[args.channels]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
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

        tx_acc, rx_acc = evaluate(model, val_loader, device)
        avg_loss = total_loss / n_batches
        history.append({'epoch': epoch, 'loss': avg_loss, 'val_tx_top1': tx_acc, 'val_rx_top1': rx_acc})
        print(f'epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}  val_tx_top1={tx_acc:.4f}  val_rx_top1={rx_acc:.4f}')

        combined = tx_acc + rx_acc
        if combined > best_val:
            best_val = combined
            torch.save(model.state_dict(), os.path.join(work_dir, 'best.pth'))

    torch.save(model.state_dict(), os.path.join(work_dir, 'last.pth'))
    with open(os.path.join(work_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f'[done] {args.source}/{args.channels} -> {work_dir} (best combined val top1 = {best_val:.4f})')


if __name__ == '__main__':
    main()
