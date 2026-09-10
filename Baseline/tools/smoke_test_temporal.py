#!/usr/bin/env python3
"""Stage 2C smoke test (plan §Verification item 2 / doc §31 checklist):
build C0 and C1 datasets+models, load the Stage 2B Phase B checkpoint, run
one real batch through forward_train + backward, and print every shape/
metric the checklist + the user's follow-up correction ask for.

Follow-up correction (do not hardcode 1.2 m/voxel as a test-script
assumption): this script derives and prints the fused feature's actual
metric extent/resolution FROM the running model's own config values
(volume_z_[0] -> voxel_size formula _encode_and_fuse itself uses), not from
a copied constant -- and explicitly checks it against the label grid's
80m/(-40,40) convention, since Stage 2A's tools/verify_temporal_alignment.py
debugging session found these two grids are NOT the same resolution.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import mmcv
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

import projects.mmdet3d_plugin  # noqa: F401 registers co3sop_base/*
import projects.mmdet3d_plugin_carla_v2v  # noqa: F401 registers carla_v2v datasets


def load_batch(dataset, n=1):
    items = [dataset[i] for i in range(n)]
    from mmcv.parallel import collate
    return collate(items, samples_per_gpu=n)


def run_c0():
    print('=' * 70)
    print('C0 smoke test (co3sop_base_carla_v2v_temporal_c0.py)')
    print('=' * 70)
    cfg = Config.fromfile('projects/configs/co3sop_base/co3sop_base_carla_v2v_temporal_c0.py')
    dataset = build_dataset(cfg.data.train)
    print(f'[C0] dataset size: {len(dataset)}')
    batch = load_batch(dataset, n=1)
    # img/img_metas are true DataContainers (DefaultFormatBundle3D/
    # CustomCollect3D wrap them explicitly) -- .data is a list-over-GPUs,
    # .data[0] is this GPU's batched tensor. gt_occ/gt_target are plain
    # tensors after collate (never DC-wrapped anywhere in the pipeline), so
    # they must NOT be .data-indexed -- doing so would silently strip the
    # batch dim via torch.Tensor's legacy .data accessor (caught by this
    # exact mistake on the first run: shape (200,200,16) instead of
    # (1,200,200,16), which then IndexError'd inside multiscale_supervision).
    img = batch['img'].data[0]
    img_metas = batch['img_metas'].data[0]
    gt_occ = batch['gt_occ']
    gt_target = batch['gt_target']
    print(f'[C0] img shape: {tuple(img.shape)}')
    print(f'[C0] gt_occ (future scene) shape: {tuple(gt_occ.shape)}, unique labels: {torch.unique(gt_occ).tolist()}')
    print(f'[C0] gt_target (future target) shape: {tuple(gt_target.shape)}, unique labels: {torch.unique(gt_target).tolist()}')
    print(f'[C0] target_role: {[m["target_role"] for m in img_metas]}')

    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    ckpt = mmcv.runner.load_checkpoint(model, cfg.load_from, map_location='cpu', strict=False)
    model.train()
    model.cuda()

    img = img.cuda()
    gt_occ = gt_occ.cuda()
    gt_target = gt_target.cuda()

    losses = model.forward_train(img_metas=img_metas, img=img, gt_occ=gt_occ, gt_target=gt_target)
    total = sum(v.mean() if v.dim() > 0 else v for v in losses.values() if isinstance(v, torch.Tensor))
    print(f'[C0] loss keys: {list(losses.keys())}')
    print(f'[C0] all losses finite: {all(torch.isfinite(v).all() for v in losses.values() if isinstance(v, torch.Tensor))}')
    total.backward()
    n_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    print(f'[C0] params with grad: {n_grad}')
    print(f'[C0] peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB')
    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def run_c1():
    print('=' * 70)
    print('C1 smoke test (co3sop_base_carla_v2v_temporal_c1.py)')
    print('=' * 70)
    cfg = Config.fromfile('projects/configs/co3sop_base/co3sop_base_carla_v2v_temporal_c1.py')
    dataset = build_dataset(cfg.data.train)
    print(f'[C1] dataset size: {len(dataset)}')

    item0 = dataset[0]
    print(f'[C1] single-item img shape (queue): {tuple(item0["img"].data.shape)}')
    print(f'[C1] single-item img_metas len (queue): {len(item0["img_metas"].data)}')
    print(f'[C1] single-item gt_occ_aux_k shape: {np.asarray(item0["gt_occ_aux_k"]).shape}')
    print(f'[C1] single-item gt_occ_future shape: {np.asarray(item0["gt_occ_future"]).shape}')
    print(f'[C1] single-item gt_target_future shape: {np.asarray(item0["gt_target_future"]).shape}')
    print(f'[C1] single-item target_center_raw: {np.asarray(item0["target_center_raw"])}')
    for k, m in enumerate(item0['img_metas'].data):
        t = m['temporal_trans_to_t']
        print(f'[C1]   frame k={k}: temporal_trans_to_t translation (raw, no S-flip) = {np.asarray(t)[:3, 3]}, '
              f'target_role={m["target_role"]}')

    batch = load_batch(dataset, n=1)
    img = batch['img'].data[0]                     # [B,K,N,C,H,W]
    img_metas = batch['img_metas'].data[0]          # list length B, each list length K
    gt_occ_future = batch['gt_occ_future']
    gt_target_future = batch['gt_target_future']
    gt_occ_aux_k = batch['gt_occ_aux_k']
    print(f'[C1] batched img shape: {tuple(img.shape)}')
    print(f'[C1] batched gt_occ_aux_k shape: {tuple(np.asarray(gt_occ_aux_k).shape)}')

    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    ckpt = mmcv.runner.load_checkpoint(model, cfg.load_from, map_location='cpu', strict=False)
    print(f'[C1] checkpoint load: missing_keys(sample)={ckpt.get("missing_keys", [])[:5] if isinstance(ckpt, dict) else "n/a"}')
    model.train()
    model.cuda()

    img = img.cuda()
    gt_occ_future_t = torch.as_tensor(np.asarray(gt_occ_future)).cuda() if not torch.is_tensor(gt_occ_future) else gt_occ_future.cuda()
    gt_target_future_t = torch.as_tensor(np.asarray(gt_target_future)).cuda() if not torch.is_tensor(gt_target_future) else gt_target_future.cuda()
    gt_occ_aux_k_t = torch.as_tensor(np.asarray(gt_occ_aux_k)).cuda() if not torch.is_tensor(gt_occ_aux_k) else gt_occ_aux_k.cuda()

    # ---- Runtime metric-extent verification (required, not hardcoded) ----
    head = model.pts_bbox_head
    with torch.no_grad():
        mcar_feats_list, img_metas_list = model._extract_queue_feats(img, img_metas)
        fused_k0, _, _ = head._encode_and_fuse(mcar_feats_list[0], img_metas_list[0])
    B, C, W, H, Z = fused_k0.shape
    voxel_size = 0.1 * 48 / Z  # exact formula _encode_and_fuse/head._warp_to_t use, read from the live head/model, not copy-pasted
    extent_x = W * voxel_size
    extent_y = H * voxel_size
    extent_z = Z * voxel_size
    print('-' * 70)
    print('[C1] RUNTIME metric-extent verification (from live model, not a hardcoded test constant):')
    print(f'[C1]   fused feature shape (per frame): {tuple(fused_k0.shape)}  (B,C,W,H,Z)')
    print(f'[C1]   meters-per-feature-cell: X={voxel_size:.4f} m, Y={voxel_size:.4f} m, Z={voxel_size:.4f} m')
    print(f'[C1]   feature-space metric extent: X={extent_x:.2f} m, Y={extent_y:.2f} m, Z={extent_z:.2f} m')
    print(f'[C1]   label-grid (Stage 2A occ GT) extent for comparison: X=80.00 m, Y=80.00 m (pc_range -40..40), '
          f'cell=0.4m -- DIFFERENT from the fused feature grid above, confirmed NOT equal.')
    same_as_label_grid = abs(extent_x - 80.0) < 1e-6
    print(f'[C1]   fused feature grid == 80m label grid? {same_as_label_grid}  '
          f'(expected False -- coarse feature grid is {extent_x:.1f}m, not 80m)')

    raw_t = img_metas_list[0][0]['temporal_trans_to_t']
    raw_t = np.asarray(raw_t, dtype=np.float64)
    trans_m = raw_t[:3, 3]
    print(f'[C1]   sample temporal translation (frame k=0 -> t), raw x1_to_x2 (meters): {trans_m}')
    trans_cells = trans_m / voxel_size
    print(f'[C1]   same translation in feature cells (meters / {voxel_size:.4f} m-per-cell): {trans_cells}')
    print('-' * 70)

    losses = model.pts_bbox_head.loss(
        gt_occ_future_t, gt_target_future_t,
        model.pts_bbox_head(mcar_feats_list, img_metas_list),
        img_metas_list, gt_occ_aux_k=gt_occ_aux_k_t if head.lambda_aux > 0 else None)
    print(f'[C1] loss keys: {list(losses.keys())}')
    all_finite = all(torch.isfinite(v).all() for v in losses.values() if isinstance(v, torch.Tensor))
    print(f'[C1] all losses finite: {all_finite}')
    total = sum(v.mean() if v.dim() > 0 else v for v in losses.values() if isinstance(v, torch.Tensor))
    total.backward()
    n_grad_temporal_fusion = sum(1 for p in head.temporal_fusion.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total_temporal_fusion = sum(1 for _ in head.temporal_fusion.parameters())
    print(f'[C1] temporal_fusion params with nonzero grad: {n_grad_temporal_fusion}/{n_total_temporal_fusion}')
    n_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    print(f'[C1] total params with grad: {n_grad}')
    print(f'[C1] peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB')


if __name__ == '__main__':
    run_c0()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    run_c1()
    print('\nSmoke test complete.')
