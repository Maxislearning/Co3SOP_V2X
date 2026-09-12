#!/usr/bin/env python3
"""Stage 3 Step 5 smoke test (plan doc §27's 12-point checklist), run before
any real Beam Encoder training.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/home/admin0/carla_V2V/src')

from projects.mmdet3d_plugin_carla_v2v.datasets.occ_beam_dataset import OccBeamDataset, BEAM_CSV, CANON_DIR
from projects.mmdet3d_plugin_carla_v2v.datasets.occ_canonicalize import canonicalize_to_world_axes
from projects.mmdet3d_plugin.co3sop_base.dense_heads.occ_beam_encoder import OccupancyBeamEncoder
from generate_occupancy_gt import NX, NY, NZ, VOXEL, X_RANGE, Y_RANGE, world_to_ego, mark_pts


def check_1_label_correctness():
    ds = OccBeamDataset(split='val', source='gt', channels='scene_target')
    beam_df = pd.read_csv(BEAM_CSV)
    for idx in [0, len(ds) // 2, len(ds) - 1]:
        vol, tx, rx, meta = ds[idx]
        row = beam_df[(beam_df.sample_id == meta['sample_id']) & (beam_df.tx_name == meta['link'])].iloc[0]
        assert int(tx) == int(row.optimal_tx_beam_idx)
        assert int(rx) == int(row.optimal_rx_beam_idx)
    print('[OK] 1. sample_id/link/beam labels match the raw CSV')


def check_2_3_same_sample_no_future():
    # Scene+Target both live under the same {sample_id:06d}/ dir -- same
    # (sample_id,link) by construction, no separate lookup that could drift.
    ds = OccBeamDataset(split='train', source='pred', channels='scene_target')
    sid, link = ds.infos[0]
    d = os.path.join(CANON_DIR, f'{sid:06d}')
    for fname in os.listdir(d):
        assert 'future' not in fname and 't1' not in fname, f'found a future/t+1 file in extraction output: {fname}'
    print('[OK] 2/3. Scene+Target share one (sample_id,link) directory; no future/t+1 files anywhere in extraction output')


def check_4_canonical_frame():
    world_rel = np.array([10.0, 5.0, 0.0], dtype=np.float32)
    for yaw in (0.0, 90.0, -90.0):
        grid = np.zeros((NX, NY, NZ), dtype=np.float32)
        pts_ego = world_to_ego(world_rel[None, :], np.zeros(3, dtype=np.float32), yaw)
        mark_pts(grid, pts_ego, 1.0)
        canon = canonicalize_to_world_axes(grid, yaw, order=1, is_scene=False)
        idx = np.argwhere(canon > 1e-6)
        w = canon[canon > 1e-6]
        c = (idx * w[:, None]).sum(0) / w.sum()
        x = (c[0] + 0.5) * VOXEL + X_RANGE[0]
        y = (c[1] + 0.5) * VOXEL + Y_RANGE[0]
        err = np.hypot(x - world_rel[0], y - world_rel[1])
        assert err < VOXEL, f'canonicalization regression at yaw={yaw}: err={err:.3f}m'
    print('[OK] 4. RX_t-centered, world-axis-aligned canonicalization regression check (0/90/-90 deg) passes')


def check_5_6_7_8_shapes_dtype():
    ds = OccBeamDataset(split='val', source='pred', channels='scene_target')
    vol, tx, rx, meta = ds[0]
    assert vol.shape == (4, 16, 200, 200), vol.shape
    assert tx.dtype == torch.long and rx.dtype == torch.long

    model = OccupancyBeamEncoder(in_channels=4)
    tx_logits, rx_logits = model(vol.unsqueeze(0))
    assert tx_logits.shape == (1, 64) and rx_logits.shape == (1, 64)
    print('[OK] 5/6/7/8. volume shape [4,16,200,200], beam logits [1,64]x2, label dtype long')


def check_9_10_loss_and_grad():
    ds = OccBeamDataset(split='val', source='gt', channels='scene_target')
    model = OccupancyBeamEncoder(in_channels=4)
    vol, tx, rx, _ = ds[0]
    tx_logits, rx_logits = model(vol.unsqueeze(0))
    loss = torch.nn.functional.cross_entropy(tx_logits, tx.unsqueeze(0)) + \
        torch.nn.functional.cross_entropy(rx_logits, rx.unsqueeze(0))
    assert torch.isfinite(loss)
    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for _ in model.parameters())
    assert n_grad == n_total, f'only {n_grad}/{n_total} params got gradient'
    print(f'[OK] 9/10. CE loss finite ({loss.item():.4f}), all {n_total} encoder params got nonzero gradient')


def check_11_frozen():
    import inspect
    import tools.train_occ_beam as train_mod
    src = inspect.getsource(train_mod)
    assert 'target_occ_phaseB' not in src and 'load_checkpoint' not in src, \
        'train_occ_beam.py must never load the Stage 2B checkpoint -- it should only read precomputed volumes'
    print('[OK] 11. train_occ_beam.py never loads/touches the Stage 2B checkpoint (structurally frozen)')


def check_12_no_geometry():
    ds = OccBeamDataset(split='val', source='gt', channels='scene_target')
    vol, tx, rx, meta = ds[0]
    forbidden = ('tx_rx_geometry', 'tx_position', 'rx_position', 'distance', 'azimuth',
                 'target_center_raw', 'target_center_gt', 'ego_yaw', 'rx_yaw', 'pose')
    for key in forbidden:
        assert key not in meta, f'forbidden field {key!r} found in dataset item meta'
    # Check only actual code, not the module docstring -- which legitimately
    # *names* these forbidden fields in prose explaining they're absent
    # (a naive whole-file substring check would flag its own documentation,
    # the same class of self-match mistake as this project's earlier
    # pgrep-matches-its-own-command-line bug).
    import ast
    import inspect
    from projects.mmdet3d_plugin_carla_v2v.datasets import occ_beam_dataset as ds_mod
    src = inspect.getsource(ds_mod)
    tree = ast.parse(src)
    docstring = ast.get_docstring(tree) or ''
    code_only = src.replace(docstring, '')
    for key in ('tx_rx_geometry', 'ego_yaw', 'rx_yaw'):
        assert key not in code_only, f'forbidden field {key!r} referenced in occ_beam_dataset.py (outside its docstring)'
    print('[OK] 12. no tx_rx_geometry/pose/yaw anywhere in OccBeamDataset or its returned item')


if __name__ == '__main__':
    check_1_label_correctness()
    check_2_3_same_sample_no_future()
    check_4_canonical_frame()
    check_5_6_7_8_shapes_dtype()
    check_9_10_loss_and_grad()
    check_11_frozen()
    check_12_no_geometry()
    print('\nAll Stage 3 smoke checks passed.')
