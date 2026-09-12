#!/usr/bin/env python3
"""Stage 3 Audit 1b: synthetic correctness test for
occ_canonicalize.canonicalize_to_world_axes, required before any Beam
Encoder code trusts it (same discipline as Stage 2C's
verify_temporal_alignment.py).

1. A marker at a fixed world-relative offset, placed into ego-frame grids at
   5 different RX yaws via the REAL world_to_ego/mark_pts, must recover the
   same world-relative position after canonicalization, at every yaw, within
   one voxel (0.4m).
2. The boundary-padding fix: a Scene one-hot/softmax volume with cval=0
   padding at rotated-in edges must come back with EVERY voxel's 3 channels
   summing to ~1 (no illegal [0,0,0] anywhere), and specifically Empty=1 at
   padding voxels (not just "sums to 1 doing something else").
3. Target volumes must NOT be touched by the padding fix -- verified by
   calling without is_scene and confirming outside stays exactly 0.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/home/admin0/carla_V2V/src')

from generate_occupancy_gt import NX, NY, NZ, VOXEL, X_RANGE, Y_RANGE, world_to_ego, mark_pts  # noqa: E402
from projects.mmdet3d_plugin_carla_v2v.datasets.occ_canonicalize import canonicalize_to_world_axes  # noqa: E402

WORLD_REL_OFFSET = np.array([10.0, 5.0, 0.0], dtype=np.float32)


def build_ego_marker_grid(ego_yaw_deg):
    grid = np.zeros((NX, NY, NZ), dtype=np.float32)
    pts_ego = world_to_ego(WORLD_REL_OFFSET[None, :], np.zeros(3, dtype=np.float32), ego_yaw_deg)
    mark_pts(grid, pts_ego, 1.0)
    return grid


def centroid_metric(grid):
    idx = np.argwhere(grid > 1e-6)
    w = grid[grid > 1e-6]
    c = (idx * w[:, None]).sum(0) / w.sum()
    x = (c[0] + 0.5) * VOXEL + X_RANGE[0]
    y = (c[1] + 0.5) * VOXEL + Y_RANGE[0]
    return np.array([x, y])


def check_marker_recovery():
    ref = centroid_metric(build_ego_marker_grid(0.0))
    print(f'[ref] canonical (yaw=0) marker position: {ref}, expected ~= {WORLD_REL_OFFSET[:2]}')
    for yaw in (0.0, 90.0, -90.0, 37.0, -142.0):
        ego_grid = build_ego_marker_grid(yaw)
        canon = canonicalize_to_world_axes(ego_grid, yaw, order=1, is_scene=False)
        m = centroid_metric(canon)
        err = float(np.linalg.norm(m - ref))
        ok = err < VOXEL  # within 1 voxel, same tolerance convention as verify_temporal_alignment.py
        print(f'[{"OK" if ok else "FAIL"}] yaw={yaw:>7.1f}  canonical pos={m}  err={err:.3f}m (tol={VOXEL}m)')
        assert ok, f'yaw={yaw}: err {err:.3f}m exceeds 1 voxel tolerance'


def check_scene_padding_fix():
    # Build a dense 3-channel "scene" volume where every in-grid voxel is
    # split between road/vehicle (never empty), so that after a 90 deg
    # rotation, the corners rotated in from OUTSIDE the original square
    # footprint's circumscribed circle... actually a full square grid
    # rotated in-place has no true corners "outside" for a 90/180/270
    # rotation. Use an oblique 37deg rotation instead, which DOES leave
    # true corner regions with no source data (cval=0 padding), to
    # exercise the fix for real.
    rng = np.random.default_rng(0)
    road = rng.uniform(0.3, 0.7, size=(NX, NY, NZ)).astype(np.float32)
    vehicle = 1.0 - road
    empty = np.zeros((NX, NY, NZ), dtype=np.float32)
    scene = np.stack([empty, road, vehicle], axis=0)  # [3,X,Y,Z], sums to 1 everywhere
    assert np.allclose(scene.sum(axis=0), 1.0, atol=1e-5)

    canon = canonicalize_to_world_axes(scene, ego_yaw_deg=37.0, order=1, is_scene=True)
    sums = canon.sum(axis=0)
    print(f'[scene padding] channel-sum range after canonicalize: min={sums.min():.6f} max={sums.max():.6f}')
    assert np.allclose(sums, 1.0, atol=1e-4), 'every voxel must sum to ~1 after the padding fix'

    # padding voxels (far corner, guaranteed outside the original footprint
    # for a 37deg rotation of a full square) must read Empty=1 exactly.
    corner = canon[:, 0, 0, 0]
    print(f'[scene padding] corner voxel [P(empty),P(road),P(vehicle)] = {corner}')
    assert np.allclose(corner, [1.0, 0.0, 0.0], atol=1e-5), 'padding voxel must be exactly Empty=1'
    print('[OK] scene padding fix: no illegal all-zero voxels, padding reads Empty=1')


def check_target_not_padded():
    rng = np.random.default_rng(1)
    target = (rng.uniform(size=(1, NX, NY, NZ)) > 0.98).astype(np.float32)
    canon = canonicalize_to_world_axes(target, ego_yaw_deg=37.0, order=1, is_scene=False)
    corner = canon[0, 0, 0, 0]
    print(f'[target padding] corner voxel value = {corner} (expect 0.0, no Empty-style rewrite)')
    assert abs(corner) < 1e-6, 'target volumes must not be touched by the scene padding fix'
    print('[OK] target volume padding left alone (outside == 0, legal as-is)')


if __name__ == '__main__':
    check_marker_recovery()
    check_scene_padding_fix()
    check_target_not_padded()
    print('\nAll occ_canonicalize checks passed.')
