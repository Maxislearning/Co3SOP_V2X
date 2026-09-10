#!/usr/bin/env python3
"""Stage 2C §3b: synthetic correctness test for temporal_trans_to_t, required
before any model code touches it. Doesn't trust x1_to_x2's docstring/name --
derives the expected transform by hand from x_to_world()'s actual (not the
unused eulerAngles2rotationMat call inside it -- that result is computed and
then never assigned, see transformation_utils.py:62,70) rotation formula,
and checks both the raw 4x4 matrix AND the full discretize+warp_affine3d
pipeline (mirroring V2VOccHead._encode_and_fuse's inter-agent alignment call
pattern exactly -- same S-flip, same voxel_size formula, same argument
order -- since that pipeline is already empirically proven correct by
Stage 2B's REAR target IoU=0.85).

pose format throughout: [x, y, z, roll, yaw, pitch], degrees, world frame
(same convention as trans2ego's inputs, NOT generate_occupancy_gt.py's own
world_to_ego/rot_z -- those are a separate utility with its own convention,
not used here).
"""
import sys
import os

import numpy as np
import torch
import kornia

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from projects.mmdet3d_plugin.models.utils.transformation_utils import x1_to_x2
from projects.mmdet3d_plugin.co3sop_base.dense_heads.v2vocchead import (
    get_discretized_transformation_matrix_3d, get_transformation_matrix_3d,
)

Z, H, W = 4, 50, 50  # matches volume_z_/volume_h_/volume_w_ in the carla_v2v configs
VOXEL_SIZE = 0.1 * 48 / Z  # exact formula _encode_and_fuse uses -- 1.2m at Z=4
S = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])  # same axis-flip _encode_and_fuse applies before warping


def analytic_check(name, pose_k, pose_t, expected_translation, tol=1e-6):
    M = x1_to_x2(np.array(pose_k, dtype=np.float64), np.array(pose_t, dtype=np.float64))
    t = M[:3, 3]
    ok = np.allclose(t, expected_translation, atol=tol)
    print(f'[{"OK" if ok else "FAIL"}] analytic {name}: x1_to_x2 translation = {t}, expected {expected_translation}')
    assert ok, f'{name}: got {t}, expected {expected_translation}'
    return M


def analytic_point_check(name, pose_k, pose_t, point_k, expected_point_t, tol=1e-6):
    """Applies the *full* matrix (rotation+translation) to a specific point
    in k's local frame and checks where it lands in t's frame -- needed for
    the yaw case, where the raw translation column alone (both poses at the
    same world position) is correctly all-zero and doesn't by itself say
    anything about how a non-origin point gets rotated."""
    M = x1_to_x2(np.array(pose_k, dtype=np.float64), np.array(pose_t, dtype=np.float64))
    p = M @ np.array([*point_k, 1.0])
    ok = np.allclose(p[:3], expected_point_t, atol=tol)
    print(f'[{"OK" if ok else "FAIL"}] analytic {name}: point {point_k} in k-frame -> {p[:3]} in t-frame, expected {expected_point_t}')
    assert ok, f'{name}: got {p[:3]}, expected {expected_point_t}'


def warp_check(name, pose_k, pose_t, marker_grid_zhw, expected_metric_shift_xy):
    """Places a marker at a known voxel, runs the exact _encode_and_fuse warp
    pipeline (S-flip -> discretize -> transformation matrix -> warp_affine3d),
    and checks the weighted centroid of the warped marker moved by the
    expected metric amount (centroid, not exact-voxel, since bilinear
    interpolation smears a hard 1.0 marker across neighbors).

    IMPORTANT axis-order finding (empirically probed, not assumed -- this is
    exactly the kind of thing that's easy to get backwards, per the review
    that asked for this test, and a first attempt at this test DID get it
    backwards): kornia.warp_affine3d's 3x4 matrix rows are in (X,Y,Z) pixel-
    coordinate order, where row0's translation always lands on the tensor's
    *last* spatial axis (dim4), row1 on the middle (dim3), row2 on the first
    (dim2) -- confirmed with a minimal standalone probe (translate row0 alone
    on a [1,1,10,10,10] tensor -> marker moves along dim4, not dim2).

    fused_k (_encode_and_fuse's actual return value) is shaped
    `[bs, C, W, H, Z]` (dense_heads/v2vocchead.py:359-366,
    `.reshape(bs,Z,H,W,-1).permute(0,4,3,2,1)`) -- its *last* axis is Z
    (vertical, only 4 voxels). Calling warp_affine3d directly on that native
    layout with dsize=(W,H,Z) would put the X-translation row on the Z axis
    and any real X shift falls out of the 4-voxel range instantly (this is
    exactly what happened: 'marker vanished, total mass 0.0' for a pure
    +X-translation case, while the identity case -- zero translation --
    looked fine and masked the bug). The proven inter-agent warp avoids this
    because `features[car]` is `[bs,C,Z,H,W]` (dsize=(Z,H,W)) -- its last
    axis (W) really is world-X, matching row0.

    So the real C1 model code must PERMUT fused_k from `[bs,C,W,H,Z]` to
    `[bs,C,Z,H,W]` (`.permute(0,1,4,3,2)`) before calling
    `warp_affine3d(..., dsize=(Z,H,W))`, then permute the result back to
    `[bs,C,W,H,Z]` before concatenation/temporal_fusion/_run_deblocks. This
    test builds its marker tensor directly in that already-correct
    `[1,1,Z,H,W]` layout (mirroring the inter-agent case exactly), since
    that's what warp_affine3d itself must always be called with.
    """
    matrix = x1_to_x2(np.array(pose_k, dtype=np.float64), np.array(pose_t, dtype=np.float64))
    matrix = matrix.copy()
    matrix[:3, :3] = S @ matrix[:3, :3] @ S
    matrix[:3, 3] = S @ matrix[:3, 3]

    batch_transform_matrix = torch.tensor(matrix, dtype=torch.float32).view(1, 1, 4, 4)
    batch_transform_matrix = get_discretized_transformation_matrix_3d(batch_transform_matrix, VOXEL_SIZE, 1)
    batch_transform_matrix = batch_transform_matrix.view(1, 3, 4)
    # NOTE: get_transformation_matrix_3d's own dsize argument order must
    # exactly mirror the proven inter-agent call at v2vocchead.py:327,
    # `get_transformation_matrix_3d(batch_transform_matrix, (W,H,Z))` --
    # literally (W,H,Z), NOT (Z,H,W) (get_rotation_matrix3d's internal
    # `H,W,Z = dsize` unpacking makes this a *different* argument-order
    # convention than warp_affine3d's own dsize below; harmless in the real
    # model only because its H==W==50 makes the H/W swap invisible there --
    # but the call site's literal argument order must still be copied
    # exactly, not "fixed" to look consistent, since that's not what was
    # empirically proven to work).
    batch_transform_matrix = get_transformation_matrix_3d(batch_transform_matrix, (W, H, Z)).view(1, 3, 4)

    feat = torch.zeros(1, 1, Z, H, W)  # matches warp_affine3d's required (X,Y,Z)-row / last-axis=X convention
    zi, hi, wi = marker_grid_zhw
    feat[0, 0, zi, hi, wi] = 1.0

    warped = kornia.geometry.transform.warp_affine3d(
        feat, batch_transform_matrix, (Z, H, W), flags='bilinear', padding_mode='zeros', align_corners=True)

    total = warped.sum().item()
    if total < 1e-6:
        print(f'[FAIL] warp {name}: marker vanished entirely (out of grid) -- total mass {total}')
        assert False, f'{name}: marker mass vanished'
    idx = torch.nonzero(warped[0, 0] > 1e-8, as_tuple=False).float()
    weights = warped[0, 0][warped[0, 0] > 1e-8]
    centroid_zhw = (idx * weights[:, None]).sum(0) / weights.sum()
    # grid index -> metric: dim0=Z(->Z), dim1=H(->Y), dim2=W(->X), matching
    # this tensor's [Z,H,W] layout (same convention the proven inter-agent
    # warp already uses for features[car]).
    # metric conversion uses VOXEL_SIZE (the constant the discretization
    # step itself already uses, `0.1*48/Z`), NOT the fine label grid's
    # 80m/(-40,40) convention -- those are different grids. VOXEL_SIZE=1.2m
    # * W=50 gives a 60m extent for this coarse feature volume, not 80m; a
    # first version of this conversion wrongly assumed the 80m fine-grid
    # scale here, which silently made a correct -4.0 voxel shift look like
    # the wrong metric amount (-6.4m instead of -4.8m) despite the pipeline
    # itself being right -- caught by cross-checking against the discretized
    # matrix's own exact translation value (-4.0 voxels) computed separately.
    cz, ch, cw = centroid_zhw.tolist()
    cx = (cw - W / 2) * VOXEL_SIZE
    cy = (ch - H / 2) * VOXEL_SIZE
    orig_x = (wi - W / 2) * VOXEL_SIZE
    orig_y = (hi - H / 2) * VOXEL_SIZE
    shift = (cx - orig_x, cy - orig_y)
    ok = np.allclose(shift, expected_metric_shift_xy, atol=VOXEL_SIZE)  # within 1 voxel of the analytic prediction
    print(f'[{"OK" if ok else "FAIL"}] warp {name}: marker centroid shifted by {shift}, expected {expected_metric_shift_xy} (tol={VOXEL_SIZE}m)')
    assert ok, f'{name}: got shift {shift}, expected {expected_metric_shift_xy}'


def main():
    print(f'Z,H,W={Z},{H},{W}  voxel_size={VOXEL_SIZE}m')

    # 1. Identity: pose_k == pose_t -> exact identity matrix, marker doesn't move.
    p = [10.0, -3.0, 0.0, 0.0, 37.0, 0.0]
    M = analytic_check('identity', p, p, expected_translation=[0, 0, 0])
    assert np.allclose(M, np.eye(4), atol=1e-8), 'identity case must be exact eye(4)'

    # 2. Pure world +X translation, zero yaw both: ego moves from world X=0 to X=5.
    #    A marker at frame k's local origin should appear 5m BEHIND ego in frame t
    #    (translation = -5 in X) -- verified by hand from x_to_world()'s actual
    #    (yaw=0 -> identity rotation) formula, see module docstring.
    analytic_check('pure +X ego motion', [0, 0, 0, 0, 0, 0], [5, 0, 0, 0, 0, 0], expected_translation=[-5, 0, 0])

    # 3. Pure world +Y translation, zero yaw both.
    analytic_check('pure +Y ego motion', [0, 0, 0, 0, 0, 0], [0, 5, 0, 0, 0, 0], expected_translation=[0, -5, 0])

    # 4. Pure 90 deg yaw change, zero translation: the raw translation column
    #    is correctly [0,0,0] here (both poses share the same world position) --
    #    the rotation only shows up when applied to an actual point. A marker
    #    10m ahead of k (k-local (10,0,0), = world (10,0,0) since k has yaw=0
    #    and sits at the world origin) must land at t-local (0,-10,0) once t is
    #    rotated 90 deg -- not a mirrored/flipped position.
    analytic_check('pure 90deg yaw (translation column)', [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 90, 0], expected_translation=[0, 0, 0])
    analytic_point_check('pure 90deg yaw (rotated point)', [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 90, 0],
                          point_k=[10, 0, 0], expected_point_t=[0, -10, 0])

    # Now the full discretize+warp_affine3d pipeline, X/Y translation cases
    # (yaw=0 in both so the S-flip's rotation part is inert here, see the
    # module docstring -- this checks the grid-index/warp machinery itself,
    # complementing the pure-matrix checks above).
    marker = (Z // 2, H // 2, W // 2)  # center of the grid, metric (~0,~0,~) roughly, in [Z,H,W] index order
    warp_check('pure +X ego motion (grid)', [0, 0, 0, 0, 0, 0], [4.8, 0, 0, 0, 0, 0],
               marker, expected_metric_shift_xy=(-4.8, 0))
    warp_check('pure +Y ego motion (grid)', [0, 0, 0, 0, 0, 0], [0, 4.8, 0, 0, 0, 0],
               marker, expected_metric_shift_xy=(0, -4.8))

    print('\nAll temporal alignment checks passed.')


if __name__ == '__main__':
    main()
