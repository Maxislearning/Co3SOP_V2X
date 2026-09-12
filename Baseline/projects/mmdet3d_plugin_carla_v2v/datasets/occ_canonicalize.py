"""Stage 3, Audit 1b: rotates Stage 2B's RX-centered, RX-YAW-ALIGNED Scene/
Target Occupancy into RX-centered, WORLD-AXIS-ALIGNED Occupancy, offline,
before either GT or Predicted volumes ever reach OccBeamDataset/the Beam
Encoder.

Why: Sionna's TX/RX beam codebook uses a fixed, heading-independent antenna
orientation (see the Stage 3 plan's Audit 1 -- carla_V2V/data/script.py's
Transmitter/Receiver both get orientation=(0,-pi/2,0) regardless of
position, and _setup_dft_codebook()'s angle table never touches a vehicle
pose either). Stage 2B's occupancy grids, by contrast, are built by
world_to_ego(pts_world, origin, yaw_deg) (generate_occupancy_gt.py:84-86),
which rotates by the RX's own yaw -- so the same world-frame beam direction
lands on a different occupancy grid direction depending on RX heading.
Canonicalizing removes that RX-yaw rotation (translation stays -- RX is
already the grid's own center) so the Beam Encoder sees a heading-
independent representation, without ever being given RX yaw as a feature.

world_to_ego's rotation is p_ego = rot_z(yaw_deg) @ (p_world - origin), where
this codebase's rot_z(t) = [[cos,sin,0],[-sin,cos,0],[0,0,1]]
(generate_occupancy_gt.py:76-81) -- i.e. rot_z(t)^-1 = rot_z(-t), so
p_canonical = (p_world - origin) = rot_z(-yaw_deg) @ p_ego. This is a PURE
2D rotation in the grid's own X-Y plane (Z/height untouched, no
translation) -- simpler than Stage 2C's warp_fused_to_t (which needed a
full 3D rigid transform between two *different* RX positions/times), so
this uses scipy.ndimage.rotate directly on the label-resolution
(0.4m/voxel, 200x200x16) volume rather than reusing warp_fused_to_t's
kornia machinery (which was built and tolerance-tuned for the coarse
1.2m/voxel *feature* grid -- conflating the two resolutions is exactly the
mistake Stage 2C caught).

VERIFIED, not assumed from the hand-derived sign above: a synthetic marker
at a fixed world-relative offset, placed into ego-frame grids at 5 RX yaws
(0, 90, -90, 37, -142 degrees) via the real world_to_ego/mark_pts, recovers
the same world-relative position after canonicalize_to_world_axes to within
one voxel (0.4m) at every angle tested -- including that the correct
scipy.ndimage.rotate `angle` argument is +ego_yaw_deg, NOT -ego_yaw_deg (the
naive point-transform derivation's sign gets flipped back by scipy's own
array-resampling convention -- caught by testing both signs against the
known answer, not by trusting the point-transform derivation applied
directly to the array call).
"""
import numpy as np
from scipy import ndimage

# Voxel grid axis order everywhere on disk (GT npz files, Step 1's saved
# canonical volumes): [X, Y, Z] for a bare grid, or [C, X, Y, Z] once a
# channel dim is added -- matches generate_occupancy_gt.py's own
# grid[NX, NY, NZ]. NOT the same as the PyTorch Conv3d [C, Z, Y, X] layout
# OccBeamDataset.__getitem__ permutes into -- that permute happens once,
# downstream of this file, never inside it.
_EMPTY_ONE_HOT = np.array([1.0, 0.0, 0.0], dtype=np.float32)


def canonicalize_to_world_axes(volume, ego_yaw_deg, order, is_scene=False, eps=1e-4):
    """volume: np.ndarray, shape [X,Y,Z] (single-channel) or [C,X,Y,Z]
    (multi-channel, C on axis 0). Rotates the X-Y plane (axes 1,2 if
    multi-channel, else axes 0,1) by `ego_yaw_deg` to undo world_to_ego's
    RX-yaw alignment. `order`: 1 (bilinear) for soft probabilities, 0
    (nearest) for GT categorical/binary masks -- caller's choice, this
    function doesn't infer it from dtype.

    is_scene=True (Scene channels only, GT one-hot or Pred softmax alike):
    applies the boundary-padding fix -- cval=0 leaves rotated-in edge voxels
    with an illegal all-zero [P(empty),P(road),P(vehicle)], which is not
    "zero probability of everything", it's "this voxel came from outside
    the originally observed grid" -- the physically correct reading is
    P(empty)=1. Valid (in-bounds) voxels are renormalized to sum to 1,
    cleaning up residual interpolation error, without touching the padding
    voxels' meaning. Target volumes must NOT use is_scene=True: `outside==0`
    is already the correct, legal reading there (no target observed there),
    nothing to renormalize against.
    """
    volume = np.asarray(volume, dtype=np.float32)
    multi_channel = volume.ndim == 4
    axes = (1, 2) if multi_channel else (0, 1)

    rotated = ndimage.rotate(
        volume, angle=ego_yaw_deg, axes=axes, reshape=False,
        order=order, mode='constant', cval=0.0)

    if is_scene:
        if not multi_channel or rotated.shape[0] != 3:
            raise ValueError('is_scene=True requires a [3,X,Y,Z] one-hot/softmax volume')
        channel_sum = rotated.sum(axis=0, keepdims=True)  # [1,X,Y,Z]
        empty_mask = (channel_sum < eps)[0]  # [X,Y,Z]
        # renormalize valid voxels
        safe_sum = np.where(channel_sum < eps, 1.0, channel_sum)
        rotated = rotated / safe_sum
        # explicit Empty=1 for rotated-in padding voxels
        rotated[0][empty_mask] = 1.0
        rotated[1][empty_mask] = 0.0
        rotated[2][empty_mask] = 0.0

    return rotated
