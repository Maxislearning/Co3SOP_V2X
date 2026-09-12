# ---------------------------------------------
# Stage 2C: K=3-frame temporal extension of TargetAwareOccHead (C1).
# Predicts t+1 Scene/Target Occupancy from camera history [t-2,t-1,t]
# instead of a single frame t (Stage 2B / C0's job). Reuses
# V2VOccHead._encode_and_fuse per frame UNCHANGED (same per-car encode +
# confidence-gated cross-agent warp/fuse it already does for one frame);
# this class only adds ONE new step on top -- warping each frame's fused
# feature from its own RX_k pose into the RX_t (last/current frame) pose
# before concatenating and decoding. That warp reuses
# get_discretized_transformation_matrix_3d/get_transformation_matrix_3d
# exactly as the existing inter-agent warp does, just fed
# temporal_trans_to_t instead of trans2ego -- same S-flip, same voxel_size
# formula, same call pattern.
#
# AXIS-ORDER REQUIREMENT (empirically verified, not guessed from names --
# see tools/verify_temporal_alignment.py's docstring for the full
# derivation and the standalone kornia probe that found this):
# _encode_and_fuse's return value (fused_k here) is shaped [B,C,W,H,Z], but
# kornia.warp_affine3d requires its (X,Y,Z)-ordered matrix rows to land on
# the tensor's (last,middle,first) axes respectively. Calling
# warp_affine3d directly on fused_k's native [B,C,W,H,Z] layout would put
# the X-translation row on the 4-voxel Z axis and any real forward/lateral
# motion falls out of range instantly (silently -- the identity/zero-motion
# case looks fine and masks the bug). So fused_k MUST be permuted to
# [B,C,Z,H,W] before warp_affine3d (dsize=(Z,H,W), matching the proven
# inter-agent call `warp_affine3d(features[car], M, (Z,H,W), ...)` at
# v2vocchead.py:338-343), then permuted back to [B,C,W,H,Z] afterward.
# ---------------------------------------------
import kornia
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import force_fp32
from mmdet.models import HEADS

from projects.mmdet3d_plugin.co3sop_base.dense_heads.target_occ_head import TargetAwareOccHead
from projects.mmdet3d_plugin.co3sop_base.dense_heads.v2vocchead import (
    get_discretized_transformation_matrix_3d, get_transformation_matrix_3d,
)
from projects.mmdet3d_plugin.co3sop_base.loss.loss_utils import multiscale_supervision

# Same axis-flip _encode_and_fuse applies to trans2ego before warping
# (v2vocchead.py:319-321) -- kept identical here for temporal_trans_to_t.
S_FLIP = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)


def warp_fused_to_t(fused_k, img_metas_k, identity=False):
    """Shared by TemporalTargetAwareOccHead (v1, naive concat) and
    GatedTemporalTargetAwareOccHead (v2, gated residual) -- this warp step
    itself doesn't change between v1/v2, only what happens to its output
    afterward does. Extracted to a module-level function (it never used
    `self`) so v2 doesn't have to inherit v1's temporal_fusion/aux_occ
    modules just to reuse this.

    fused_k: [B,C,W,H,Z] -- Co3SOP's NATIVE fused-feature layout (see
    _encode_and_fuse / v2vocchead.py:359-366). This is NOT the same axis
    order kornia.warp_affine3d requires -- see the AXIS-ORDER REQUIREMENT
    note in this module's header docstring and tools/verify_temporal_
    alignment.py: kornia's (X,Y,Z)-ordered matrix rows must land on the
    tensor's (last,middle,first) *spatial* axes, so this function converts
    Co3SOP's native [B,C,W,H,Z] layout to kornia's required [B,C,Z,H,W]
    layout before calling warp_affine3d, and converts back to [B,C,W,H,Z]
    (Co3SOP's convention) on the way out. Every caller of this function
    receives/passes Co3SOP-layout tensors; the kornia-layout permutation is
    entirely internal to this function.

    img_metas_k: list of B dicts, each carrying that batch item's raw (no
    S-flip) temporal_trans_to_t matrix for this frame k (dataset-computed
    x1_to_x2(pose_k, pose_t), identity when k is already the last/current
    frame).

    identity=True (Stage 2C-v2 "misaligned-history" sanity check only):
    skip the real temporal_trans_to_t entirely and warp with an identity
    matrix instead -- tests whether the model depends on *correct*
    geometric alignment of otherwise-real, own-sample history content,
    as distinct from the "shuffled-history" check (real alignment, wrong
    sample's content) that lives in the analysis script.
    """
    B, C, W, H, Z = fused_k.shape
    voxel_size = 0.1 * 48 / Z  # exact formula _encode_and_fuse uses (v2vocchead.py:313)

    if identity:
        raw = np.tile(np.eye(4, dtype=np.float64)[None], (B, 1, 1))
    else:
        raw = np.stack([np.asarray(m['temporal_trans_to_t'], dtype=np.float64)
                         for m in img_metas_k], axis=0)  # [B,4,4], no S-flip yet
        raw = raw.copy()
        raw[:, :3, :3] = S_FLIP[None] @ raw[:, :3, :3] @ S_FLIP[None]
        raw[:, :3, 3] = np.einsum('ij,bj->bi', S_FLIP, raw[:, :3, 3])

    matrix = fused_k.new_tensor(raw).view(B, 1, 4, 4)
    matrix = get_discretized_transformation_matrix_3d(matrix, voxel_size, 1)
    matrix = matrix.view(B, 3, 4)
    # dsize argument order for get_transformation_matrix_3d must literally
    # mirror the proven inter-agent call at v2vocchead.py:327 -- (W,H,Z),
    # NOT (Z,H,W) (get_rotation_matrix3d's own `H,W,Z = dsize` unpacking
    # makes this a *different* convention than warp_affine3d's own dsize
    # below -- harmless in the real model only because H==W==50 there, but
    # the call site's literal argument order must still be copied exactly).
    matrix = get_transformation_matrix_3d(matrix, (W, H, Z)).view(B, 3, 4)

    # Co3SOP native layout [B,C,W,H,Z] -> kornia-required layout [B,C,Z,H,W].
    fused_k_zhw = fused_k.permute(0, 1, 4, 3, 2).contiguous()
    aligned_zhw = kornia.geometry.transform.warp_affine3d(
        fused_k_zhw, matrix, (Z, H, W), flags='bilinear', padding_mode='zeros', align_corners=True)
    # kornia layout -> back to Co3SOP native [B,C,W,H,Z].
    aligned = aligned_zhw.permute(0, 1, 4, 3, 2).contiguous()
    return aligned


@HEADS.register_module(force=True)
class TemporalTargetAwareOccHead(TargetAwareOccHead):
    # Fixed at 3 for Stage 2C v1 -- deliberately not a general K=1..N head
    # (plan §3: "not written as a false 'K=1 also works' generic class").
    QUEUE_LEN = 3

    def __init__(self, *args, lambda_aux=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        # OFF by default for the main C0-vs-C1 comparison: L_aux_scene only
        # exists on C1 (C0 has no history frames to supervise), so leaving
        # it on would mean C1 gets both temporal history *and* extra
        # supervision C0 structurally can't have -- any gain couldn't be
        # cleanly attributed to temporal history alone, which is Stage 2C's
        # actual question. Module is still built (cheap) so a later
        # lambda_aux>0 run (a separate, clearly-labeled ablation) doesn't
        # need new code.
        self.lambda_aux = lambda_aux

        fused_channels = self.conv_input[0]  # 192 = _dim_[0], fused_k's channel count pre-deblocks
        self.temporal_fusion = nn.Sequential(
            build_conv_layer(
                dict(type='Conv3d', bias=False),
                in_channels=fused_channels * self.QUEUE_LEN,
                out_channels=fused_channels,
                kernel_size=3, stride=1, padding=1),
            build_norm_layer(dict(type='GN', num_groups=24, requires_grad=True), fused_channels)[1],
            nn.ReLU(inplace=True),
        )
        self.aux_occ = build_conv_layer(
            dict(type='Conv3d', bias=False),
            in_channels=fused_channels, out_channels=self.num_classes,
            kernel_size=1, stride=1, padding=0)

    def _warp_to_t(self, fused_k, img_metas_k, identity=False):
        """Thin wrapper kept for backward compatibility -- body now lives in
        the module-level warp_fused_to_t (shared with v2), see its
        docstring for the full axis-layout explanation."""
        return warp_fused_to_t(fused_k, img_metas_k, identity=identity)

    def forward(self, mcar_feats_list, img_metas_list):
        """mcar_feats_list / img_metas_list: length QUEUE_LEN, frame order
        [t-2, t-1, t] (last = current/reference frame). Each img_metas_list[k]
        is a list of B per-batch-item dicts (matching every existing
        single-frame img_metas convention)."""
        assert len(mcar_feats_list) == self.QUEUE_LEN == len(img_metas_list)

        aligned_feats = []
        aux_scene_preds = [] if self.lambda_aux > 0 else None
        for k in range(self.QUEUE_LEN):
            fused_k, _, _ = self._encode_and_fuse(mcar_feats_list[k], img_metas_list[k])
            if self.lambda_aux > 0:
                aux_scene_preds.append(self.aux_occ(fused_k))
            aligned_feats.append(self._warp_to_t(fused_k, img_metas_list[k]))

        fused_cat = torch.cat(aligned_feats, dim=1)  # [B, C*QUEUE_LEN, W, H, Z]
        fused_temporal = self.temporal_fusion(fused_cat)  # [B, C, W, H, Z]

        outputs = self._run_deblocks(fused_temporal)
        occ_preds = [self.occ[i](outputs[i]) for i in range(len(outputs))]

        role = fused_temporal.new_tensor(
            [m['target_role'] for m in img_metas_list[-1]], dtype=torch.long)
        role_embed = self.role_embedding(role)

        target_preds = []
        for i in range(len(outputs)):
            scale, shift = self.film_proj[i](role_embed).chunk(2, dim=1)
            scale = scale[:, :, None, None, None]
            shift = shift[:, :, None, None, None]
            modulated = outputs[i] * (1 + scale) + shift
            target_preds.append(self.target_occ[i](modulated))

        result = {'occ_preds': occ_preds, 'target_preds': target_preds}
        if self.lambda_aux > 0:
            result['aux_scene_preds'] = aux_scene_preds
        return result

    @force_fp32(apply_to=('preds_dicts',))
    def loss(self, gt_occ_future, gt_target_future, preds_dicts, img_metas_list, gt_occ_aux_k=None):
        # Scene/target loss on the t+1 prediction: identical math to
        # TargetAwareOccHead.loss (gt_occ_future/gt_target_future are just
        # another dense voxel grid, same shape contract as Stage 2B's
        # current-frame gt_occ/gt_target) -- reused unchanged, not
        # reimplemented. img_metas isn't actually read inside that method
        # today, but pass frame t's own metas for correctness.
        loss_dict = super().loss(gt_occ_future, gt_target_future, preds_dicts, img_metas_list[-1])

        if self.lambda_aux > 0:
            assert gt_occ_aux_k is not None, 'lambda_aux>0 requires gt_occ_aux_k from the dataset'
            criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
            for k, pred in enumerate(preds_dicts['aux_scene_preds']):
                pred = pred.float()
                # aux_occ runs at fused_k's native (undownsampled) 50x50x4
                # resolution -- ratio=4 matches NX/W=NY/H=NZ/Z=200/50=16/4=4.
                gt_k = multiscale_supervision(gt_occ_aux_k[:, k].clone(), 4, pred.shape)
                loss_dict[f'loss_aux_scene_{k}'] = self.lambda_aux * criterion(pred, gt_k.long())

        return loss_dict
