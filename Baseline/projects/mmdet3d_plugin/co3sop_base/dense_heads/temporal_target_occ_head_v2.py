# ---------------------------------------------
# Stage 2C-v2: current-centric GATED RESIDUAL temporal fusion. Rescue
# experiment after v1's naive-concat design (temporal_target_occ_head.py)
# came back inconclusive-to-negative: v1 degraded when history was zeroed,
# but did NOT degrade when history was shuffled to another sample's content
# or its order reversed -- suggesting v1 depended on "is there a nonzero
# signal in the history slots" rather than on correct temporal
# correspondence. See 记录something/V2X_beam_selection_实验记录.txt 任务10.
#
# Reuses the exact same per-frame encode/fuse/warp as v1 -- _encode_and_fuse
# (inherited from V2VOccHead via TargetAwareOccHead, unchanged) and
# warp_fused_to_t (imported from temporal_target_occ_head.py, NOT
# duplicated, so both v1/v2 share one axis-layout-correct implementation).
# The only thing that changes is what happens AFTER warping: instead of
# concatenating all 3 aligned frames and convolving (v1), the CURRENT
# frame's own feature is the explicit anchor, and each history frame
# contributes only a gated RESIDUAL correction relative to that anchor:
#
#   H2, H1, C = warp_fused_to_t(fused_t-2), warp_fused_to_t(fused_t-1), warp_fused_to_t(fused_t)
#   D1, D2 = H1 - C, H2 - C                    # "what changed" vs now, not "what's present"
#   g1 = sigmoid(Gate1(cat[C, D1]))            # per-channel gate, not scalar
#   g2 = sigmoid(Gate2(cat[C, D2]))
#   R  = TemporalConv(cat[g1*D1, g2*D2])
#   F_temporal = C + R
#
# TENSOR-LAYOUT CONVENTION (repeated here deliberately, not just in
# warp_fused_to_t's docstring, precisely so this bug class can't recur):
# every tensor in this file (fused_k, H2/H1/C, D1/D2, gate/conv inputs and
# outputs, F_temporal) is in Co3SOP's NATIVE feature layout [B,C,W,H,Z].
# This is DIFFERENT from kornia.warp_affine3d's own required spatial layout
# [B,C,Z,H,W] (its (X,Y,Z)-ordered matrix rows must land on the tensor's
# (last,middle,first) axes) -- that permutation is entirely internal to
# warp_fused_to_t and never leaks out to this file. Every op here
# (subtraction, concatenation, conv) is elementwise/channel-wise and layout-
# agnostic, so nothing downstream of warp_fused_to_t needs to know about the
# kornia layout at all -- but a future reader must not assume this file's
# [B,C,W,H,Z] tensors can be passed directly into a raw kornia call.
#
# ZERO-INIT: temporal_conv2 (the LAST layer of the R branch) has its weight
# AND bias zero-initialized, so at init R==0 exactly and F_temporal==C --
# the model starts out strictly equivalent to C0 (same Stage 2B checkpoint
# init for everything else), and only learns to pull in temporal residual
# corrections during training, instead of a freshly-initialized fusion
# block immediately perturbing the pretrained decoder's input distribution.
# Gate1/Gate2 do NOT need special init for this guarantee to hold: R is
# zero regardless of what g1*D1/g2*D2 evaluate to, since temporal_conv2's
# own weights are zero.
# ---------------------------------------------
import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import force_fp32
from mmdet.models import HEADS

from projects.mmdet3d_plugin.co3sop_base.dense_heads.target_occ_head import TargetAwareOccHead
from projects.mmdet3d_plugin.co3sop_base.dense_heads.temporal_target_occ_head import warp_fused_to_t


@HEADS.register_module(force=True)
class GatedTemporalTargetAwareOccHead(TargetAwareOccHead):
    QUEUE_LEN = 3  # same as v1, fixed for this stage (see temporal_target_occ_head.py's note)

    def __init__(self, *args, lambda_aux=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        # lambda_aux accepted ONLY because co3sop_base_carla_v2v_temporal_c1_v2.py
        # inherits pts_bbox_head from v1's config via mmcv's `_base_` (a
        # recursive dict-merge, not a full replace) and v1's config sets
        # lambda_aux=0.0 -- this is a config-compatibility no-op, not a real
        # feature. There is NO aux_occ module and NO aux loss term anywhere
        # in this class, at any lambda_aux value; Stage 2C-v2's main
        # experiment doesn't touch auxiliary scene supervision at all, per
        # the approved plan ("不要先开启 λ_aux", "不要把 aux 混入主结果").
        C = self.conv_input[0]  # 192 = _dim_[0], fused_k's channel count pre-deblocks

        conv_cfg = dict(type='Conv3d', bias=True)
        self.gate1 = build_conv_layer(conv_cfg, in_channels=2 * C, out_channels=C,
                                       kernel_size=1, stride=1, padding=0)
        self.gate2 = build_conv_layer(conv_cfg, in_channels=2 * C, out_channels=C,
                                       kernel_size=1, stride=1, padding=0)

        self.temporal_conv1 = nn.Sequential(
            build_conv_layer(conv_cfg, in_channels=2 * C, out_channels=C,
                              kernel_size=3, stride=1, padding=1),
            build_norm_layer(dict(type='GN', num_groups=24, requires_grad=True), C)[1],
            nn.ReLU(inplace=True),
        )
        self.temporal_conv2 = build_conv_layer(conv_cfg, in_channels=C, out_channels=C,
                                                kernel_size=1, stride=1, padding=0)
        nn.init.zeros_(self.temporal_conv2.weight)
        nn.init.zeros_(self.temporal_conv2.bias)

    def fuse_aligned(self, aligned_feats):
        """aligned_feats: [H2, H1, C] (order matches self.QUEUE_LEN's frame
        order [t-2,t-1,t]), each [B,C,W,H,Z], already warped into RX_t.
        Split out from fuse_temporal so eval-time sanity checks (Stage 2C-v2
        shuffled-history in particular) can splice a donor sample's H2/H1
        together with this sample's own C without re-deriving anything --
        the donor's H2/H1 must themselves already be the donor's OWN
        correctly-encoded-and-warped features (own backbone, own intra-frame
        Co3SOP fusion, own temporal alignment), never raw donor images
        spliced into this sample's pipeline (that would also corrupt
        per-frame calibration, not just cross-time correspondence -- see the
        analysis script's docstring for why v1's shuffled-history check was
        flawed in exactly this way).

        Returns (F_temporal, (g1, g2)) -- gates returned for the gate-
        activation diagnostics (mean/std/saturation), not used in forward().
        """
        H2, H1, C = aligned_feats
        D1, D2 = H1 - C, H2 - C  # "what changed vs now", all still in [B,C,W,H,Z] layout
        g1 = torch.sigmoid(self.gate1(torch.cat([C, D1], dim=1)))
        g2 = torch.sigmoid(self.gate2(torch.cat([C, D2], dim=1)))
        R = self.temporal_conv2(self.temporal_conv1(torch.cat([g1 * D1, g2 * D2], dim=1)))
        return C + R, (g1, g2)

    def fuse_temporal(self, mcar_feats_list, img_metas_list, warp_kwargs=None):
        """Runs _encode_and_fuse + warp_fused_to_t for all QUEUE_LEN frames,
        then fuse_aligned. warp_kwargs: optional list of length QUEUE_LEN,
        each a dict of extra kwargs for that frame's warp_fused_to_t call
        (e.g. [{}, {}, {}] normally; the misaligned-history sanity check
        passes {'identity': True} for the history frames' entries to skip
        the real temporal_trans_to_t)."""
        warp_kwargs = warp_kwargs or [{}] * self.QUEUE_LEN
        aligned_feats = []
        for k in range(self.QUEUE_LEN):
            fused_k, _, _ = self._encode_and_fuse(mcar_feats_list[k], img_metas_list[k])
            aligned_feats.append(warp_fused_to_t(fused_k, img_metas_list[k], **warp_kwargs[k]))
        return self.fuse_aligned(aligned_feats)

    def forward(self, mcar_feats_list, img_metas_list):
        """mcar_feats_list / img_metas_list: length QUEUE_LEN, frame order
        [t-2, t-1, t] (last = current/reference frame), matching v1 exactly."""
        assert len(mcar_feats_list) == self.QUEUE_LEN == len(img_metas_list)
        fused_temporal, _gates = self.fuse_temporal(mcar_feats_list, img_metas_list)

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

        return {'occ_preds': occ_preds, 'target_preds': target_preds}

    @force_fp32(apply_to=('preds_dicts',))
    def loss(self, gt_occ_future, gt_target_future, preds_dicts, img_metas_list, gt_occ_aux_k=None):
        # v2 has no auxiliary scene branch at all -- gt_occ_aux_k is accepted
        # and ignored purely because Co3SOPTemporalTargetOcc.forward_train
        # (shared detector code with v1) always passes it as a kwarg. NOT
        # swallowing it here would crash with the exact same "unexpected
        # keyword argument" TypeError already hit once this stage (v1's
        # target_center_raw bug) -- forward_train() has its own **kwargs for
        # the dataset-batch side of this, but the head's own loss() call
        # site is a separate, explicit argument list that needs the same
        # treatment. Scene/target loss itself is identical math to
        # TargetAwareOccHead.loss (reused unchanged, not reimplemented).
        return super().loss(gt_occ_future, gt_target_future, preds_dicts, img_metas_list)
