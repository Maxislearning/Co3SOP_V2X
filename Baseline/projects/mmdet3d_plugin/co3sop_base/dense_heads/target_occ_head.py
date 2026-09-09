# ---------------------------------------------
# Stage 2B: Target-aware Occupancy head. Scene branch reuses V2VOccHead
# unchanged (real GT for the beam run exists for the first time since
# Stage 2A); adds a second, FiLM-conditioned branch off the SAME shared
# deblocks features that predicts a binary target/non-target mask for
# whichever TX (FRONT/REAR) the current sample's link corresponds to.
#
# GPS-free boundary (see plan / 记录something/V2X_beam_selection_实验记录.txt
# 任务7 for the full audit): inter-agent trans2ego-based warp inside
# V2VOccHead._encode_and_fuse is kept as-is -- that's CO3SOP's cooperative
# perception alignment mechanism, not in scope to remove. What this head
# must not do is take tx_rx_geometry/TX xyz/distance/azimuth as a forward
# input -- role conditioning is the FRONT/REAR identity token only.
# ---------------------------------------------
import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer
from mmcv.runner import force_fp32
from mmdet.models import HEADS

from projects.mmdet3d_plugin.co3sop_base.dense_heads.v2vocchead import V2VOccHead
from projects.mmdet3d_plugin.co3sop_base.loss.loss_utils import (
    multiscale_supervision, geo_scal_loss, sem_scal_loss, focal_loss, dice_loss,
)

FRONT, REAR = 0, 1


@HEADS.register_module(force=True)
class TargetAwareOccHead(V2VOccHead):
    def __init__(self,
                 *args,
                 role_dim=32,
                 freeze_scene_branch=False,
                 lambda_scene=1.0,
                 lambda_target=1.0,
                 lambda_focal=1.0,
                 lambda_dice=1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_scene_branch = freeze_scene_branch
        self.lambda_scene = lambda_scene
        self.lambda_target = lambda_target
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice

        self.role_embedding = nn.Embedding(2, role_dim)  # 0=FRONT, 1=REAR -- identity only

        conv_cfg = dict(type='Conv3d', bias=False)
        self.film_proj = nn.ModuleList()
        self.target_occ = nn.ModuleList()
        for i in self.out_indices:
            channels = self.conv_output[i]
            self.film_proj.append(nn.Linear(role_dim, 2 * channels))
            self.target_occ.append(build_conv_layer(
                conv_cfg, in_channels=channels, out_channels=2,
                kernel_size=1, stride=1, padding=0))

        if self.freeze_scene_branch:
            self._freeze_scene_branch()

    def _freeze_scene_branch(self):
        print('freeze_scene_branch')
        for module in (self.transformer, self.v2v_fuse, self.deblocks,
                       self.occ, self.confidence, self.volume_embedding):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(V2VOccHead, self).train(mode)
        if self.freeze_scene_branch:
            self._freeze_scene_branch()

    def forward(self, mcar_feats, img_metas):
        fused, _, _ = self._encode_and_fuse(mcar_feats, img_metas)
        outputs = self._run_deblocks(fused)

        occ_preds = [self.occ[i](outputs[i]) for i in range(len(outputs))]

        role = fused.new_tensor(
            [m['target_role'] for m in img_metas], dtype=torch.long)
        role_embed = self.role_embedding(role)  # [B, role_dim]

        target_preds = []
        for i in range(len(outputs)):
            scale, shift = self.film_proj[i](role_embed).chunk(2, dim=1)
            scale = scale[:, :, None, None, None]
            shift = shift[:, :, None, None, None]
            modulated = outputs[i] * (1 + scale) + shift
            target_preds.append(self.target_occ[i](modulated))

        return {'occ_preds': occ_preds, 'target_preds': target_preds}

    @force_fp32(apply_to=('preds_dicts',))
    def loss(self, gt_occ, gt_target, preds_dicts, img_metas):
        loss_dict = {}

        # Scene: identical to V2VOccHead.loss()'s semantic branch (reused,
        # not reimplemented).
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        for i, pred in enumerate(preds_dicts['occ_preds']):
            pred = pred.float()
            ratio = int(2 ** (len(preds_dicts['occ_preds']) - 1 - i))
            gt = multiscale_supervision(gt_occ.clone(), ratio, pred.shape)
            loss_i = criterion(pred, gt.long()) + geo_scal_loss(pred, gt.long()) + sem_scal_loss(pred, gt.long())
            loss_dict[f'loss_scene_{i}'] = self.lambda_scene * loss_i * (0.5 ** (len(preds_dicts['occ_preds']) - 1 - i))

        # Target: focal + dice, same multiscale_supervision downsampling
        # (target_mask is just another dense grid, same helper applies).
        for i, pred in enumerate(preds_dicts['target_preds']):
            pred = pred.float()
            ratio = int(2 ** (len(preds_dicts['target_preds']) - 1 - i))
            gt = multiscale_supervision(gt_target.clone(), ratio, pred.shape)
            loss_focal = focal_loss(pred, gt.long())
            loss_dice = dice_loss(pred, gt.long())
            loss_i = self.lambda_focal * loss_focal + self.lambda_dice * loss_dice
            loss_dict[f'loss_target_{i}'] = self.lambda_target * loss_i * (0.5 ** (len(preds_dicts['target_preds']) - 1 - i))

        return loss_dict
