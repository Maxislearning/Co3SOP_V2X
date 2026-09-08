# ---------------------------------------------
# Beam selection head for CO3SOP-derived V2V beam prediction.
# Reuses V2VOccHead's per-car voxel encoding + cross-agent fusion
# (V2VOccHead._encode_and_fuse) and branches off the fused 3D scene feature
# instead of decoding it into occupancy logits.
# ---------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet.models import HEADS

from projects.mmdet3d_plugin.co3sop_base.dense_heads.v2vocchead import V2VOccHead

VEHICLE_CLASS_IDX = 2  # class_names = ['empty', 'road', 'vehicle']


@HEADS.register_module(force=True)
class BeamSelectionHead(V2VOccHead):
    """Two-head (TX beam / RX beam) classification on top of the fused
    ego-centric 3D volume `V2VOccHead._encode_and_fuse` already produces.

    Joint (tx, rx) classification isn't used here: with a 64x64 DFT codebook
    (see carla_V2V/data/script.py RealisticMIMOBeamSelection) that would be
    4096 classes off a few hundred training samples — two independent 64-way
    heads need far less data and let TX/RX accuracy be diagnosed separately.

    Three independent input branches, each individually switchable (see
    use_visual/use_geometry/use_localization below) so one class covers the
    2x2 localization ablation instead of four near-duplicate head classes:
      - visual:       global-pooled fused 3D feature (opaque, implicit).
      - geometry:     exact GPS-derived (dx,dy,dz,distance,rel_yaw).
      - localization: TX position *predicted from the occupancy decoder*
                       (soft-argmax over its 'vehicle' class probability,
                       disambiguated between the two candidate neighbor
                       vehicles using the GPS direction as a coarse prior —
                       see loss()'s loss_localization docstring for why this
                       means even use_geometry=False isn't a strict
                       no-GPS-at-all condition).
    """

    def __init__(self,
                 *args,
                 num_tx_beams=64,
                 num_rx_beams=64,
                 geometry_dim=5,
                 geometry_hidden_dim=64,
                 beam_hidden_dim=256,
                 beam_dropout=0.1,
                 use_visual=True,
                 use_geometry=True,
                 use_localization=False,
                 pc_range=None,
                 loc_hidden_dim=32,
                 loc_prior_sigma_xy=8.0,
                 loc_ego_exclude_radius=3.0,
                 loc_loss_weight=1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if not (use_visual or use_geometry or use_localization):
            raise ValueError('BeamSelectionHead needs at least one of '
                              'use_visual/use_geometry/use_localization enabled')
        self.use_visual = use_visual
        self.use_geometry = use_geometry
        self.use_localization = use_localization
        self.loc_loss_weight = loc_loss_weight

        # batch_fuse_features from _encode_and_fuse is a 3D volume
        # [B, C, W, H, Z], not a 2D BEV map -> pool over all three spatial dims.
        trunk_in = 0
        if use_visual:
            self.beam_pool = nn.AdaptiveAvgPool3d(1)
            trunk_in += self.embed_dims[0]
        if use_geometry:
            self.geometry_mlp = nn.Sequential(
                nn.Linear(geometry_dim, geometry_hidden_dim),
                nn.ReLU(inplace=True),
            )
            trunk_in += geometry_hidden_dim
        if use_localization:
            assert pc_range is not None, 'use_localization=True needs pc_range'
            self.pc_range = pc_range
            self.loc_prior_sigma_xy = loc_prior_sigma_xy
            self.loc_ego_exclude_radius = loc_ego_exclude_radius
            self._init_localization_grid()
            self.loc_mlp = nn.Sequential(
                nn.Linear(4, loc_hidden_dim),  # [r, sin(az), cos(az), z]
                nn.ReLU(inplace=True),
            )
            trunk_in += loc_hidden_dim

        self.beam_trunk = nn.Sequential(
            nn.Linear(trunk_in, beam_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(beam_dropout),
        )
        self.tx_beam_head = nn.Linear(beam_hidden_dim, num_tx_beams)
        self.rx_beam_head = nn.Linear(beam_hidden_dim, num_rx_beams)

    def _init_localization_grid(self):
        """Metric-coordinate grid for the coarsest occupancy decoder stage
        (self.deblocks[0]/self.occ[0], stride=1 so same W/H/Z resolution as
        batch_fuse_features itself -- no upsample-stride bookkeeping needed).
        Formula matches Encoder.point_sampling (co3sop_base/modules/encoder.py
        :96-101) exactly -- reusing this codebase's own grid<->metric
        convention rather than a fresh derivation. batch_fuse_features axis
        order is [B, C, W, H, Z] (verified from the .permute(0,4,3,2,1) in
        V2VOccHead._encode_and_fuse), so W maps to X via pc_range[0]/[3],
        H to Y via pc_range[1]/[4], Z to Z via pc_range[2]/[5].
        """
        W, H, Z = self.volume_w[0], self.volume_h[0], self.volume_z[0]
        pc = self.pc_range
        x = (torch.arange(W, dtype=torch.float32) + 0.5) / W * (pc[3] - pc[0]) + pc[0]
        y = (torch.arange(H, dtype=torch.float32) + 0.5) / H * (pc[4] - pc[1]) + pc[1]
        z = (torch.arange(Z, dtype=torch.float32) + 0.5) / Z * (pc[5] - pc[2]) + pc[2]
        ego_mask = ((x.view(W, 1) ** 2 + y.view(1, H) ** 2) >
                    self.loc_ego_exclude_radius ** 2).float()  # [W, H]
        self.register_buffer('loc_x', x, persistent=False)
        self.register_buffer('loc_y', y, persistent=False)
        self.register_buffer('loc_z', z, persistent=False)
        self.register_buffer('loc_ego_mask', ego_mask, persistent=False)

    def _localize_tx(self, fused, img_metas):
        """Soft-argmax TX position from the occupancy decoder's 'vehicle'
        class probability. Returns (loc_embed [B, loc_hidden_dim],
        pred_xyz [B, 3] metric ego-frame -- also used by loss()).
        """
        occ_logits = self.occ[0](self.deblocks[0](fused))  # [B, 3, W, H, Z]
        vehicle_prob = F.softmax(occ_logits, dim=1)[:, VEHICLE_CLASS_IDX]  # [B, W, H, Z]

        anchor_xy = fused.new_tensor(
            np.stack([m['tx_rx_geometry'][:2] for m in img_metas]))  # [B, 2]
        gaussian = torch.exp(-(
            (self.loc_x.view(1, -1, 1, 1) - anchor_xy[:, 0].view(-1, 1, 1, 1)) ** 2 +
            (self.loc_y.view(1, 1, -1, 1) - anchor_xy[:, 1].view(-1, 1, 1, 1)) ** 2
        ) / (2 * self.loc_prior_sigma_xy ** 2))  # [B, W, H, 1] broadcasts to [B,W,H,Z]

        weight = vehicle_prob * gaussian * self.loc_ego_mask.view(1, -1, self.loc_ego_mask.shape[1], 1)
        weight_sum = weight.sum(dim=(1, 2, 3)) + 1e-6  # [B]

        x_pred = (weight * self.loc_x.view(1, -1, 1, 1)).sum(dim=(1, 2, 3)) / weight_sum
        y_pred = (weight * self.loc_y.view(1, 1, -1, 1)).sum(dim=(1, 2, 3)) / weight_sum
        z_pred = (weight * self.loc_z.view(1, 1, 1, -1)).sum(dim=(1, 2, 3)) / weight_sum
        pred_xyz = torch.stack([x_pred, y_pred, z_pred], dim=1)  # [B, 3]

        r = torch.sqrt(x_pred ** 2 + y_pred ** 2 + 1e-6)
        az = torch.atan2(y_pred, x_pred)
        loc_input = torch.stack([r, torch.sin(az), torch.cos(az), z_pred], dim=1)
        return self.loc_mlp(loc_input), pred_xyz

    def forward(self, mcar_feats, img_metas):
        fused, _, _ = self._encode_and_fuse(mcar_feats, img_metas)

        parts = []
        if self.use_visual:
            parts.append(self.beam_pool(fused).flatten(1))
        if self.use_geometry:
            geometry = fused.new_tensor(
                np.stack([m['tx_rx_geometry'] for m in img_metas]))
            parts.append(self.geometry_mlp(geometry))
        pred_xyz = None
        if self.use_localization:
            loc_embed, pred_xyz = self._localize_tx(fused, img_metas)
            parts.append(loc_embed)

        trunk_out = self.beam_trunk(torch.cat(parts, dim=1))
        out = {
            'tx_beam_logits': self.tx_beam_head(trunk_out),
            'rx_beam_logits': self.rx_beam_head(trunk_out),
        }
        if pred_xyz is not None:
            out['loc_pred_xyz'] = pred_xyz
        return out

    @force_fp32(apply_to=('preds_dicts',))
    def loss(self, gt_beam, preds_dicts, img_metas):
        gt_beam = gt_beam.long()
        losses = {
            'loss_tx_beam': F.cross_entropy(preds_dicts['tx_beam_logits'], gt_beam[:, 0]),
            'loss_rx_beam': F.cross_entropy(preds_dicts['rx_beam_logits'], gt_beam[:, 1]),
        }
        if self.use_localization:
            # This is what actually gives self.deblocks[0]/self.occ[0]
            # (loaded from the pretrained occupancy checkpoint, otherwise
            # dead weight -- forward() never called them before this) a
            # gradient signal again: ties their output to correctly locating
            # the TX vehicle, supervised by the same exact GPS-derived
            # (dx,dy,dz) already used as the disambiguation anchor above.
            gt_xyz = preds_dicts['loc_pred_xyz'].new_tensor(
                np.stack([m['tx_rx_geometry'][:3] for m in img_metas]))
            loss_loc = F.smooth_l1_loss(preds_dicts['loc_pred_xyz'], gt_xyz)
            losses['loss_localization'] = loss_loc * self.loc_loss_weight
        return losses
