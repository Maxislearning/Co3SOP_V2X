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


@HEADS.register_module(force=True)
class BeamSelectionHead(V2VOccHead):
    """Two-head (TX beam / RX beam) classification on top of the fused
    ego-centric 3D volume `V2VOccHead._encode_and_fuse` already produces.

    Joint (tx, rx) classification isn't used here: with a 64x64 DFT codebook
    (see carla_V2V/data/script.py RealisticMIMOBeamSelection) that would be
    4096 classes off a few hundred training samples — two independent 64-way
    heads need far less data and let TX/RX accuracy be diagnosed separately.
    """

    def __init__(self,
                 *args,
                 num_tx_beams=64,
                 num_rx_beams=64,
                 geometry_dim=5,
                 geometry_hidden_dim=64,
                 beam_hidden_dim=256,
                 beam_dropout=0.1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # batch_fuse_features from _encode_and_fuse is a 3D volume
        # [B, C, W, H, Z], not a 2D BEV map -> pool over all three spatial dims.
        self.beam_pool = nn.AdaptiveAvgPool3d(1)
        self.geometry_mlp = nn.Sequential(
            nn.Linear(geometry_dim, geometry_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.beam_trunk = nn.Sequential(
            nn.Linear(self.embed_dims[0] + geometry_hidden_dim, beam_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(beam_dropout),
        )
        self.tx_beam_head = nn.Linear(beam_hidden_dim, num_tx_beams)
        self.rx_beam_head = nn.Linear(beam_hidden_dim, num_rx_beams)

    def forward(self, mcar_feats, img_metas):
        fused, _, _ = self._encode_and_fuse(mcar_feats, img_metas)

        visual = self.beam_pool(fused).flatten(1)
        geometry = fused.new_tensor(
            np.stack([m['tx_rx_geometry'] for m in img_metas]))
        geo_embed = self.geometry_mlp(geometry)
        trunk_out = self.beam_trunk(torch.cat([visual, geo_embed], dim=1))

        return {
            'tx_beam_logits': self.tx_beam_head(trunk_out),
            'rx_beam_logits': self.rx_beam_head(trunk_out),
        }

    @force_fp32(apply_to=('preds_dicts',))
    def loss(self, gt_beam, preds_dicts, img_metas):
        gt_beam = gt_beam.long()
        return {
            'loss_tx_beam': F.cross_entropy(preds_dicts['tx_beam_logits'], gt_beam[:, 0]),
            'loss_rx_beam': F.cross_entropy(preds_dicts['rx_beam_logits'], gt_beam[:, 1]),
        }
