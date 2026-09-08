# ---------------------------------------------
# Beam-selection variant of Co3SOPBase. Kept as its own detector class
# (rather than branching inside Co3SOPBase) so the occupancy training path
# is untouched: forward_test's post-processing is genuinely different in
# kind (top-k beam accuracy vs. voxel IoU accumulation), not just parameters.
# ---------------------------------------------
import numpy as np
from mmdet.models import DETECTORS

from projects.mmdet3d_plugin.co3sop_base.detectors.co3sop_base import Co3SOPBase


@DETECTORS.register_module(force=True)
class Co3SOPBeam(Co3SOPBase):
    def forward_train(self, img_metas=None, gt_beam=None, img=None):
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        preds = self.pts_bbox_head(img_feats, img_metas)
        return self.pts_bbox_head.loss(gt_beam, preds, img_metas)

    def forward_test(self, img_metas, img=None, gt_beam=None, **kwargs):
        preds = self.simple_test_pts(
            self.extract_feat(img=img, img_metas=img_metas), img_metas)

        k = min(5, preds['tx_beam_logits'].shape[1])
        tx_top5 = preds['tx_beam_logits'].topk(k, dim=1).indices
        rx_top5 = preds['rx_beam_logits'].topk(k, dim=1).indices

        # samples_per_gpu=1 throughout this codebase (see co3sop_base_carla_v2v_*
        # configs) -- img_metas is always a single-element list, same convention
        # V2VOccHead._encode_and_fuse and Co3SOPBase.forward_test already rely on.
        oracle_rate = np.array(
            [img_metas[0].get('beam_eval_meta', {}).get('max_transmission_rate', np.nan)])

        return {'evaluation': {
            'tx_gt': gt_beam[:, 0].cpu().numpy(),
            'rx_gt': gt_beam[:, 1].cpu().numpy(),
            'tx_top5': tx_top5.cpu().numpy(),
            'rx_top5': rx_top5.cpu().numpy(),
            'oracle_rate': oracle_rate,
        }}
