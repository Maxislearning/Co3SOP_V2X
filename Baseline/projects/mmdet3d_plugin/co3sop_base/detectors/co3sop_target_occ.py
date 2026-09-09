# ---------------------------------------------
# Stage 2B detector: Camera -> Scene Occupancy + Target Occupancy. Own class
# (not branching inside Co3SOPBase/Co3SOPBeam) -- forward_test's evaluation
# is a different kind of thing (dense segmentation scoring) from both voxel
# IoU accumulation and beam top-k accuracy.
# ---------------------------------------------
import torch
from mmdet.models import DETECTORS

from projects.mmdet3d_plugin.co3sop_base.detectors.co3sop_base import Co3SOPBase
from projects.mmdet3d_plugin.datasets.evaluation_metrics import evaluation_semantic


@DETECTORS.register_module(force=True)
class Co3SOPTargetOcc(Co3SOPBase):
    def forward_train(self, img_metas=None, gt_occ=None, gt_target=None, img=None):
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        preds = self.pts_bbox_head(img_feats, img_metas)
        return self.pts_bbox_head.loss(gt_occ, gt_target, preds, img_metas)

    def forward_test(self, img_metas, img=None, gt_occ=None, gt_target=None, **kwargs):
        preds = self.simple_test_pts(
            self.extract_feat(img=img, img_metas=img_metas), img_metas)

        scene_pred = torch.argmax(preds['occ_preds'][-1], dim=1)
        target_pred = torch.argmax(preds['target_preds'][-1], dim=1)

        scene_score = evaluation_semantic(scene_pred, gt_occ, img_metas[0], class_num=3)
        target_score = evaluation_semantic(target_pred, gt_target, img_metas[0], class_num=2)

        # Same {'evaluation': [...]} contract co3sop_beam.py documents --
        # custom_multi_gpu_test does occ_results.extend(result['evaluation']).
        return {'evaluation': [{
            'scene_score': scene_score[0],
            'target_score': target_score[0],
        }]}
