# ---------------------------------------------
# Stage 2C C1 detector: camera history [t-2,t-1,t] -> Scene/Target
# Occupancy at t+1. Own class (not branching inside Co3SOPTargetOcc) since
# forward_train receives a *queue* of per-frame (img, img_metas) instead of
# one flat pair -- extract_feat is called once per queue frame k, itself
# completely UNCHANGED (Co3SOPBase's own per-frame image backbone/neck, no
# temporal awareness needed there; all temporal-specific work -- alignment,
# fusion -- lives inside TemporalTargetAwareOccHead).
# ---------------------------------------------
import torch
from mmdet.models import DETECTORS

from projects.mmdet3d_plugin.co3sop_base.detectors.co3sop_base import Co3SOPBase
from projects.mmdet3d_plugin.datasets.evaluation_metrics import evaluation_semantic


@DETECTORS.register_module(force=True)
class Co3SOPTemporalTargetOcc(Co3SOPBase):
    def _extract_queue_feats(self, img, img_metas):
        """img: [B,K,N,C,H,W]. img_metas: list length B, each a list length K
        of per-frame dicts -- that's what CarlaV2VTemporalTargetOccCo3SOP's
        cpu_only DataContainer collates into (batch outer, frame inner, same
        as any other cpu_only DC list). Returns (mcar_feats_list,
        img_metas_list), both length K with frame outer / batch inner --
        what TemporalTargetAwareOccHead.forward expects, matching every
        existing single-frame img_metas convention where img_metas[b] is one
        batch item's dict."""
        K = img.shape[1]
        mcar_feats_list, img_metas_list = [], []
        for k in range(K):
            frame_metas = [m[k] for m in img_metas]
            mcar_feats_list.append(self.extract_feat(img=img[:, k], img_metas=frame_metas))
            img_metas_list.append(frame_metas)
        return mcar_feats_list, img_metas_list

    def forward_train(self, img_metas=None, img=None,
                       gt_occ_future=None, gt_target_future=None, gt_occ_aux_k=None):
        mcar_feats_list, img_metas_list = self._extract_queue_feats(img, img_metas)
        preds = self.pts_bbox_head(mcar_feats_list, img_metas_list)
        return self.pts_bbox_head.loss(
            gt_occ_future, gt_target_future, preds, img_metas_list, gt_occ_aux_k=gt_occ_aux_k)

    def forward_test(self, img_metas, img=None,
                      gt_occ_future=None, gt_target_future=None, gt_occ_aux_k=None, **kwargs):
        mcar_feats_list, img_metas_list = self._extract_queue_feats(img, img_metas)
        preds = self.pts_bbox_head(mcar_feats_list, img_metas_list)

        scene_pred = torch.argmax(preds['occ_preds'][-1], dim=1)
        target_pred = torch.argmax(preds['target_preds'][-1], dim=1)

        scene_score = evaluation_semantic(scene_pred, gt_occ_future, img_metas_list[-1][0], class_num=3)
        target_score = evaluation_semantic(target_pred, gt_target_future, img_metas_list[-1][0], class_num=2)

        # Same {'evaluation': [...]} contract co3sop_target_occ.py documents.
        return {'evaluation': [{
            'scene_score': scene_score[0],
            'target_score': target_score[0],
        }]}
