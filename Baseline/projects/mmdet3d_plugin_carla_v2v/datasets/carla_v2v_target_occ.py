"""Stage 2B dataset: same (sample_id, link) indexing / target_occ_root
loading as CarlaV2VBeamCo3SOP (reused unchanged) -- only evaluate() differs,
since this reports dense Scene/Target segmentation scores instead of beam
top-k accuracy. Not a training/beam dataset: gt_beam still gets computed by
the parent's get_data_info() but the Stage 2B pipeline simply doesn't
collect it (no Beam Head/Beam loss this stage).
"""
import numpy as np
from mmdet.datasets import DATASETS

from projects.mmdet3d_plugin_carla_v2v.datasets.carla_v2v_beam import CarlaV2VBeamCo3SOP


@DATASETS.register_module(force=True)
class CarlaV2VTargetOccCo3SOP(CarlaV2VBeamCo3SOP):
    @staticmethod
    def _iou(tp, p, g, eps=1e-5):
        return float(tp / (p + g - tp + eps))

    def evaluate(self, results, **kwargs):
        scene_scores = np.stack([r['scene_score'] for r in results], axis=0).mean(0)  # [3,3]
        target_scores = np.stack([r['target_score'] for r in results], axis=0).mean(0)  # [2,3]

        out = {}
        scene_names = ['empty', 'road', 'vehicle']
        scene_ious = []
        for j, name in enumerate(scene_names):
            tp, p, g = scene_scores[j]
            iou = self._iou(tp, p, g)
            out[f'scene_{name}_iou'] = iou
            scene_ious.append(iou)
        out['scene_mIoU'] = float(np.mean(scene_ious[1:]))  # exclude empty, matches V2VOccHead's own convention

        tp, p, g = target_scores[1]  # class 1 = target
        out['target_iou'] = self._iou(tp, p, g)
        out['target_precision'] = float(tp / (p + 1e-5))
        out['target_recall'] = float(tp / (g + 1e-5))
        out['target_dice'] = float(2 * tp / (p + g + 1e-5))
        return out
