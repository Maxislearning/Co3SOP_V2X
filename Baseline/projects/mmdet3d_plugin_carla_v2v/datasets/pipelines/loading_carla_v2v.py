import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module(force=True)
class LoadCarlaOccupancy(object):
    """Same as projects.mmdet3d_plugin.datasets.pipelines.loading.LoadOccupancy,
    except carla_V2V's labels.npz stores the dense semantic grid under the key
    'semantics' (Occ3D-nuScenes convention, see generate_occupancy_gt.py) rather
    than 'voxels'. occ_size is set to exactly match the native grid shape
    (200,200,16) in the carla_v2v configs, so the center-crop below is a no-op;
    kept for safety/parity with the original transform.
    """

    def __init__(self, use_semantic=True):
        self.use_semantic = use_semantic

    def __call__(self, results):
        occ_size = results["occ_size"]
        voxels = np.load(results['occ_path'])["semantics"]
        f = int((voxels.shape[0] - occ_size[0]) / 2)
        b = int((voxels.shape[0] + occ_size[0]) / 2)
        l = int((voxels.shape[1] - occ_size[1]) / 2)
        r = int((voxels.shape[1] + occ_size[1]) / 2)
        voxels = voxels[f:b, l:r, :occ_size[2]]
        results['gt_occ'] = voxels
        return results

    def __repr__(self):
        return self.__class__.__name__
