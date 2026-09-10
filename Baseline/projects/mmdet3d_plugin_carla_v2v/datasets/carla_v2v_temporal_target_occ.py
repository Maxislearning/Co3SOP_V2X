"""Stage 2C: two datasets for the fair C0 (single-frame forecasting)
vs C1 (K=3-frame temporal forecasting) comparison. Both restrict
load_annotations to the SAME matched t-index set (history [t-2,t-1] and
future [t+1] all available) so C0 and C1 train/evaluate on exactly the
same samples and the same t+1 GT -- the only difference is whether C1's
architecture actually gets to use the history frames or not. See plan
effervescent-shimmying-turing.md §3 for the full design rationale (why
this is two separate classes, not one queue_len-parametrized class
pretending to serve both).
"""
from pathlib import Path

import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS

from projects.mmdet3d_plugin.models.utils.transformation_utils import x1_to_x2
from projects.mmdet3d_plugin_carla_v2v.datasets.carla_v2v_beam import RX_VEHICLE
from projects.mmdet3d_plugin_carla_v2v.datasets.carla_v2v_target_occ import CarlaV2VTargetOccCo3SOP

VEHICLES = (1, 2, 3)


def _history_available(by_key, t, k_back, vehicles=VEHICLES):
    """True iff every vehicle's frame exists for t, t-1, ..., t-k_back.
    Stage 2A/2B's single-frame _build_data_info already implicitly assumes
    every used sample_id exists for all 3 vehicles (it calls
    get_vehicle_data on every declared neighbor unconditionally, before any
    distance filtering) -- that assumption held silently for a single t.
    Now that a queue of t-k_back..t is built per sample, it must be checked
    explicitly for every frame in the queue, not just t itself."""
    return all((vn, t - dt) in by_key for vn in vehicles for dt in range(k_back + 1))


@DATASETS.register_module(force=True)
class CarlaV2VForecastCo3SOP(CarlaV2VTargetOccCo3SOP):
    """C0: literally Stage 2B's TargetAwareOccHead/Co3SOPTargetOcc pointed at
    the t+1 forecast GT instead of t's own current-frame GT -- zero
    architecture change (uses the *same* pts_bbox_head/detector classes as
    Stage 2B). Only the GT files loaded differ, plus the sample restriction
    below. This is deliberate (plan §4): C0's entire value is being the
    *exact same* single-frame architecture pointed at the forecasting task,
    so any C1 gain can only be attributed to temporal history, not
    incidental architecture differences.

    target_occ_root (parent kwarg) is deliberately pointed at the future
    target dir (target_occupancy_gt_future_t1/) by the C0 config, not a
    separate kwarg here -- CarlaV2VBeamCo3SOP.get_data_info already loads
    gt_target from target_occ_root/{sid:06d}/{link}.npz unchanged, and
    that's exactly generate_temporal_occupancy_gt.py's own output layout.
    """

    def __init__(self, future_occ_root, *args, **kwargs):
        self.future_occ_root = Path(future_occ_root)
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        infos = super().load_annotations(ann_file)  # CarlaV2VBeamCo3SOP's (scene,'vehicle_2',sid,link) list
        filtered = []
        for entry in infos:
            scene, vehicle, sid_str, link = entry
            t = int(sid_str)
            # k_back=2 even though C0 itself never touches t-1/t-2 -- this is
            # what makes the C0/C1 comparison fair (identical sample set),
            # not a technical requirement of C0's own architecture.
            if not _history_available(self._by_key, t, k_back=2):
                continue
            if not (self.future_occ_root / f'{t:06d}' / 'labels.npz').exists():
                continue
            if not (self.target_occ_root / f'{t:06d}' / f'{link}.npz').exists():
                continue
            filtered.append(entry)
        return filtered

    def get_data_info(self, index):
        data = super().get_data_info(index)  # gt_target from target_occ_root (already future, per config), target_role, tx_rx_geometry, beam_eval_meta
        scene, vehicle, frame_num, link = self.data_infos[index]
        t = int(frame_num)
        # Overrides Stage 2A's current-frame occ_path -- LoadCarlaOccupancy
        # reads 'semantics' from whatever occ_path points at, so this alone
        # redirects gt_occ to the t+1 forecast state.
        data['occ_path'] = str(self.future_occ_root / f'{t:06d}' / 'labels.npz')
        return data


@DATASETS.register_module(force=True)
class CarlaV2VTemporalTargetOccCo3SOP(CarlaV2VTargetOccCo3SOP):
    """C1: camera history [t-2,t-1,t] -> Scene/Target Occupancy at t+1.

    get_data_info returns a dict with a `frames` list (3 raw per-frame data
    dicts, each built by the existing _build_data_info) instead of the flat
    single-frame keys the stock pipeline expects -- so prepare_train_data/
    prepare_test_data are overridden to run self.pipeline once per frame and
    manually assemble the queue-shaped example (mirrors CustomCollect3D's own
    DC conventions by hand: img -> DC(stack=True) of shape [K,N,C,H,W],
    img_metas -> DC([...], cpu_only=True) of length K; batch-collate then
    naturally produces [B,K,N,C,H,W] and a length-B list of length-K lists).
    """

    QUEUE_LEN = 3

    def __init__(self, future_occ_root, future_target_root, *args, **kwargs):
        self.future_occ_root = Path(future_occ_root)
        self.future_target_root = Path(future_target_root)
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        infos = super().load_annotations(ann_file)
        filtered = []
        for entry in infos:
            scene, vehicle, sid_str, link = entry
            t = int(sid_str)
            if not _history_available(self._by_key, t, k_back=self.QUEUE_LEN - 1):
                continue
            if not (self.future_occ_root / f'{t:06d}' / 'labels.npz').exists():
                continue
            if not (self.future_target_root / f'{t:06d}' / f'{link}.npz').exists():
                continue
            filtered.append(entry)
        return filtered

    def get_data_info(self, index):
        scene, vehicle, frame_num, link = self.data_infos[index]
        t = int(frame_num)
        frame_ids = [t - (self.QUEUE_LEN - 1 - k) for k in range(self.QUEUE_LEN)]  # [t-2, t-1, t]

        rx_poses = [self.get_vehicle_data(scene, vehicle, str(fid))['pose'] for fid in frame_ids]
        pose_t = rx_poses[-1]

        frame_dicts = []
        for fid, pose_k in zip(frame_ids, rx_poses):
            fd = self._build_data_info(scene, vehicle, str(fid))
            fd['target_role'] = 0 if link == 'TX_CAR1' else 1
            # Raw (no S-flip) matrix -- S-flip is applied inside
            # TemporalTargetAwareOccHead at forward time, exactly mirroring
            # how _encode_and_fuse already handles trans2ego (also stored
            # raw by _build_data_info, S-flipped only when consumed).
            fd['temporal_trans_to_t'] = np.asarray(x1_to_x2(pose_k, pose_t), dtype=np.float32)
            frame_dicts.append(fd)

        future_scene = np.load(self.future_occ_root / f'{t:06d}' / 'labels.npz')['semantics']
        tgt = np.load(self.future_target_root / f'{t:06d}' / f'{link}.npz')

        return {
            'frames': frame_dicts,
            'gt_occ_future': future_scene,
            'gt_target_future': tgt['target_mask'].astype(np.int64),
            'target_center_raw': tgt['target_center_raw'].astype(np.float32),
        }

    def _prepare_queue(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None

        frame_examples = []
        for fd in input_dict['frames']:
            self.pre_pipeline(fd)
            ex = self.pipeline(fd)
            if ex is None:
                return None
            frame_examples.append(ex)

        img = torch.stack([ex['img'].data for ex in frame_examples], dim=0)  # [K,N,C,H,W]
        img_metas = [ex['img_metas'].data for ex in frame_examples]  # list length K of dicts
        # ex['gt_occ']: current-frame Scene GT at that queue frame, straight
        # off the existing occ_path/LoadCarlaOccupancy path (Stage 2A's
        # already-on-disk files) -- this *is* the aux scene-supervision
        # target, no separate generation needed.
        gt_occ_aux_k = np.stack([np.asarray(ex['gt_occ']) for ex in frame_examples], axis=0)  # [K,200,200,16]

        return {
            'img': DC(img, stack=True),
            'img_metas': DC(img_metas, cpu_only=True),
            'gt_occ_aux_k': gt_occ_aux_k,
            'gt_occ_future': input_dict['gt_occ_future'],
            'gt_target_future': input_dict['gt_target_future'],
            'target_center_raw': input_dict['target_center_raw'],
        }

    def prepare_train_data(self, index):
        return self._prepare_queue(index)

    def prepare_test_data(self, index):
        return self._prepare_queue(index)
