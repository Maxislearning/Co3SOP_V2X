"""Beam-selection variant of CarlaV2VCo3SOP: RX is fixed to vehicle_2, TX is
one of TX_CAR1 (vehicle_1) / TX_CAR2 (vehicle_3) -- phase 1 only covers these
two car-mounted links (roadside TX_ROADSIDE_01..05 have no matching camera
view and are left for phase 2, see 记录something/Claude+提示词 doc discussion).

Beam labels come from carla_V2V/src/run_sionna_from_carla_csv.py's output
(channel_summary_from_carla_*.csv): one row per (sample_id, tx_id), with
optimal_tx_beam_idx/optimal_rx_beam_idx from a 64x64 DFT codebook (8x8 planar
arrays, see carla_V2V/data/script.py RealisticMIMOBeamSelection). Every
sample_id here yields 2 dataset items (one per link) sharing the same camera
observation but different TX identity/geometry/label.

CarlaV2VCo3SOP.load_annotations() can't be reused for the train/val split: it
enumerates (vehicle, sample_id) tokens off carla_flashocc_infos_all_{split}.pkl,
whose train/val boundary is a single cut across all 3 vehicles concatenated
in vehicle_1/2/3 order -- for this run that puts *all* of vehicle_2 in the
train half and none in val, which would make a beam val split empty. So this
class does its own chronological 80/20 split over vehicle_2's own sample_ids
instead (same TRAIN_RATIO carla_to_flashocc.py uses per-vehicle).
"""
import numpy as np
import pandas as pd
from mmdet.datasets import DATASETS

from projects.mmdet3d_plugin.models.utils.transformation_utils import cal_dist, x1_to_x2
from projects.mmdet3d_plugin_carla_v2v.datasets.carla_v2v_co3sop import CarlaV2VCo3SOP

RX_VEHICLE = 2
TRAIN_RATIO = 0.8


@DATASETS.register_module(force=True)
class CarlaV2VBeamCo3SOP(CarlaV2VCo3SOP):
    LINK_VEHICLE = {'TX_CAR1': 1, 'TX_CAR2': 3}

    def __init__(self, beam_csv_path, links=('TX_CAR1', 'TX_CAR2'), *args, **kwargs):
        self.beam_csv_path = beam_csv_path
        self.links = links
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        # Runs CarlaV2VCo3SOP.load_annotations purely for its side effects
        # (self._by_key, self._rows, self.vehicle_infos, self.scenes); its
        # returned token list isn't used, see module docstring.
        super().load_annotations(ann_file)

        beam_df = pd.read_csv(self.beam_csv_path)
        beam_df = beam_df[beam_df['tx_name'].isin(self.links)]
        self._beam_by_key = {(int(r.sample_id), r.tx_name): r for r in beam_df.itertuples()}

        vehicle_2_sids = sorted(sid for (vn, sid) in self._by_key if vn == RX_VEHICLE)
        n_train = int(len(vehicle_2_sids) * TRAIN_RATIO)
        split = {'train': 'train', 'validate': 'val', 'val': 'val', 'test': 'val'}[ann_file]
        split_sids = vehicle_2_sids[:n_train] if split == 'train' else vehicle_2_sids[n_train:]

        scene = self.scenes[0]
        infos = []
        for sid in split_sids:
            for link in self.links:
                tx_vn = self.LINK_VEHICLE[link]
                if (sid, link) not in self._beam_by_key:
                    continue
                if (tx_vn, sid) not in self._by_key:
                    # tx vehicle's own image missing for this sample_id
                    # (carla_to_flashocc.py drops samples with missing images
                    # independently per vehicle) -- skip, can't build this link.
                    continue
                infos.append((scene, f'vehicle_{RX_VEHICLE}', str(sid), link))
        return infos

    def get_data_info(self, index):
        scene, vehicle, frame_num, link = self.data_infos[index]
        data = self._build_data_info(scene, vehicle, frame_num)

        row = self._beam_by_key[(int(frame_num), link)]
        data['gt_beam'] = np.array(
            [row.optimal_tx_beam_idx, row.optimal_rx_beam_idx], dtype=np.int64)

        tx_vn = self.LINK_VEHICLE[link]
        tx_pose = self.get_vehicle_data(scene, f'vehicle_{tx_vn}', frame_num)['pose']
        rx_pose = self.get_vehicle_data(scene, vehicle, frame_num)['pose']
        rel = x1_to_x2(tx_pose, rx_pose)  # tx expressed in rx's ego frame
        rel_yaw_deg = tx_pose[4] - rx_pose[4]
        rel_yaw = np.arctan2(np.sin(np.deg2rad(rel_yaw_deg)), np.cos(np.deg2rad(rel_yaw_deg)))
        data['tx_rx_geometry'] = np.array(
            [rel[0, 3], rel[1, 3], rel[2, 3], cal_dist(tx_pose, rx_pose), rel_yaw],
            dtype=np.float32)
        data['beam_eval_meta'] = {'max_transmission_rate': float(row.max_transmission_rate)}
        return data

    def evaluate(self, results, **kwargs):
        results = [r['evaluation'] for r in results]
        tx_gt = np.concatenate([r['tx_gt'] for r in results])
        rx_gt = np.concatenate([r['rx_gt'] for r in results])
        tx_top5 = np.concatenate([r['tx_top5'] for r in results])
        rx_top5 = np.concatenate([r['rx_top5'] for r in results])

        out = {}
        for k in (1, 3, 5):
            out[f'tx_top{k}_acc'] = float((tx_top5[:, :k] == tx_gt[:, None]).any(1).mean())
            out[f'rx_top{k}_acc'] = float((rx_top5[:, :k] == rx_gt[:, None]).any(1).mean())
        out['oracle_rate_mean'] = float(
            np.nanmean(np.concatenate([r['oracle_rate'] for r in results])))
        return out
