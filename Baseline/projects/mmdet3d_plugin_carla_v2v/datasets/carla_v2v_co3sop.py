"""Adapts Co3SOP's dataset/model wiring to carla_V2V's own data format instead
of OPV2V's yaml+pcd layout — same rationale as CoHFF's carla_v2v_dataset.py:
we have no LiDAR, and camera calibration/occupancy GT is already computed by
carla_V2V/src/{carla_to_flashocc.py,generate_occupancy_gt.py} into
carla_flashocc_infos_*.pkl + occupancy_gt/vehicle_N/{sid}/labels.npz.

Subclasses the real Co3SOP dataset class (not a from-scratch rewrite) so all
the NuScenesDataset/Custom3DDataset boilerplate it relies on (CLASSES,
group-flag, evaluate()) keeps working unchanged; only the three methods that
touch the OPV2V-specific file layout are overridden:
  - load_annotations(): enumerate (vehicle, sample_id) pairs from our infos
    pkl instead of scanning an OPV2V scenario directory tree.
  - get_vehicle_data(): build camera paths + lidar2img from our infos pkl
    (same math as CoHFF's carla_v2v_dataset.py._cav_entry) instead of parsing
    OPV2V yaml. Vehicle pose is read straight from the source CSV in genuine
    CARLA world convention (x, y NOT sign-flipped) because Co3SOP's own
    x1_to_x2()/x_to_world() (reused unchanged from
    models.utils.transformation_utils) assume that convention — the infos
    pkl's ego2global_translation has y negated to align a nuScenes-style
    global frame (see carla_to_flashocc.py), which would silently flip Y in
    every relative cross-agent transform if reused here.
  - get_data_info(): same neighbor-selection/pose-noise/trans2ego logic as
    upstream Co3SOP.get_data_info, just pointing occ_path at our labels.npz
    instead of OPV2V's "{frame}_voxels.npz" additional-annotation layout.

Axis convention: Co3SOP's point_sampling (co3sop_base/modules/encoder.py)
uses pc_range as [xmin,ymin,zmin,xmax,ymax,zmax] with index 0 of each axis =
the MIN bound (increasing towards MAX) — the *same* direction as our own
labels.npz grid (see generate_occupancy_gt.py's mark_pts()). Unlike CoHFF,
no axis flip is needed here as long as occ_size == the native grid shape
(200,200,16) so LoadCarlaOccupancy's center-crop is a no-op.
"""
import os
import pickle

import mmcv
import numpy as np
import pandas as pd
from mmdet.datasets import DATASETS
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.core.bbox import get_box_type

from projects.mmdet3d_plugin.datasets.co3sop import Co3SOP

VEHICLES = (1, 2, 3)


@DATASETS.register_module(force=True)
class CarlaV2VCo3SOP(Co3SOP):
    """carla_V2V has no OPV2V-style ann_file (a real infos pkl) — 'train'/
    'validate' are just split names. The installed mmdet3d's
    Custom3DDataset.__init__ (newer than what Co3SOP was written for) tries
    to open self.ann_file as an actual file via file_client.get_local_path()
    *before* load_annotations() ever runs, which crashes on a bare split
    name. So the whole Co3SOP -> NuScenesDataset -> Custom3DDataset __init__
    chain is bypassed here and reimplemented directly (minus that file-open
    step) rather than patched upstream, keeping the vendored plugin
    untouched.
    """

    def __init__(self,
                 csv_path,
                 occ_size,
                 pc_range,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 file_client_args=dict(backend='disk'),
                 use_semantic=False,
                 overlap_test=False,
                 max_connect_car=0,
                 connect_range=50,
                 pose_noise=None,
                 cam_dirs=None,
                 **kwargs):
        self.csv_path = csv_path
        self.cam_dirs = cam_dirs or [
            'front', 'front_right', 'back_right', 'back', 'back_left', 'front_left'
        ]

        self.occ_size = occ_size
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.CLASSES = self.get_classes(classes)
        self.file_client = mmcv.FileClient(**file_client_args)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}

        self.data_infos = self.load_annotations(self.ann_file)

        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        self.overlap_test = overlap_test
        self.max_connect_car = max_connect_car
        self.pc_range = pc_range
        self.use_semantic = use_semantic
        self.class_names = classes
        self.pose_noise = pose_noise
        self.connect_range = connect_range

        if not self.test_mode:
            self._set_group_flag()

    # ---- helpers -----------------------------------------------------
    @staticmethod
    def _parse_token(token):
        vn_str, sid_str = token.split('_')
        return int(vn_str[1:]), int(sid_str)

    # ---- Co3SOP overrides ---------------------------------------------
    def load_annotations(self, ann_file):
        split = {'train': 'train', 'validate': 'val', 'val': 'val', 'test': 'val'}[ann_file]

        df = pd.read_csv(self.csv_path)
        self._rows = {int(sid): row for sid, row in zip(df['sample_id'], df.to_dict('records'))}

        infos_dir = self.data_root
        self._by_key = {}
        for vn in VEHICLES:
            for sp in ('train', 'val'):
                p = os.path.join(infos_dir, f'carla_flashocc_infos_vehicle_{vn}_{sp}.pkl')
                if not os.path.exists(p):
                    continue
                with open(p, 'rb') as f:
                    d = pickle.load(f)
                for info in d['infos']:
                    self._by_key[(vn, info['frame_idx'])] = info

        with open(os.path.join(infos_dir, f'carla_flashocc_infos_all_{split}.pkl'), 'rb') as f:
            all_data = pickle.load(f)

        scene = 'carla_v2v'
        self.scenes = [scene]
        self.vehicle_infos = {
            scene: {
                "vehicles": [f'vehicle_{vn}' for vn in VEHICLES],
                "frames": [],
            }
        }

        data_infos = []
        for info in all_data['infos']:
            ego_vn, sid = self._parse_token(info['token'])
            data_infos.append((scene, f'vehicle_{ego_vn}', str(sid)))
        return data_infos

    def get_vehicle_data(self, scene, vehicle, frame_num):
        from scipy.spatial.transform import Rotation

        vn = int(vehicle.split('_')[1])
        sid = int(frame_num)
        info = self._by_key[(vn, sid)]
        cams = list(info['cams'].values())

        imgs, intrins, lidar2cams, lidar2imgs = [], [], [], []
        for c in cams:
            qw, qx, qy, qz = c['sensor2ego_rotation']
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t = np.array(c['sensor2ego_translation'], dtype=np.float64)
            cam2ego = np.eye(4)
            cam2ego[:3, :3] = R
            cam2ego[:3, 3] = t
            # lidar frame == ego frame for carla_V2V (lidar2ego is identity,
            # see carla_to_flashocc.py build_info())
            lidar2cam = np.linalg.inv(cam2ego)

            intrinsic = np.array(c['cam_intrinsic'], dtype=np.float64)
            viewpad = np.eye(4)
            viewpad[:3, :3] = intrinsic
            lidar2img = (viewpad @ lidar2cam).astype(np.float32)

            imgs.append(c['data_path'])
            intrins.append(viewpad.astype(np.float32))
            lidar2cams.append(lidar2cam.astype(np.float32))
            lidar2imgs.append(lidar2img)

        row = self._rows[sid]
        # genuine CARLA world pose [x, y, z, roll, yaw, pitch] (degrees) for
        # x1_to_x2()/x_to_world() — NOT the y-negated ego2global from the infos
        # pkl, see module docstring. Flat road -> roll = pitch = 0.
        pose = np.array([
            float(row[f'vehicle_{vn}_x']),
            float(row[f'vehicle_{vn}_y']),
            float(row[f'vehicle_{vn}_z']),
            0.0,
            float(row[f'vehicle_{vn}_yaw']),
            0.0,
        ])

        return {
            "imgs": imgs,
            "vehicle_id": vehicle,
            "intrins": intrins,
            "lidar2img": lidar2imgs,
            "lidar2cams": lidar2cams,
            "pose": pose,
        }

    def get_data_info(self, index):
        scene, vehicle, frame_num = self.data_infos[index]
        return self._build_data_info(scene, vehicle, frame_num)

    def _build_data_info(self, scene, vehicle, frame_num):
        """Body of get_data_info(), keyed explicitly by (scene, vehicle,
        frame_num) instead of a self.data_infos[index] tuple unpack -- so
        CarlaV2VBeamCo3SOP (carla_v2v_beam.py), whose data_infos entries carry
        an extra `link` element, can reuse this without duplicating the
        ego/neighbor-gathering logic."""
        neighbors = [x for x in self.vehicle_infos[scene]["vehicles"] if x != vehicle]

        vn = int(vehicle.split('_')[1])
        sid = int(frame_num)
        ego_occ_info = self._by_key[(vn, sid)]

        data = {}
        data["occ_path"] = os.path.join(ego_occ_info['occ_path'], 'labels.npz')
        data["occ_size"] = self.occ_size
        data["pc_range"] = self.pc_range
        data["img_filename"] = []
        data["lidar2img"] = []
        data["lidar2cams"] = []
        data["cam_intrinsic"] = []
        data["pose"] = []
        data["trans2ego"] = []
        data["vehicle_id"] = []

        from projects.mmdet3d_plugin.models.utils.transformation_utils import cal_dist, x1_to_x2

        ego_info = self.get_vehicle_data(scene, vehicle, frame_num)
        data["img_filename"].extend(ego_info["imgs"])
        data["cam_intrinsic"].extend(ego_info["intrins"])
        data["lidar2img"].extend(ego_info["lidar2img"])
        data["lidar2cams"].extend(ego_info["lidar2cams"])
        data["vehicle_id"].append(ego_info["vehicle_id"])
        data["pose"].append(ego_info["pose"])
        data["trans2ego"].append(np.asarray(x1_to_x2(ego_info["pose"], ego_info["pose"])))

        near_neighbor = []
        for neighbor in neighbors:
            neighbor_info = self.get_vehicle_data(scene, neighbor, frame_num)
            if cal_dist(neighbor_info["pose"], ego_info["pose"]) > self.connect_range:
                continue
            near_neighbor.append(neighbor_info)
        near_neighbor = sorted(near_neighbor, key=lambda n: cal_dist(n["pose"], ego_info["pose"]))

        neighbor_num = 0
        for neighbor_info in near_neighbor:
            if neighbor_num >= self.max_connect_car:
                break
            data["img_filename"].extend(neighbor_info["imgs"])
            data["cam_intrinsic"].extend(neighbor_info["intrins"])
            data["lidar2cams"].extend(neighbor_info["lidar2cams"])
            data["lidar2img"].extend(neighbor_info["lidar2img"])
            data["vehicle_id"].append(neighbor_info["vehicle_id"])
            if self.pose_noise is not None:
                neighbor_info["pose"] = self.add_pose_noise(
                    neighbor_info["pose"],
                    pos_mean=self.pose_noise["pos_mean"],
                    pos_std=self.pose_noise["pos_std"],
                    yaw_std_deg=self.pose_noise["yaw_std_deg"])
            data["pose"].append(neighbor_info["pose"])
            data["trans2ego"].append(np.asarray(x1_to_x2(neighbor_info["pose"], ego_info["pose"])))
            neighbor_num += 1

        for _ in range(self.max_connect_car - neighbor_num):
            data["img_filename"].extend(ego_info["imgs"])
            data["cam_intrinsic"].extend(ego_info["intrins"])
            data["lidar2cams"].extend(ego_info["lidar2cams"])
            data["lidar2img"].extend(ego_info["lidar2img"])
            data["vehicle_id"].append(ego_info["vehicle_id"])
            data["pose"].append(ego_info["pose"])
            data["trans2ego"].append(np.asarray(x1_to_x2(ego_info["pose"], ego_info["pose"])))

        return data
