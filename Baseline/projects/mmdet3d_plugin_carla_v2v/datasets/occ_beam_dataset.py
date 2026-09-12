"""Stage 3 Step 2: OccBeamDataset -- plain torch.utils.data.Dataset (NOT
mmdet3d-registered; this never touches a camera image, the per-car
backbone, or Co3SOP's cross-agent fusion, so there's nothing the
Config/Runner/DATASETS registry machinery would buy here). Loads Step 1's
already-canonicalized GT/Predicted Scene+Target volumes plus the beam
labels, for one of the B0-B3 ablations.

Train/val split is NEVER recomputed here -- it locks onto Stage 2B's own
actual (sample_id,link) sets (Audit 1c) by building CarlaV2VTargetOccCo3SOP
directly (same object tools/extract_stage2b_occ_probs.py already validated
against), reading its .data_infos. No `tx_rx_geometry`, no pose, no RX yaw
anywhere in this file -- RX yaw was fully consumed inside Step 1's offline
canonicalization.
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CANON_DIR = '/home/admin0/carla_V2V/occupancy_prediction_beam/beam_occ_canonical'
BEAM_CSV = '/home/admin0/carla_V2V/channel_analysis/channel_summary_from_carla_20260904_145458.csv'
CONFIG_FOR_SPLIT = 'projects/configs/co3sop_base/co3sop_base_carla_v2v_target_occ.py'


def _build_split_infos(split):
    """Returns the exact list of (sample_id, link) CarlaV2VTargetOccCo3SOP
    uses for this split -- imported lazily (only needs mmdet3d once, at
    dataset-construction time, not per __getitem__) to keep this module
    importable without the full mmdet3d_plugin registration if ever needed
    standalone."""
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    import projects.mmdet3d_plugin_carla_v2v  # noqa: F401 registers the dataset class

    cfg = Config.fromfile(CONFIG_FOR_SPLIT)
    ds = build_dataset(cfg.data.train if split == 'train' else cfg.data.val)
    return [(int(info[2]), info[3]) for info in ds.data_infos]


class OccBeamDataset(Dataset):
    CHANNEL_SPECS = {
        'scene': ('scene',),
        'target': ('target',),
        'scene_target': ('scene', 'target'),
    }

    def __init__(self, split, source, channels, canon_dir=CANON_DIR, beam_csv_path=BEAM_CSV):
        assert split in ('train', 'val')
        assert source in ('gt', 'pred')
        assert channels in self.CHANNEL_SPECS, f'unknown channels={channels!r}'
        self.split = split
        self.source = source
        self.channels = channels
        self.canon_dir = canon_dir

        self.infos = _build_split_infos(split)  # [(sample_id, link), ...] -- locked to Stage 2B's own split

        beam_df = pd.read_csv(beam_csv_path)
        beam_df = beam_df[beam_df['tx_name'].isin(['TX_CAR1', 'TX_CAR2'])]
        self._beam_by_key = {
            (int(r.sample_id), r.tx_name): r for r in beam_df.itertuples(index=False)
        }
        # every (sample_id,link) this dataset will serve must have a beam label
        missing = [k for k in self.infos if k not in self._beam_by_key]
        assert not missing, f'{len(missing)} (sample_id,link) pairs have no beam label: {missing[:5]}...'

    def __len__(self):
        return len(self.infos)

    def _load_scene(self, sid):
        key = 'onehot' if self.source == 'gt' else 'probs'
        fname = 'gt_scene.npz' if self.source == 'gt' else 'pred_scene.npz'
        arr = np.load(os.path.join(self.canon_dir, f'{sid:06d}', fname))[key].astype(np.float32)
        return arr  # [3,X,Y,Z]

    def _load_target(self, sid, link):
        key = 'mask' if self.source == 'gt' else 'prob'
        fname = f'{link}_gt_target.npz' if self.source == 'gt' else f'{link}_pred_target.npz'
        arr = np.load(os.path.join(self.canon_dir, f'{sid:06d}', fname))[key].astype(np.float32)
        return arr  # [1,X,Y,Z]

    def __getitem__(self, idx):
        sid, link = self.infos[idx]
        parts = []
        if 'scene' in self.CHANNEL_SPECS[self.channels]:
            parts.append(self._load_scene(sid))
        if 'target' in self.CHANNEL_SPECS[self.channels]:
            parts.append(self._load_target(sid, link))
        volume = np.concatenate(parts, axis=0)  # [C,X,Y,Z], X=200,Y=200,Z=16

        # Disk layout is [C,X,Y,Z] (generate_occupancy_gt.py's own grid[NX,NY,NZ]
        # convention). PyTorch Conv3d wants [C,D,H,W] -- LOCKED convention
        # (per review): D=Z, H=Y, W=X. Permute once, here, explicitly
        # asserted -- never re-permuted anywhere downstream.
        volume = np.transpose(volume, (0, 3, 2, 1))  # [C,X,Y,Z] -> [C,Z,Y,X]
        assert volume.shape[-3:] == (16, 200, 200), f'unexpected shape after permute: {volume.shape}'

        row = self._beam_by_key[(sid, link)]
        tx_label = int(row.optimal_tx_beam_idx)
        rx_label = int(row.optimal_rx_beam_idx)

        return (
            torch.from_numpy(volume).float(),
            torch.tensor(tx_label, dtype=torch.long),
            torch.tensor(rx_label, dtype=torch.long),
            {'sample_id': sid, 'link': link},
        )
