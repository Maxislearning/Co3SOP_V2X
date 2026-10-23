#!/usr/bin/env python3
"""任务34: cache val 64-way logits of one checkpoint (same format as sweep_v4/logits_<combo>.npz)."""
import argparse, os, sys
import numpy as np
import torch
ROOT = '/home/admin0/Co3SOP/Baseline'
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tools')); os.chdir(ROOT)
from full_eval_occ_beam import collect_logits, CHANNELS_TO_CIN  # noqa: E402
from projects.mmdet3d_plugin_carla_v2v.datasets.occ_beam_dataset import OccBeamDataset  # noqa: E402
from projects.mmdet3d_plugin.co3sop_base.dense_heads.occ_beam_encoder import HierarchicalOccupancyBeamEncoder  # noqa: E402
import panel_beamforming as pb  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--source', required=True); p.add_argument('--channels', required=True)
p.add_argument('--work-dir', required=True); p.add_argument('--out', required=True)
a = p.parse_args()
model = HierarchicalOccupancyBeamEncoder(in_channels=CHANNELS_TO_CIN[a.channels],
                                         num_panels=pb.NUM_PANELS, beams_per_panel=pb.BEAMS_PER_PANEL)
model.load_state_dict(torch.load(os.path.join(a.work_dir, 'best.pth'), map_location='cpu'))
model.eval()
tx_l, rx_l, _, _ = collect_logits(model, OccBeamDataset('val', a.source, a.channels), 'hierarchical')
np.savez(a.out, tx=tx_l, rx=rx_l)
print('saved', a.out, tx_l.shape)
