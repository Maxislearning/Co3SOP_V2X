#!/usr/bin/env python3
"""Post-hoc re-ranking with the training-label prior (no retraining).

  per side : p = (1-a) * p_model + a * prior_side
  joint    : p = (1-a) * p_model(tx) * p_model(rx) + a * prior_pair         (mixture)
             s = log p_model(tx) + log p_model(rx) + b * log prior_pair      (log-linear, for comparison)

Priors use TRAIN labels only. a/b are chosen by 2-fold cross-validation on val,
folds split by source run (sample_id // 1000: {0,2,4} vs {1,3,5}) so tuning and
reporting never use the same rows; the reported number pools the two held-out folds.
The objective for choosing a/b is mean Top1..10 accuracy (area under the Top-k curve).

Stage 1 caches val logits per combo to logits_<combo>.npz (reused if present).
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

ROOT = '/home/admin0/Co3SOP/Baseline'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
os.chdir(ROOT)
from full_eval_occ_beam import collect_logits, CHANNELS_TO_CIN  # noqa: E402
from projects.mmdet3d_plugin_carla_v2v.datasets.occ_beam_dataset import OccBeamDataset  # noqa: E402
from projects.mmdet3d_plugin.co3sop_base.dense_heads.occ_beam_encoder import HierarchicalOccupancyBeamEncoder  # noqa: E402
import panel_beamforming as pb  # noqa: E402

OUT = os.path.join(ROOT, 'work_dirs/sweep_v4')
COMBOS = ['gt_scene', 'gt_target', 'gt_scene_target', 'pred_scene', 'pred_target', 'pred_scene_target']
KS = np.arange(1, 11)
C = 64
GRID_A = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
GRID_B = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def labels(split):
    ds = OccBeamDataset(split, 'gt', 'target')
    y = np.array([(int(ds._beam_by_key[k].optimal_tx_beam_idx), int(ds._beam_by_key[k].optimal_rx_beam_idx))
                  for k in ds.infos])
    return y, np.array([sid for sid, _ in ds.infos])


def softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def topk_curve(scores, gt):
    order = np.argsort(-scores, axis=1)[:, :KS.max()]
    hit = order == gt[:, None]
    return np.array([hit[:, :k].any(1).mean() for k in KS])


def cv_select(build_scores, gt, folds, grid):
    """build_scores(param, rows) -> scores; returns pooled held-out curve and chosen params."""
    pooled_hits = np.zeros(len(KS))
    chosen = []
    for tune, test in ((folds, ~folds), (~folds, folds)):
        best = max(grid, key=lambda g: topk_curve(build_scores(g, tune), gt[tune]).mean())
        chosen.append(best)
        pooled_hits += topk_curve(build_scores(best, test), gt[test]) * test.sum()
    return pooled_hits / len(gt), chosen


def main():
    ytr, _ = labels('train')
    yva, sid_va = labels('val')
    folds = np.isin(sid_va // 1000, [0, 2, 4])
    eps = 1.0
    prior_tx = np.bincount(ytr[:, 0], minlength=C) + eps / C
    prior_tx /= prior_tx.sum()
    prior_rx = np.bincount(ytr[:, 1], minlength=C) + eps / C
    prior_rx /= prior_rx.sum()
    pair = np.zeros((C, C)) + eps / (C * C)
    for t, r in ytr:
        pair[t, r] += 1
    prior_pair = (pair / pair.sum()).reshape(-1)
    gt_pair = yva[:, 0] * C + yva[:, 1]

    results = {}
    base_prior = {
        'tx': topk_curve(np.tile(prior_tx, (len(yva), 1)), yva[:, 0]),
        'rx': topk_curve(np.tile(prior_rx, (len(yva), 1)), yva[:, 1]),
        'joint': topk_curve(np.tile(prior_pair, (len(yva), 1)), gt_pair),
    }
    results['prior_only'] = {k: v.tolist() for k, v in base_prior.items()}

    for combo in COMBOS:
        src, ch = combo.split('_', 1)
        cache = os.path.join(OUT, f'logits_{combo}.npz')
        if not os.path.exists(cache):
            model = HierarchicalOccupancyBeamEncoder(in_channels=CHANNELS_TO_CIN[ch],
                                                     num_panels=pb.NUM_PANELS, beams_per_panel=pb.BEAMS_PER_PANEL)
            model.load_state_dict(torch.load(f'work_dirs/occ_beam_{combo}_a2_merged_v4/best.pth', map_location='cpu'))
            model.eval()
            tx_l, rx_l, tx_g, rx_g = collect_logits(model, OccBeamDataset('val', src, ch), 'hierarchical')
            assert (tx_g == yva[:, 0]).all() and (rx_g == yva[:, 1]).all()
            np.savez(cache, tx=tx_l, rx=rx_l)
        z = np.load(cache)
        ptx, prx = softmax(z['tx']), softmax(z['rx'])
        pjoint = (ptx[:, :, None] * prx[:, None, :]).reshape(len(yva), -1)

        r = {'model': {'tx': topk_curve(ptx, yva[:, 0]).tolist(), 'rx': topk_curve(prx, yva[:, 1]).tolist(),
                       'joint': topk_curve(pjoint, gt_pair).tolist()}}
        for side, p, pr, g in (('tx', ptx, prior_tx, yva[:, 0]), ('rx', prx, prior_rx, yva[:, 1])):
            curve, ch_a = cv_select(lambda a, rows: (1 - a) * p[rows] + a * pr, g, folds, GRID_A)
            r.setdefault('mix', {})[side] = curve.tolist()
            r.setdefault('mix_alpha', {})[side] = ch_a
        curve, ch_a = cv_select(lambda a, rows: (1 - a) * pjoint[rows] + a * prior_pair, gt_pair, folds, GRID_A)
        r['mix']['joint'] = curve.tolist()
        r['mix_alpha']['joint'] = ch_a
        logj = np.log(pjoint + 1e-30)
        logpp = np.log(prior_pair)
        curve, ch_b = cv_select(lambda b, rows: logj[rows] + b * logpp, gt_pair, folds, GRID_B)
        r['loglinear_joint'] = curve.tolist()
        r['loglinear_beta'] = ch_b
        results[combo] = r
        print(f'[{combo}] joint Top1/5/10  model {r["model"]["joint"][0]:.3f}/{r["model"]["joint"][4]:.3f}/{r["model"]["joint"][9]:.3f}'
              f'  mix {r["mix"]["joint"][0]:.3f}/{r["mix"]["joint"][4]:.3f}/{r["mix"]["joint"][9]:.3f} (alpha {ch_a})'
              f'  loglin {r["loglinear_joint"][0]:.3f}/{r["loglinear_joint"][4]:.3f}/{r["loglinear_joint"][9]:.3f} (beta {ch_b})', flush=True)
    json.dump(results, open(os.path.join(OUT, 'rerank_prior_results.json'), 'w'), indent=1)
    print('saved rerank_prior_results.json')


if __name__ == '__main__':
    main()
