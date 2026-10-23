#!/bin/bash
# 任务34 用法: tools/sweep_v4_seed_lane.sh <source> "<seeds>"   多种子重复v4 A2 sweep（配置同 sweep_v4_run_lane.sh），可断点续跑
source /home/admin0/anaconda3/etc/profile.d/conda.sh; conda activate flashocc
cd /home/admin0/Co3SOP/Baseline
S=$1; SEEDS=${2:-"1 2"}
for SEED in $SEEDS; do
for C in scene target scene_target; do
  W=work_dirs/occ_beam_${S}_${C}_a2_merged_v4_s${SEED}
  L=work_dirs/sweep_v4/logits_${S}_${C}_s${SEED}.npz
  if [ -f $L ]; then echo "[seed] skip $S/$C/s$SEED (done)"; continue; fi
  mkdir -p $W
  echo "[seed] $(date +%F_%T) train $S/$C/s$SEED"
  python tools/train_occ_beam.py --source $S --channels $C --architecture hierarchical --loss-weighting sqrt_inv_freq \
      --epochs 30 --patience 10 --seed $SEED --work-dir $W > $W.train.log 2>&1 || { echo "[seed] TRAIN FAILED $S/$C/s$SEED"; continue; }
  python tools/full_eval_occ_beam.py --source $S --channels $C --architecture hierarchical --work-dir $W > $W/eval.log 2>&1 \
      || echo "[seed] EVAL FAILED $S/$C/s$SEED"
  python tools/dump_val_logits.py --source $S --channels $C --work-dir $W --out $L > $W/dump_logits.log 2>&1 \
      && echo "[seed] $(date +%F_%T) done $S/$C/s$SEED" || echo "[seed] LOGITS FAILED $S/$C/s$SEED"
done
done
echo "[seed] lane $S ALL DONE"
