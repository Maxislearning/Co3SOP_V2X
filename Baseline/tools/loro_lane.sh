#!/bin/bash
# 任务35 用法: tools/loro_lane.sh <gt|pred> "<seeds>" ["<extra train args>"] [tag]
# 留一个run交叉验证：对每个 seed × 组合(target, scene_target) × 折 训练 A2，并导出测试 run 的 logits；可断点续跑。
# 折的划分文件来自 tools/loro_cv.py（work_dirs/loro_cv/splits）。环境变量 FOLDS 可以只跑部分折（例：FOLDS=seed5）。
source /home/admin0/anaconda3/etc/profile.d/conda.sh; conda activate flashocc
cd /home/admin0/Co3SOP/Baseline
S=$1; SEEDS=${2:-"0 1 2"}; EXTRA=${3:-""}; TAG=${4:-main}
SPLITS=${SPLITS:-work_dirs/loro_cv/splits};  # 任务48：v5 用 SPLITS=work_dirs/loro_cv/splits_v5
OUT=work_dirs/loro_cv/$TAG; mkdir -p $OUT/logits
DUMP_EXTRA=$(grep -o -- "--coord-conv\( [a-z]*\)\?\|--occ-dir [^ ]*" <<< "$EXTRA" | tr "\n" " ")
for SEED in $SEEDS; do
for C in target scene_target; do
for F in ${FOLDS:-legacy seed1 seed2 seed3 seed4 seed5}; do
  W=$OUT/runs/${S}_${C}_${F}_s${SEED}
  L=$OUT/logits/${S}_${C}_${F}_s${SEED}.npz
  if [ -f $L ]; then echo "[loro] skip $S/$C/$F/s$SEED (done)"; continue; fi
  mkdir -p $W
  echo "[loro] $(date +%F_%T) train $S/$C/$F/s$SEED"
  python tools/train_occ_beam.py --source $S --channels $C --architecture hierarchical --loss-weighting sqrt_inv_freq \
      --epochs 30 --patience 10 --seed $SEED --num-workers 2 --split-file $SPLITS/fold_$F.json $EXTRA \
      --work-dir $W > $W/train.log 2>&1 || { echo "[loro] TRAIN FAILED $S/$C/$F/s$SEED"; continue; }
  python tools/dump_val_logits.py --source $S --channels $C --work-dir $W --out $L $DUMP_EXTRA \
      --split-file $SPLITS/fold_$F.json --split test > $W/dump_logits.log 2>&1 \
      && echo "[loro] $(date +%F_%T) done $S/$C/$F/s$SEED" || echo "[loro] LOGITS FAILED $S/$C/$F/s$SEED"
done
done
done
echo "[loro] lane $S $TAG ALL DONE"
