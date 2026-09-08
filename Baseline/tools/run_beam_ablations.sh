#!/usr/bin/env bash
# Runs the 2x2 TX-localization ablation (see
# /home/admin0/.claude/plans/effervescent-shimmying-turing.md) sequentially
# on one GPU, stopping on first failure rather than silently burning hours
# on a broken later config. Each cell writes to its own work_dirs/ so they
# never clobber each other or the existing no-localization baseline
# (work_dirs/carla_v2v_beam).
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG_DIR=projects/configs/co3sop_base
CELLS=(geo_vis geo_novis nogeo_vis nogeo_novis)

for cell in "${CELLS[@]}"; do
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] starting ablation cell: $cell ==="
    bash tools/dist_train.sh \
        "${CONFIG_DIR}/co3sop_base_carla_v2v_beam_ablation_${cell}.py" \
        1 \
        "work_dirs/carla_v2v_beam_ablation_${cell}"
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] finished ablation cell: $cell ==="
done

echo "=== all 4 ablation cells finished ==="
