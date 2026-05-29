#!/usr/bin/env bash
# R1.4 sweep dispatcher — single-lane runner.
#
# Invocation:   LANE=0 RESULTS_TAG=repro_R1 bash scripts/sweep_repro_R1.sh
#               LANE=1 RESULTS_TAG=repro_R1 bash scripts/sweep_repro_R1.sh
# Each lane pins to one GPU. Two lanes run concurrently → 2-GPU saturation.
#
# Sweep matrix: 4 methods × 2 datasets × 5 seeds = 40 trainings.
# Jobs are ordered seed-major (all 8 method×dataset cells of seed 42 first,
# then seed 43, ...) so the first seed completes a full cell-sweep early
# and gives us early signal across the full method/dataset matrix.
# Lane assignment is index-modulo-2 → each lane gets ~20 trainings.
#
# Training config (matches recipe §7.5 + Tenali 2026 Table I "joint" rows):
#   noise_severity=1     # ↔ paper's λ_train = 0.7
#   test_noise=0.5       # ↔ paper's λ_test = 0.5 (one of 4 columns; other 3
#                        #    eval'd in R1.5b via load_and_eval=True)
#   joint_training=True  # Hydra default; explicit here for the record
#   exp_setup=both_modalities_one_twice_as_severe  # Hydra default; paper-aligned
#   wandb=False + WANDB_MODE=disabled  # avoid the broken-pipe crash from recipe §7.x
#
# Phase R1 — see paper/outline/R1-baseline-reproduction-plan.md.

set -uo pipefail   # no -e; per-job failure shouldn't abort the whole lane
cd "$(dirname "$0")/.."

LANE="${LANE:?need LANE=0 or LANE=1}"
GPU="${GPU:-$LANE}"
RESULTS_TAG="${RESULTS_TAG:-repro_R1}"
DATA_DIR="${DATA_DIR:-/data/csc_datasets}"
EPOCHS="${EPOCHS:-50}"   # Hydra default; verification jobs override to 2

METHODS=(credibility_weighted cs_credibility_weighted einet_dirichlet conditional_einet)
DATASETS=(clean_avmnist nyud2)
SEEDS=(42 43 44 45 46)

# Build full job list in seed-major order (all method×dataset cells per seed).
JOBS=()
for s in "${SEEDS[@]}"; do
  for m in "${METHODS[@]}"; do
    for d in "${DATASETS[@]}"; do
      JOBS+=("${d}|${d}_${m}|${s}")
    done
  done
done

# Partition by lane (index modulo 2).
LANE_JOBS=()
for i in "${!JOBS[@]}"; do
  if (( i % 2 == LANE )); then
    LANE_JOBS+=("${JOBS[i]}")
  fi
done

TOTAL=${#JOBS[@]}
N=${#LANE_JOBS[@]}
echo "[R1.5] lane=$LANE gpu=$GPU tag=$RESULTS_TAG jobs=$N (of $TOTAL total) start=$(date -Iseconds)"

i=0
for job in "${LANE_JOBS[@]}"; do
  i=$((i + 1))
  IFS='|' read -r dataset experiment seed <<< "$job"
  echo "[R1.5] $(date -Iseconds) [$i/$N] START dataset=$dataset experiment=$experiment seed=$seed gpu=$GPU"

  WANDB_MODE=disabled python main.py \
    dataset="$dataset" \
    experiment="$experiment" \
    seed="$seed" \
    gpu="$GPU" \
    noise_severity=1 \
    test_noise=0.5 \
    joint_training=True \
    epochs="$EPOCHS" \
    group_tag="$RESULTS_TAG" \
    wandb=False \
    data_dir="$DATA_DIR"

  rc=$?
  if [ $rc -eq 0 ]; then
    echo "[R1.5] $(date -Iseconds) [$i/$N] OK    dataset=$dataset experiment=$experiment seed=$seed"
  else
    echo "[R1.5] $(date -Iseconds) [$i/$N] FAIL  dataset=$dataset experiment=$experiment seed=$seed rc=$rc"
  fi
done

echo "[R1.5] lane=$LANE complete end=$(date -Iseconds)"
