#!/bin/bash
# Full TRACER pipeline: progress labeling -> advantage computation -> training
#
# Prerequisites:
#   - Self-play rollouts completed (run_hgpo_selfplay.sh)
#   - Graph indices available
#
# Usage:
#   bash scripts/run_hgpo_train.sh

set -e

export PYTHONPATH=$PYTHONPATH:$(pwd)

# ── Config ────────────────────────────────────────────────────────────────────
ROLLOUT_DIR='toolplan_data/hgpo/rollouts'
MERGED_FILE="${ROLLOUT_DIR}/all_rollouts.jsonl"
PROGRESS_FILE='toolplan_data/hgpo/progress_labels.jsonl'
ADVANTAGE_FILE='toolplan_data/hgpo/advantages.jsonl'
GRAPH_INDEX_DIR='index_data/SWE-bench_Verified/graph_index_v2.3'
DATASET='princeton-nlp/SWE-bench_Verified'

BASE_MODEL='Qwen/Qwen3-8B'
SFT_ADAPTER='outputs/qwen3-8b-v5-sft-v2/adapter'  # existing SFT adapter
OUTPUT_DIR='outputs'
EXP_NAME='qwen3-8b-hgpo'

# TRACER hyperparameters
HISTORY_LENGTH=2   # K: hierarchical context depth
GAMMA=0.95         # discount factor
ALPHA=1.0          # hierarchy weighting exponent
BETA=1.5           # advantage -> weight scaling
MODE='mean_norm'   # normalization mode

# Training hyperparameters
EPOCHS=2
BATCH_SIZE=1
GRAD_ACCUM=8
LR=5e-5
LORA_R=16
MAX_SEQ=16384

echo "========================================"
echo "  TRACER Training Pipeline"
echo "========================================"

# ── Phase 1: Progress Labeling ────────────────────────────────────────────────
echo ""
echo "=== Phase 1: Labeling trajectories with progress ==="

if [ ! -f "$PROGRESS_FILE" ]; then
    python -m toolplan.data.progress_labeler \
        --traj_file "${MERGED_FILE}" \
        --dataset "${DATASET}" \
        --split test \
        --step_cost_penalty 0.02 \
        --output_file "${PROGRESS_FILE}"
    echo "Progress labels saved to ${PROGRESS_FILE}"
else
    echo "Progress labels already exist at ${PROGRESS_FILE}, skipping."
fi

N_TRAJS=$(wc -l < "${PROGRESS_FILE}")
echo "Total trajectories: ${N_TRAJS}"

# ── Phase 2: TRACER Advantage Computation ──────────────────────────────────────
echo ""
echo "=== Phase 2: Computing TRACER advantages ==="
echo "  history_length=${HISTORY_LENGTH}, gamma=${GAMMA}, alpha=${ALPHA}, beta=${BETA}"

if [ ! -f "$ADVANTAGE_FILE" ]; then
    python -m toolplan.training.tracer_advantage \
        --progress_file "${PROGRESS_FILE}" \
        --graph_index_dir "${GRAPH_INDEX_DIR}" \
        --output_file "${ADVANTAGE_FILE}" \
        --history_length "${HISTORY_LENGTH}" \
        --gamma "${GAMMA}" \
        --step_cost 0.02 \
        --alpha "${ALPHA}" \
        --beta "${BETA}" \
        --mode "${MODE}"
    echo "TRACER advantages saved to ${ADVANTAGE_FILE}"
else
    echo "Advantages already exist at ${ADVANTAGE_FILE}, skipping."
fi

# Print stats
STATS_FILE="${ADVANTAGE_FILE%.jsonl}_stats.json"
if [ -f "$STATS_FILE" ]; then
    echo "Advantage statistics:"
    cat "$STATS_FILE" | python -m json.tool
fi

# ── Phase 3: TRACER Training ──────────────────────────────────────────────────
echo ""
echo "=== Phase 3: TRACER Training ==="
echo "  model=${BASE_MODEL}, lr=${LR}, epochs=${EPOCHS}"
echo "  adapter_from=${SFT_ADAPTER}"

accelerate launch --num_processes 4 \
    -m toolplan.training.tracer_trainer \
    --advantage_file "${ADVANTAGE_FILE}" \
    --base_model "${BASE_MODEL}" \
    --adapter_path "${SFT_ADAPTER}" \
    --max_seq_length "${MAX_SEQ}" \
    --max_tool_output_chars 2000 \
    --output_dir "${OUTPUT_DIR}" \
    --exp_name "${EXP_NAME}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum_steps "${GRAD_ACCUM}" \
    --learning_rate "${LR}" \
    --lora_r "${LORA_R}" \
    --advantage_clip 5.0 \
    --warmup_steps 10

echo ""
echo "========================================"
echo "  TRACER Training Complete!"
echo "  Adapter: ${OUTPUT_DIR}/${EXP_NAME}/adapter"
echo "========================================"
