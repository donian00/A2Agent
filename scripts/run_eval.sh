#!/bin/bash
# Run code-localization evaluation on a benchmark.
#
# Prerequisites:
#   1. Build the graph / BM25 indices for the target benchmark (see README).
#   2. Serve the trained policy with vLLM and export the OpenAI-compatible
#      endpoint (see the Serving section in the README):
#        export OPENAI_API_BASE="http://<host>:<port>/v1"
#        export OPENAI_API_KEY="dummy"
#
# Usage:
#   bash scripts/run_eval.sh

set -e
export PYTHONPATH=$PYTHONPATH:$(pwd)

# ── Config ──────────────────────────────────────────────────────────────────
BENCH='SWE-bench_Verified'                       # or 'SWE-bench_Pro'
DATASET='princeton-nlp/SWE-bench_Verified'       # HuggingFace dataset id
MODEL='openai/qwen3-8b-rl'                        # served vLLM model name
RESULT_PATH='./result_path/eval'
N_LIMIT=500
NUM_PROCESSES=8

export GRAPH_INDEX_DIR="index_data/${BENCH}/graph_index_v2.3"
export BM25_INDEX_DIR="index_data/${BENCH}/BM25_index"

mkdir -p "$RESULT_PATH/location" logs

python auto_search_main.py \
    --dataset "$DATASET" \
    --split 'test' \
    --model "$MODEL" \
    --localize \
    --merge \
    --output_folder "$RESULT_PATH/location" \
    --eval_n_limit "$N_LIMIT" \
    --num_processes "$NUM_PROCESSES" \
    --timeout 600 \
    --use_function_calling \
    --enable_commit_search \
    --enable_file_summary \
    --num_samples 1 \
    --temperature 0.0 \
    --seed 42

echo "=== Done. Results in ${RESULT_PATH}/location/loc_outputs.jsonl ==="
