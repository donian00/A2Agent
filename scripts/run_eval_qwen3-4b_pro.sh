#!/bin/bash
# Eval qwen3-4b-hgpo (adapter pulled from <SERVER>, merged into Qwen3-4B-Instruct-2507)
# on SWE-bench Pro (Python subset, 266 instances).
#
# Pipeline:
#   1. Merge adapter_254 with Qwen3-4B-Instruct-2507 if merged_254/ missing
#   2. Stop any running vLLM on port 8001 (likely 8B from prior run)
#   3. Start vLLM with merged_254 on port 8001 (TP=4, served as qwen3-4b-hgpo)
#   4. Run pro eval with retry loop
#
# Usage: nohup bash scripts/run_eval_qwen3-4b-hgpo_pro.sh > logs/eval_qwen3-4b-hgpo_pro.log 2>&1 &

set -e
cd .
export PYTHONPATH=$PYTHONPATH:$(pwd)

PY='python'
VLLM='vllm'

BASE_MODEL='Qwen/Qwen3-4B-Instruct-2507'
ADAPTER_DIR='outputs/qwen3-4b-hgpo/adapter'
MERGED_DIR='outputs/qwen3-4b-hgpo/merged_254'
SERVED_NAME='qwen3-4b-hgpo'
MODEL_NAME="openai/${SERVED_NAME}"
PORT=8001

RESULT_PATH='./result_path/eval_qwen3-4b-hgpo_pro'
MAX_RETRIES=10
TOTAL_INSTANCES=266

mkdir -p $RESULT_PATH/location logs/vllm

# ---------- Step 1: merge if not already done ----------
if [ ! -f "$MERGED_DIR/config.json" ]; then
    echo "=== [$(date)] Merging $ADAPTER_DIR + $BASE_MODEL -> $MERGED_DIR ==="
    # Need free GPU for merge. If old vLLM still on GPUs, stop it first.
    OLD_VLLM_PIDS=$(pgrep -f "vllm.*--port ${PORT}" || true)
    if [ -n "$OLD_VLLM_PIDS" ]; then
        echo "Stopping existing vLLM (PIDs: $OLD_VLLM_PIDS)"
        kill $OLD_VLLM_PIDS || true
        sleep 15
    fi
    CUDA_VISIBLE_DEVICES=0 $PY scripts/merge_lora.py \
        --base "$BASE_MODEL" \
        --adapter "$ADAPTER_DIR" \
        --out "$MERGED_DIR" \
        --dtype bfloat16 \
        --device cuda
    echo "=== [$(date)] Merge done ==="
else
    echo "=== [$(date)] Skipping merge ($MERGED_DIR already exists) ==="
fi

# ---------- Step 2: stop old vLLM if still alive ----------
OLD_VLLM_PIDS=$(pgrep -f "vllm.*--port ${PORT}" || true)
if [ -n "$OLD_VLLM_PIDS" ]; then
    echo "=== [$(date)] Stopping prior vLLM on port $PORT (PIDs: $OLD_VLLM_PIDS) ==="
    kill $OLD_VLLM_PIDS || true
    sleep 15
fi

# ---------- Step 3: start vLLM with merged 4B ----------
echo "=== [$(date)] Starting vLLM with $MERGED_DIR as $SERVED_NAME on port $PORT ==="
nohup $VLLM serve "$MERGED_DIR" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --max-num-seqs 128 \
    --port $PORT \
    --host 0.0.0.0 \
    --trust-remote-code > logs/vllm/vllm_4b_hgpo.log 2>&1 &
VLLM_PID=$!
echo "vLLM PID=$VLLM_PID"

# Wait until ready (max 5 min)
echo "=== [$(date)] Waiting for vLLM to be ready ==="
for i in $(seq 1 60); do
    if curl -sf "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q "$SERVED_NAME"; then
        echo "=== [$(date)] vLLM READY (waited ${i}*5s) ==="
        break
    fi
    sleep 5
    if [ $i -eq 60 ]; then
        echo "=== [$(date)] ERROR: vLLM did not become ready in 5 min ==="
        tail -30 logs/vllm/vllm_4b_hgpo.log
        exit 1
    fi
done

# ---------- Step 4: run eval ----------
export OPENAI_API_KEY="dummy"
export OPENAI_API_BASE="http://localhost:${PORT}/v1"
export GRAPH_INDEX_DIR='index_data/swe-bench_pro/graph_index_v2.3'
export BM25_INDEX_DIR='index_data/swe-bench_pro/BM25_index'
export SUMMARY_INDEX_DIR='index_data/swe-bench_pro/file_summaries'

echo "=== [$(date)] Starting eval $SERVED_NAME on SWE-bench Pro (Python $TOTAL_INSTANCES) ==="

$PY auto_search_main.py \
    --dataset 'ScaleAI/swe-bench_pro' \
    --split 'test' \
    --model "$MODEL_NAME" \
    --used_list pro_python_ids \
    --localize \
    --merge \
    --output_folder $RESULT_PATH/location \
    --eval_n_limit $TOTAL_INSTANCES \
    --num_processes 10 \
    --timeout 1800 \
    --use_function_calling \
    --simple_desc \
    --enable_commit_search \
    --enable_file_summary \
    --intercept_first_finish \
    --num_samples 1

for i in $(seq 2 $MAX_RETRIES); do
    TOTAL=$(wc -l < $RESULT_PATH/location/loc_outputs.jsonl 2>/dev/null || echo 0)
    EMPTY=$($PY -c "
import json
count = 0
with open('$RESULT_PATH/location/loc_outputs.jsonl') as f:
    for line in f:
        if not line.strip(): continue
        d = json.loads(line)
        if d['found_files'] == [[]]:
            count += 1
print(count)
" 2>/dev/null || echo 0)

    echo "=== [$(date)] Status: total=$TOTAL, empty=$EMPTY ==="

    if [ "$EMPTY" -eq 0 ] && [ "$TOTAL" -ge "$TOTAL_INSTANCES" ]; then
        echo "=== [$(date)] All instances completed ==="
        break
    fi

    echo "=== [$(date)] Run $i/$MAX_RETRIES (retrying empty/missing) ==="
    $PY auto_search_main.py \
        --dataset 'ScaleAI/swe-bench_pro' \
        --split 'test' \
        --model "$MODEL_NAME" \
        --used_list pro_python_ids \
        --localize \
        --merge \
        --output_folder $RESULT_PATH/location \
        --eval_n_limit $TOTAL_INSTANCES \
        --num_processes 10 \
        --timeout 1800 \
        --use_function_calling \
        --simple_desc \
        --enable_commit_search \
        --enable_file_summary \
        --intercept_first_finish \
        --num_samples 1 \
        --rerun_empty_location
done

echo ""
echo "=== [$(date)] Final Results ==="
$PY -c "
import json
total = empty = 0
with open('$RESULT_PATH/location/loc_outputs.jsonl') as f:
    for line in f:
        if not line.strip(): continue
        total += 1
        d = json.loads(line)
        if d['found_files'] == [[]]:
            empty += 1
print(f'Total: {total}, Empty: {empty}, Completed: {total - empty}')
"
echo "=== [$(date)] Done ==="
