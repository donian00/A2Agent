#!/bin/bash
# Eval qwen3-8b-hgpo (adapter pulled from <SERVER>, merged into Qwen3-8B)
# on SWE-bench Pro (Python subset, 266 instances)
# Usage: nohup bash scripts/run_eval_qwen3-8b-hgpo_pro.sh > logs/eval_qwen3-8b-hgpo_pro.log 2>&1 &

set -e
cd .
export PYTHONPATH=$PYTHONPATH:$(pwd)
export GRAPH_INDEX_DIR='index_data/swe-bench_pro/graph_index_v2.3'
export BM25_INDEX_DIR='index_data/swe-bench_pro/BM25_index'
export SUMMARY_INDEX_DIR='index_data/swe-bench_pro/file_summaries'
export OPENAI_API_KEY="dummy"
export OPENAI_API_BASE="http://localhost:8001/v1"

RESULT_PATH='./result_path/eval_qwen3-8b-hgpo_pro'
MAX_RETRIES=10
TOTAL_INSTANCES=266
MODEL_NAME='openai/qwen3-8b-hgpo'
PY='python'

mkdir -p $RESULT_PATH/location logs

echo "=== [$(date)] Starting eval qwen3-8b-hgpo on SWE-bench Pro (Python 266) ==="

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
