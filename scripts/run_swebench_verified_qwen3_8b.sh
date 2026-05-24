#!/bin/bash
# Run qwen3-8b-v5-sft-v2 (checkpoint-420) on SWE-bench Verified
# Usage: nohup bash scripts/run_swebench_verified_qwen3_8b.sh > logs/swebench_verified_qwen3_8b.log 2>&1 &

set -e
cd .
export PYTHONPATH=$PYTHONPATH:$(pwd)
export GRAPH_INDEX_DIR='index_data/SWE-bench_Verified/graph_index_v2.3'
export BM25_INDEX_DIR='index_data/SWE-bench_Verified/BM25_index'
export OPENAI_API_KEY="dummy"
export OPENAI_API_BASE="http://localhost:8000/v1"

RESULT_PATH='./result_path/swebench_verified_qwen3_8b_v5_sft_v2'
MAX_RETRIES=10
TOTAL_INSTANCES=500

mkdir -p $RESULT_PATH/location logs

echo "=== [$(date)] Starting qwen3-8b-v5-sft-v2 on SWE-bench Verified ==="

# Initial run
echo "=== [$(date)] Run 1/$MAX_RETRIES (initial) ==="
python auto_search_main.py \
    --dataset 'princeton-nlp/SWE-bench_Verified' \
    --split 'test' \
    --model 'openai/qwen3-8b-sft-v2' \
    --localize \
    --merge \
    --output_folder $RESULT_PATH/location \
    --eval_n_limit $TOTAL_INSTANCES \
    --num_processes 10 \
    --timeout 1800 \
    --use_function_calling \
    --native_tool_calling \
    --simple_desc \
    --enable_commit_search \
    --enable_file_summary \
    --num_samples 1

# Retry loop
for i in $(seq 2 $MAX_RETRIES); do
    TOTAL=$(wc -l < $RESULT_PATH/location/loc_outputs.jsonl 2>/dev/null || echo 0)
    EMPTY=$(python3 -c "
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
        echo "=== [$(date)] All instances completed! ==="
        break
    fi

    echo "=== [$(date)] Run $i/$MAX_RETRIES (retrying empty/missing) ==="
    python auto_search_main.py \
        --dataset 'princeton-nlp/SWE-bench_Verified' \
        --split 'test' \
        --model 'openai/qwen3-8b-sft-v2' \
        --localize \
        --merge \
        --output_folder $RESULT_PATH/location \
        --eval_n_limit $TOTAL_INSTANCES \
        --num_processes 10 \
        --timeout 1800 \
        --use_function_calling \
        --native_tool_calling \
        --simple_desc \
        --enable_commit_search \
        --enable_file_summary \
        --num_samples 1 \
        --rerun_empty_location
done

echo ""
echo "=== [$(date)] Final Results ==="
python3 -c "
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
