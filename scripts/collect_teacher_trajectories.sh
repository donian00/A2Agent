#!/bin/bash
# Generate teacher trajectories on SWE-Gym training set.
#
# Prereq: serve the teacher model with vLLM (OpenAI-compatible endpoint).
# Usage:
#   nohup bash scripts/collect_teacher_trajectories.sh > logs/teacher.log 2>&1 &

export PYTHONPATH=$PYTHONPATH:$(pwd)
export GRAPH_INDEX_DIR='index_data/SWE-Gym/graph_index_v2.3'
export BM25_INDEX_DIR='index_data/SWE-Gym/BM25_index'
export SUMMARY_INDEX_DIR='index_data/SWE-Gym/file_summaries'

DATASET='SWE-Gym/SWE-Gym'
SPLIT='train'
MODEL='openai/qwen3-30b'
NUM_PROCESSES=64
TIMEOUT=300
EVAL_N_LIMIT=2438
STALL_TIMEOUT=2400

OUTPUT_DIR='./toolplan_data/trajectories_swegym_30b/location'
OUTPUT_FILE="${OUTPUT_DIR}/loc_outputs.jsonl"
mkdir -p "${OUTPUT_DIR}" logs

while true; do
    current=$(wc -l < "$OUTPUT_FILE" 2>/dev/null || echo 0)
    if [ "$current" -ge "$EVAL_N_LIMIT" ]; then
        echo "Reached ${current}/${EVAL_N_LIMIT}, exit."
        break
    fi
    echo "=== $(date) === starting pass (progress: ${current}/${EVAL_N_LIMIT})"
    "${PYTHON_BIN:-python}" auto_search_main.py \
        --dataset "${DATASET}" \
        --split "${SPLIT}" \
        --model "${MODEL}" \
        --localize \
        --merge \
        --output_folder "${OUTPUT_DIR}" \
        --eval_n_limit "${EVAL_N_LIMIT}" \
        --num_samples 1 \
        --num_processes "${NUM_PROCESSES}" \
        --timeout "${TIMEOUT}" \
        --use_function_calling \
        --native_tool_calling \
        --enable_commit_search \
        --enable_file_summary \
        --exclude_tools examine_commit &
    main_pid=$!

    last_mod=$(stat -c %Y "$OUTPUT_FILE" 2>/dev/null || echo 0)
    while kill -0 $main_pid 2>/dev/null; do
        sleep 60
        now_mod=$(stat -c %Y "$OUTPUT_FILE" 2>/dev/null || echo 0)
        log_mod=$(stat -c %Y "${OUTPUT_DIR}/localize.log" 2>/dev/null || echo 0)
        now=$(date +%s)
        if [ "$((now - now_mod))" -gt "$STALL_TIMEOUT" ] && \
           [ "$((now - log_mod))" -gt "$STALL_TIMEOUT" ]; then
            echo "=== $(date) === stall detected, restarting..."
            kill -9 -$main_pid 2>/dev/null
            pkill -9 -P $main_pid 2>/dev/null
            sleep 5; break
        fi
        if [ "$now_mod" -ne "$last_mod" ]; then
            last_mod=$now_mod
            new_count=$(wc -l < "$OUTPUT_FILE" 2>/dev/null || echo 0)
            echo "=== $(date) === ${new_count}/${EVAL_N_LIMIT}"
            if [ "$new_count" -ge "$EVAL_N_LIMIT" ]; then
                kill $main_pid 2>/dev/null; break
            fi
        fi
    done
    wait $main_pid 2>/dev/null
    sleep 5
done

echo "=== $(date) === done."
