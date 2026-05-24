#!/bin/bash
# Build graph + BM25 + file_summary indices for SWE-bench Pro (Python subset, 266 instances, 3 repos)
# Usage: nohup bash scripts/build_pro_python_index.sh > logs/build_pro_python_index.log 2>&1 &

set -e
cd .
export PYTHONPATH=$PYTHONPATH:$(pwd)

PY='python'
DATASET='ScaleAI/swe-bench_pro'
SPLIT='test'
INSTANCE_IDS='scripts/pro_python_ids.json'
REPO_PATH='playground/build_graph'
INDEX_BASE='index_data'
NUM_PROCESSES=20

mkdir -p logs

echo "=== [$(date)] Step 1/3: Graph index ==="
$PY dependency_graph/batch_build_graph.py \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --instance_id_path "$INSTANCE_IDS" \
    --repo_path "$REPO_PATH" \
    --index_dir "$INDEX_BASE" \
    --num_processes $NUM_PROCESSES \
    --download_repo

echo ""
echo "=== [$(date)] Step 2/3: BM25 index ==="
$PY build_bm25_index.py \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --instance_id_path "$INSTANCE_IDS" \
    --repo_path "$REPO_PATH" \
    --index_dir "$INDEX_BASE" \
    --num_processes $NUM_PROCESSES \
    --download_repo

echo ""
echo "=== [$(date)] Step 3/3: File summaries (OpenAI Batch API) ==="
$PY -m toolplan.data.generate_file_summaries \
    --index_dir "${INDEX_BASE}/swe-bench_pro" \
    --output_dir "${INDEX_BASE}/swe-bench_pro/file_summaries"

echo ""
echo "=== [$(date)] Done. Submit batch; check back with --check_batch ==="
