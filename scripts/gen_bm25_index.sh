export PYTHONPATH=$PYTHONPATH:$(pwd)

python build_bm25_index.py \
        --dataset 'czlll/Loc-Bench_V1' \
        --split 'test' \
        --repo_path playground/build_graph \
        --num_processes 100 \
        --download_repo

python build_bm25_index.py \
        --dataset 'czlll/SWE-bench_Lite' \
        --split 'test' \
        --repo_path playground/build_graph \
        --num_processes 100 \
        --download_repo