export PYTHONPATH=$PYTHONPATH:$(pwd)

# generate graph index for SWE-bench_Lite
python dependency_graph/batch_build_graph.py \
        --dataset 'czlll/SWE-bench_Lite' \
        --split 'test' \
        --repo_path playground/build_graph \
        --num_processes 50 \
        --download_repo

# generate graph index for Loc-Bench
python dependency_graph/batch_build_graph.py \
        --dataset 'czlll/Loc-Bench_V1' \
        --split 'test' \
        --repo_path playground/build_graph \
        --num_processes 50 \
        --download_repo