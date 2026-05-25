# A²Agent

### Action-aware Reinforcement Learning for Repository-Level Code Localization

<!-- TODO: replace with the final paper title / links before camera-ready -->

A²Agent trains repository-level **code-localization agents** with **action-aware
Reinforcement Learning**. Instead of assigning a single trajectory-level reward to
every turn (as in GRPO/GSPO), A²Agent estimates a *per-turn* advantage by grouping
turns that share the same recent **action history** (a code-specific *state
proxy*) and normalizing each turn's discounted return within its group. Two
complementary rewards drive learning:

- **Turn-level reward**: immediate credit whenever the agent newly discovers a
  gold location during exploration.
- **Terminal reward**: final-prediction F1 plus a **Discovery-Commit** reward
  that rewards committing already-explored gold locations to the final answer.

The resulting advantages are converted to per-token weights and used in an
**advantage-weighted RL**.

---

## Setup

```bash
conda create -n a2agent python=3.12
conda activate a2agent
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

**Key dependencies** (see `requirements.txt` for the full list):

| Component | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.9.1 (CUDA 12.8) |
| Transformers / PEFT / DeepSpeed | 4.57 / 0.18 / 0.18 |
| vLLM (rollout and serving) | 0.15 |
| Accelerate / TRL / litellm | 1.12 / 0.24 / 1.52 |

**Hardware used:** 4x NVIDIA RTX Pro 6000 (96 GB). SFT/RL use DeepSpeed ZeRO-3
and LoRA; rollout/eval inference is served with vLLM.

---

## Data and Index Preparation

The agent's tools read three indices per repository. Build them once per
benchmark and point to them with environment variables.

**1) Graph index** (directed heterogeneous code graph):

```bash
python dependency_graph/batch_build_graph.py \
    --dataset princeton-nlp/SWE-bench_Verified --split test \
    --repo_path playground/build_graph \
    --num_processes 50 --download_repo
# convenience wrapper: bash scripts/gen_graph_index.sh
```

**2) BM25 index** (keyword code search):

```bash
python build_bm25_index.py \
    --dataset princeton-nlp/SWE-bench_Verified --split test \
    --repo_path playground/build_graph \
    --num_processes 50 --download_repo
# convenience wrapper: bash scripts/gen_bm25_index.sh
```

**3) File-summary index** (used by `SearchSummary` / `ViewSummary`):
the prebuilt summary index is provided under `index_data/<benchmark>/file_summaries`
(generated offline with an OpenAI-compatible LLM). The agent only *reads* it at
inference time; no proprietary model is required to run A²Agent.

**Export index paths** before training/eval:

```bash
export GRAPH_INDEX_DIR='index_data/SWE-bench_Verified/graph_index_v2.3'
export BM25_INDEX_DIR='index_data/SWE-bench_Verified/BM25_index'
export SUMMARY_INDEX_DIR='index_data/SWE-bench_Verified/file_summaries'
```

**Gold localization labels** (for training rewards / evaluation) are parsed from
gold patches and stored under `evaluation/gt_location/<benchmark>/...`. To
(re)generate SWE-Gym training labels:

```bash
python scripts/gen_gt_swegym.py        # parse gold patches into gt_location.jsonl
```

---

## Training

A²Agent training has two stages. All training runs use LoRA (r = alpha = 16) and
DeepSpeed ZeRO-3.

### Stage 1: Teacher SFT warm-up

Collect successful tool-use trajectories from a teacher model on **SWE-Gym**,
then supervise-fine-tune the base policy.

```bash
# (a) serve a teacher model with vLLM, then collect trajectories on SWE-Gym
bash scripts/collect_teacher_trajectories.sh

# (b) supervised fine-tuning
python sft_train.py \
    --model_name Qwen/Qwen3-8B \
    --data-path <sft_data>.jsonl \
    --output-dir outputs --exp-name qwen3-8b-sft \
    --epochs 3 --lora_r 16 --lora_alpha 16 --learning_rate 2e-4
```

### Stage 2: Rollout + Advantage-Weighted RL

```bash
# (a) ROLLOUT: serve the SFT policy with vLLM, then sample N trajectories/issue
python -m toolplan.data.generate_trajectories \
    --model_path <sft-policy> --dataset SWE-Gym --num_samples 8

# (b) TURN-LEVEL REWARD: label each step's progress (delta recall) + step cost
python -m toolplan.data.progress_labeler \
    --traj_file <merged_loc_trajs>.jsonl --dataset SWE-Gym --split train \
    --step_cost_penalty 0.02 --output_file progress_labels.jsonl

# (c) ACTION-LEVEL ADVANTAGE: history-aware grouping + discounted return
python -m toolplan.training.advantage \
    --progress_file progress_labels.jsonl \
    --graph_index_dir $GRAPH_INDEX_DIR \
    --output_file advantages.jsonl \
    --history_length 2 --gamma 0.9 --step_cost 0.02 --beta 1.5 --mode mean_norm

# (d) ADVANTAGE-WEIGHTED SFT
accelerate launch --num_processes 4 \
    -m toolplan.training.trainer \
    --advantage_file advantages.jsonl \
    --base_model Qwen/Qwen3-8B \
    --adapter_path outputs/qwen3-8b-sft/adapter \
    --output_dir outputs --exp_name qwen3-8b-rl \
    --epochs 2 --batch_size 1 --grad_accum_steps 8 \
    --learning_rate 5e-5 --lora_r 16 --advantage_clip 5.0 --warmup_steps 10
```

The full Stage-2 pipeline (b, c, d) is wrapped in
`bash scripts/run_train.sh`. The history depth `K` is set with
`--history_length` (the paper uses `K = 2`).

---

## Serving the Policy

Evaluation and rollout call the policy through an OpenAI-compatible vLLM server.

```bash
# Option A: serve base model + LoRA adapter
vllm serve Qwen/Qwen3-8B \
    --enable-lora --lora-modules qwen3-8b-rl=outputs/qwen3-8b-rl/adapter \
    --served-model-name qwen3-8b-rl --port 8000 --max-model-len 40960

# Option B: merge LoRA then serve the merged model
python scripts/merge_lora.py --base Qwen/Qwen3-8B \
    --adapter outputs/qwen3-8b-rl/adapter --out outputs/qwen3-8b-rl/merged
vllm serve outputs/qwen3-8b-rl/merged --served-model-name qwen3-8b-rl --port 8000

export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="dummy"
```

---

## Evaluation

```bash
python auto_search_main.py \
    --dataset princeton-nlp/SWE-bench_Verified --split test \
    --model openai/qwen3-8b-rl \
    --localize --merge \
    --output_folder result_path/verified/location \
    --eval_n_limit 500 --num_processes 8 --timeout 120 \
    --use_function_calling --simple_desc \
    --enable_commit_search --enable_file_summary \
    --num_samples 1 --rerun_empty_location
# convenience wrapper: bash scripts/run_eval.sh
```

**Metrics**: instance-level Precision / Recall / **F1** at file, module, and
function granularity:

```python
from evaluation.eval_metric import evaluate_results
results = evaluate_results('result_path/verified/location/loc_outputs.jsonl')
```

---

## Key Hyperparameters

| Hyperparameter | Value | | Hyperparameter | Value |
|---|---|---|---|---|
| History depth `H` | 2 | | Discount `gamma` | 0.9 |
| Depth weights `w_k` | proportional to (k+1) | | Step cost `c` | 0.02 |
| Advantage clip `kappa` | 5.0 | | Token-weight temp `beta` | 1.5 |
| Rollouts/issue `N` | 8 | | Max tool calls | 20 |
| LoRA `r`, `alpha` | 16, 16 | | Batch size | 16 |
| LR (SFT / RL) | 2e-4 / 5e-5 | | Optimizer | AdamW (wd 0.01) |

---

## Citation

Anonymized for double-blind review. Citation will be added upon acceptance.

This work builds on the LocAgent code-localization framework and tool
infrastructure (Chen et al., 2025).
