"""Central configuration for the ToolPlan framework."""

from dataclasses import dataclass
from pathlib import Path


# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "toolplan_data"
TRAJ_DIR = DATA_DIR / "trajectories"
PROGRESS_DIR = DATA_DIR / "progress_labels"
POLICY_DIR = DATA_DIR / "policy"

GRAPH_INDEX_DIR = PROJECT_ROOT / "index_data"
BM25_INDEX_DIR = PROJECT_ROOT / "index_data"


# ── Rollout Trajectory Generation ────────────────────────────────────────────

@dataclass
class SelfPlayConfig:
    # policy model (local vLLM)
    model_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct"  # or fine-tuned path
    vllm_port: int = 8000
    vllm_gpus: int = 4
    tensor_parallel_size: int = 4

    # generation
    dataset: str = "princeton-nlp/SWE-bench_Verified"
    split: str = "test"
    eval_n_limit: int = 0  # 0 = all
    num_samples: int = 5
    max_iterations: int = 20
    temperature: float = 0.7
    num_processes: int = 10  # conservative for 4x RTX 4090
    timeout: int = 900

    # output
    output_dir: str = str(TRAJ_DIR)


# ── Progress Label ─────────────────────────────────────────────────────────────

@dataclass
class ProgressConfig:
    # which trajectory file to label
    traj_file: str = str(TRAJ_DIR / "loc_trajs.jsonl")

    # gold labels source
    dataset: str = "princeton-nlp/SWE-bench_Verified"
    split: str = "test"

    # labeling params
    entity_level: bool = True  # also compute entity-level progress (not just file)
    step_cost_penalty: float = 0.02  # λ: penalty per step for efficiency

    # output
    output_file: str = str(PROGRESS_DIR / "progress_labels.jsonl")


# ── Advantage ────────────────────────────────────────────────────────────

RL_DIR = DATA_DIR / "rl"


@dataclass
class AdvantageConfig:
    # input
    progress_file: str = str(PROGRESS_DIR / "progress_labels.jsonl")
    graph_index_dir: str = str(GRAPH_INDEX_DIR / "SWE-bench_Verified" / "graph_index_v2.3")

    # parameters
    history_length: int = 2        # max history depth K for hierarchical grouping
    gamma: float = 0.95            # discount factor for step returns
    step_cost: float = 0.02       # per-step cost penalty in reward
    alpha: float = 1.0            # weighting exponent for hierarchy levels
    mode: str = "mean_norm"       # "mean_norm" or "mean_std_norm"
    use_episode_base: bool = True  # include episode-level advantage as base group

    # advantage -> weight conversion
    beta: float = 1.5             # scaling factor
    weight_min: float = 0.1
    weight_max: float = 3.0

    # output
    output_file: str = str(RL_DIR / "advantages.jsonl")


# ── Training ────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # model
    base_model: str = "Qwen/Qwen3-8B"
    adapter_path: str = ""  # SFT adapter to resume from (optional)
    max_seq_length: int = 16384

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 16

    # data
    advantage_file: str = str(RL_DIR / "advantages.jsonl")
    max_tool_output_chars: int = 2000

    # loss
    clip_eps: float = 0.2         # PPO clipping epsilon
    kl_coef: float = 0.05         # KL regularization coefficient
    advantage_clip: float = 5.0   # clip extreme advantages

    # training
    epochs: int = 2
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    warmup_steps: int = 10
    weight_decay: float = 0.01

    # output
    output_dir: str = str(POLICY_DIR)
    exp_name: str = "qwen3-8b-rl"


