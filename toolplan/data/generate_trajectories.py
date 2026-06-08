"""
Rollout trajectory generation using a local vLLM server.

  1. Start vLLM server with the policy model (base or fine-tuned)
  2. Call auto_search_main.localize() with model="hosted_vllm/{model}"
  3. Output: loc_trajs.jsonl

Usage:
    python -m toolplan.data.generate_trajectories \
        --model_path "Qwen/Qwen3-8B" \
        --dataset "princeton-nlp/SWE-bench_Verified" \
        --split "test" \
        --num_samples 5 \
        --output_dir toolplan_data/trajectories
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time

from toolplan.config import SelfPlayConfig, PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def start_vllm_server(
    model_path: str,
    gpus: int = 4,
    port: int = 8000,
    tensor_parallel_size: int = 4,
    max_model_len: int = 32768,
) -> subprocess.Popen:
    """Start a vLLM OpenAI-compatible server.

    Args:
        model_path: HuggingFace model path or local directory
        gpus: Number of GPUs to use
        port: Server port
        tensor_parallel_size: Tensor parallel size for vLLM
        max_model_len: Maximum model context length

    Returns:
        subprocess.Popen: Server process handle
    """
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--max-model-len", str(max_model_len),
        "--trust-remote-code",
        "--dtype", "auto",
    ]

    env = os.environ.copy()
    # Set visible GPUs
    gpu_ids = ",".join(str(i) for i in range(gpus))
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids

    logger.info(f"Starting vLLM server: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    logger.info("Waiting for vLLM server to start...")
    max_wait = 300  # 5 minutes
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            import urllib.request
            req = urllib.request.Request(f"http://localhost:{port}/health")
            urllib.request.urlopen(req, timeout=5)
            logger.info(f"vLLM server ready on port {port}")
            return process
        except Exception:
            if process.poll() is not None:
                stderr = process.stderr.read().decode() if process.stderr else ""
                raise RuntimeError(f"vLLM server exited unexpectedly: {stderr}")
            time.sleep(5)

    stop_vllm_server(process)
    raise TimeoutError(f"vLLM server did not start within {max_wait}s")


def stop_vllm_server(process: subprocess.Popen):
    """Stop a running vLLM server."""
    if process.poll() is None:
        logger.info("Stopping vLLM server...")
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        logger.info("vLLM server stopped.")


def build_args_from_config(
    cfg: SelfPlayConfig,
    model_path: str | None = None,
) -> argparse.Namespace:
    """Build an argparse.Namespace compatible with auto_search_main.localize().

    Args:
        cfg: SelfPlayConfig with generation settings
        model_path: Override model path (e.g., fine-tuned model from previous round)

    Returns:
        argparse.Namespace for auto_search_main
    """
    output_folder = cfg.output_dir
    os.makedirs(output_folder, exist_ok=True)

    # Use hosted_vllm/ prefix so auto_search_main auto-detects CodeAct mode
    actual_model = model_path or cfg.model_path
    model_name = f"hosted_vllm/{actual_model}"

    args = argparse.Namespace(
        # dataset
        dataset=cfg.dataset,
        split=cfg.split,
        eval_n_limit=cfg.eval_n_limit,
        used_list="selected_ids",
        # model — hosted_vllm/ prefix triggers CodeAct conversion in auto_search_main
        model=model_name,
        use_function_calling=True,
        # sampling
        num_samples=cfg.num_samples,
        max_attempt_num=3,
        num_processes=cfg.num_processes,
        # output
        output_folder=output_folder,
        output_file=os.path.join(output_folder, "loc_outputs.jsonl"),
        merge_file=os.path.join(output_folder, "merged_loc_outputs.jsonl"),
        # runtime
        timeout=cfg.timeout,
        log_level="INFO",
        use_example=False,
        ranking_method="mrr",
        # optional features
        enable_commit_search=False,
        enable_file_summary=False,
    )

    return args


def generate_self_play_trajectories(
    cfg: SelfPlayConfig,
    model_path: str | None = None,
) -> str:
    """Generate rollout trajectories with a local vLLM server.

    Args:
        cfg: SelfPlayConfig with generation settings
        model_path: Override model path (for fine-tuned models from later rounds)

    Returns:
        Path to loc_trajs.jsonl
    """
    actual_model = model_path or cfg.model_path

    logger.info(f"Rollout trajectory generation")
    logger.info(f"  Policy model: {actual_model}")
    logger.info(f"  Dataset: {cfg.dataset} ({cfg.split})")
    logger.info(f"  Samples per instance: {cfg.num_samples}")

    # Start vLLM server
    vllm_process = start_vllm_server(
        model_path=actual_model,
        gpus=cfg.vllm_gpus,
        port=cfg.vllm_port,
        tensor_parallel_size=cfg.tensor_parallel_size,
    )

    try:
        # Set OPENAI_API_BASE for litellm to find the vLLM server
        os.environ["OPENAI_API_BASE"] = f"http://localhost:{cfg.vllm_port}/v1"
        traj_file = _generate_standard(cfg, actual_model)
    finally:
        stop_vllm_server(vllm_process)
        os.environ.pop("OPENAI_API_BASE", None)

    return traj_file


def _generate_standard(cfg: SelfPlayConfig, model_path: str) -> str:
    """Standard trajectory generation using auto_search_main.localize()."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from auto_search_main import localize, merge

    args = build_args_from_config(cfg, model_path=model_path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(filename)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(args.output_folder, "self_play.log")),
            logging.StreamHandler(),
        ],
    )

    start = time.time()
    localize(args)
    elapsed = time.time() - start
    logger.info(f"Generation done in {elapsed / 60:.1f} min")

    # Merge results
    merge(args)
    logger.info(f"Merged results saved to {args.merge_file}")

    # Summary stats
    traj_file = os.path.join(args.output_folder, "loc_trajs.jsonl")
    if os.path.exists(traj_file):
        with open(traj_file) as f:
            trajs = [json.loads(line) for line in f]
        total_trajs = sum(len(t.get("loc_trajs", {}).get("trajs", [])) for t in trajs)
        success = sum(1 for t in trajs if t["found_files"] != [[]])
        logger.info(
            f"Summary: {len(trajs)} instances, {total_trajs} trajectories, "
            f"{success} successes ({success/max(len(trajs),1)*100:.1f}%)"
        )

    return traj_file


def main():
    parser = argparse.ArgumentParser(description="Self-play trajectory generation")
    parser.add_argument("--model_path", type=str,
                        default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", type=str,
                        default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--num_processes", type=int, default=10)
    parser.add_argument("--eval_n_limit", type=int, default=0)
    parser.add_argument("--output_dir", type=str,
                        default=str(SelfPlayConfig().output_dir))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--vllm_port", type=int, default=8000)
    parser.add_argument("--vllm_gpus", type=int, default=4)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    args = parser.parse_args()

    cfg = SelfPlayConfig(
        model_path=args.model_path,
        dataset=args.dataset,
        split=args.split,
        eval_n_limit=args.eval_n_limit,
        num_samples=args.num_samples,
        num_processes=args.num_processes,
        temperature=args.temperature,
        output_dir=args.output_dir,
        timeout=args.timeout,
        vllm_port=args.vllm_port,
        vllm_gpus=args.vllm_gpus,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    generate_self_play_trajectories(cfg)


if __name__ == "__main__":
    main()
