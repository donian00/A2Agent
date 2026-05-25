"""
Evaluate N-rollout results.

Per-instance metrics averaged across rollouts, plus pass@k and best-of-N.

Usage:
    python -m evaluation.eval_rollouts \
        --rollout_dir toolplan_data/rl/rollouts \
        --n_rollouts 8 \
        --dataset princeton-nlp/SWE-bench_Verified

    # Or with a merged file:
    python -m evaluation.eval_rollouts \
        --merged_file toolplan_data/rl/rollouts/all_rollouts.jsonl \
        --dataset princeton-nlp/SWE-bench_Verified
"""

import argparse
import collections
import json
import logging
import os
import re
from typing import Optional

import numpy as np
import pandas as pd
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def extract_gold_files_from_patch(patch: str) -> list[str]:
    """Extract modified file paths from a unified diff patch."""
    files = []
    for line in patch.split("\n"):
        if line.startswith("diff --git"):
            match = re.search(r"b/(.+)$", line)
            if match:
                fp = match.group(1)
                if fp not in files:
                    files.append(fp)
        elif line.startswith("+++ b/"):
            fp = line[6:].strip()
            if fp and fp != "/dev/null" and fp not in files:
                files.append(fp)
    return files


def build_gold_map(dataset_name: str, split: str) -> dict[str, list[str]]:
    """Build instance_id -> gold_files mapping."""
    ds = load_dataset(dataset_name, split=split)
    gold_map = {}
    for instance in ds:
        instance_id = instance["instance_id"]
        patch = instance.get("patch", "")
        gold_files = extract_gold_files_from_patch(patch)
        gold_map[instance_id] = gold_files
    return gold_map


def load_rollouts(
    rollout_dir: Optional[str] = None,
    merged_file: Optional[str] = None,
    n_rollouts: int = 8,
) -> dict[str, list[dict]]:
    """Load rollouts grouped by instance_id.

    Returns: {instance_id: [rollout_0, rollout_1, ..., rollout_N-1]}
    """
    all_entries = []

    if merged_file and os.path.exists(merged_file):
        with open(merged_file) as f:
            for line in f:
                all_entries.append(json.loads(line))
    elif rollout_dir:
        for i in range(n_rollouts):
            output_file = os.path.join(rollout_dir, f"rollout_{i}", "location", "loc_outputs.jsonl")
            if not os.path.exists(output_file):
                logger.warning(f"Rollout {i} not found: {output_file}")
                continue
            with open(output_file) as f:
                for line in f:
                    entry = json.loads(line)
                    entry["rollout_idx"] = i
                    all_entries.append(entry)

    # Group by instance_id
    by_instance = collections.defaultdict(list)
    for entry in all_entries:
        by_instance[entry["instance_id"]].append(entry)

    return dict(by_instance)


def compute_instance_recall(pred_files: list[str], gold_files: list[str]) -> float:
    """File-level recall for a single prediction."""
    if not gold_files:
        return 0.0
    pred_set = set(pred_files)
    gold_set = set(gold_files)
    return len(pred_set & gold_set) / len(gold_set)


def compute_instance_hit(pred_files: list[str], gold_files: list[str]) -> bool:
    """Whether at least one gold file was found."""
    return bool(set(pred_files) & set(gold_files))


def evaluate_rollouts(
    rollouts_by_instance: dict[str, list[dict]],
    gold_map: dict[str, list[str]],
) -> dict:
    """Compute per-instance and aggregate metrics.

    Returns dict with:
        - per_instance: {instance_id: {avg_recall, best_recall, pass_at_1, pass_at_k, n_rollouts, ...}}
        - aggregate: {avg_recall_mean, best_of_n_recall, pass_at_1, pass_at_k, ...}
    """
    per_instance = {}

    for instance_id, rollouts in rollouts_by_instance.items():
        gold_files = gold_map.get(instance_id, [])
        if not gold_files:
            continue

        recalls = []
        hits = []
        n_steps_list = []

        for rollout in rollouts:
            found_files = rollout.get("found_files", [])
            if isinstance(found_files, list) and found_files and isinstance(found_files[0], list):
                pred_files = found_files[0]
            elif isinstance(found_files, list):
                pred_files = found_files
            else:
                pred_files = []

            recall = compute_instance_recall(pred_files, gold_files)
            hit = compute_instance_hit(pred_files, gold_files)
            recalls.append(recall)
            hits.append(hit)

            # Count steps if trajectory data available
            trajs = rollout.get("loc_trajs", {}).get("trajs", [])
            if trajs:
                n_msgs = len(trajs[0].get("messages", []))
                n_steps_list.append(n_msgs // 2)  # approximate: each step = assistant + tool

        n = len(recalls)
        per_instance[instance_id] = {
            "n_rollouts": n,
            "recalls": recalls,
            "avg_recall": float(np.mean(recalls)),
            "std_recall": float(np.std(recalls)),
            "best_recall": float(np.max(recalls)),
            "worst_recall": float(np.min(recalls)),
            "pass_at_1": float(hits[0]) if hits else 0.0,  # first rollout
            "pass_at_k": float(any(hits)),  # at least one hit across all rollouts
            "n_hits": sum(hits),
            "hit_rate": sum(hits) / max(n, 1),
            "avg_steps": float(np.mean(n_steps_list)) if n_steps_list else 0.0,
            "reward_variance": float(np.var(recalls)),
        }

    # Aggregate
    all_avg_recalls = [v["avg_recall"] for v in per_instance.values()]
    all_best_recalls = [v["best_recall"] for v in per_instance.values()]
    all_pass_at_1 = [v["pass_at_1"] for v in per_instance.values()]
    all_pass_at_k = [v["pass_at_k"] for v in per_instance.values()]
    all_hit_rates = [v["hit_rate"] for v in per_instance.values()]
    all_variances = [v["reward_variance"] for v in per_instance.values()]

    n_instances = len(per_instance)
    aggregate = {
        "n_instances": n_instances,
        "n_total_rollouts": sum(v["n_rollouts"] for v in per_instance.values()),
        "avg_rollouts_per_instance": sum(v["n_rollouts"] for v in per_instance.values()) / max(n_instances, 1),
        # Recall metrics
        "avg_recall_mean": float(np.mean(all_avg_recalls)) if all_avg_recalls else 0.0,
        "avg_recall_std": float(np.std(all_avg_recalls)) if all_avg_recalls else 0.0,
        "best_of_n_recall": float(np.mean(all_best_recalls)) if all_best_recalls else 0.0,
        # Pass@k
        "pass_at_1": float(np.mean(all_pass_at_1)) if all_pass_at_1 else 0.0,
        "pass_at_k": float(np.mean(all_pass_at_k)) if all_pass_at_k else 0.0,
        # Hit analysis
        "avg_hit_rate": float(np.mean(all_hit_rates)) if all_hit_rates else 0.0,
        # Variance (need variance for advantage estimation)
        "avg_reward_variance": float(np.mean(all_variances)) if all_variances else 0.0,
        "n_with_variance": sum(1 for v in all_variances if v > 1e-6),
        "pct_with_variance": sum(1 for v in all_variances if v > 1e-6) / max(n_instances, 1) * 100,
    }

    return {"per_instance": per_instance, "aggregate": aggregate}


def print_results(results: dict):
    """Pretty-print evaluation results."""
    agg = results["aggregate"]
    per = results["per_instance"]

    print("\n" + "=" * 70)
    print("  Rollout Evaluation Results")
    print("=" * 70)

    print(f"\n  Instances: {agg['n_instances']}")
    print(f"  Total rollouts: {agg['n_total_rollouts']}")
    print(f"  Avg rollouts/instance: {agg['avg_rollouts_per_instance']:.1f}")

    print(f"\n  --- Recall ---")
    print(f"  Avg recall (mean over rollouts): {agg['avg_recall_mean']:.4f}")
    print(f"  Best-of-N recall:                {agg['best_of_n_recall']:.4f}")
    print(f"  Improvement (best vs avg):       {agg['best_of_n_recall'] - agg['avg_recall_mean']:+.4f}")

    print(f"\n  --- Pass Rate ---")
    print(f"  Pass@1 (first rollout):   {agg['pass_at_1']:.4f}")
    print(f"  Pass@K (any rollout):     {agg['pass_at_k']:.4f}")
    print(f"  Avg hit rate per instance: {agg['avg_hit_rate']:.4f}")

    print(f"\n  --- Reward Variance (for advantage estimation) ---")
    print(f"  Avg variance:             {agg['avg_reward_variance']:.6f}")
    print(f"  Instances with variance:  {agg['n_with_variance']} ({agg['pct_with_variance']:.1f}%)")

    # Per-instance breakdown: show worst and best
    sorted_instances = sorted(per.items(), key=lambda x: x[1]["avg_recall"])

    print(f"\n  --- Bottom 5 (lowest avg recall) ---")
    for iid, v in sorted_instances[:5]:
        print(f"    {iid}: avg={v['avg_recall']:.3f} best={v['best_recall']:.3f} hits={v['n_hits']}/{v['n_rollouts']}")

    print(f"\n  --- Top 5 (highest avg recall) ---")
    for iid, v in sorted_instances[-5:]:
        print(f"    {iid}: avg={v['avg_recall']:.3f} best={v['best_recall']:.3f} hits={v['n_hits']}/{v['n_rollouts']}")

    # Variance distribution
    variances = [v["reward_variance"] for v in per.values()]
    if variances:
        print(f"\n  --- Variance distribution ---")
        for threshold in [0, 0.01, 0.05, 0.1, 0.2]:
            count = sum(1 for v in variances if v > threshold)
            print(f"    variance > {threshold:.2f}: {count} instances ({count/len(variances)*100:.1f}%)")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate N-rollout self-play results")
    parser.add_argument("--rollout_dir", type=str, default="toolplan_data/rl/rollouts",
                        help="Directory with rollout_0/, rollout_1/, ... subdirs")
    parser.add_argument("--merged_file", type=str, default="",
                        help="Merged rollouts file (alternative to rollout_dir)")
    parser.add_argument("--n_rollouts", type=int, default=8)
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_file", type=str, default="",
                        help="Save detailed results to JSON")
    args = parser.parse_args()

    # Load gold files
    logger.info("Loading gold file map...")
    gold_map = build_gold_map(args.dataset, args.split)
    logger.info(f"Gold map: {len(gold_map)} instances")

    # Load rollouts
    logger.info("Loading rollouts...")
    rollouts = load_rollouts(
        rollout_dir=args.rollout_dir if not args.merged_file else None,
        merged_file=args.merged_file,
        n_rollouts=args.n_rollouts,
    )
    logger.info(f"Loaded rollouts for {len(rollouts)} instances")

    # Evaluate
    results = evaluate_rollouts(rollouts, gold_map)
    print_results(results)

    # Save
    output_file = args.output_file
    if not output_file:
        if args.merged_file:
            output_file = args.merged_file.replace(".jsonl", "_eval.json")
        else:
            output_file = os.path.join(args.rollout_dir, "eval_results.json")

    # Save aggregate (per_instance can be huge)
    with open(output_file, "w") as f:
        json.dump(results["aggregate"], f, indent=2)
    logger.info(f"Aggregate results saved to {output_file}")

    # Save per-instance detail
    detail_file = output_file.replace(".json", "_detail.json")
    with open(detail_file, "w") as f:
        json.dump(results["per_instance"], f, indent=2)
    logger.info(f"Per-instance results saved to {detail_file}")


if __name__ == "__main__":
    main()
