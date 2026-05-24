"""
Step 3: Prepare SFT training data from teacher trajectories.

Converts loc_trajs.jsonl → sft_train.jsonl compatible with sft_train.py.
Filters to successful trajectories only and selects the best trajectory
per instance (fewest steps with highest recall).

Usage:
    python -m toolplan.training.prepare_sft_data \
        --progress_file toolplan_data/progress_labels/progress_labels.jsonl \
        --output_file toolplan_data/policy/sft_train_data.jsonl \
        --strategy best
"""

import argparse
import json
import logging
import os
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def select_trajectories(
    labeled_trajs: list[dict],
    strategy: str = "best",
) -> list[dict]:
    """Select which trajectories to use for SFT.

    Strategies:
        - "all_success": all trajectories that found at least one gold file
        - "best": best trajectory per instance (highest recall, then fewest steps)
        - "all": all trajectories including failures
    """
    if strategy == "all":
        return labeled_trajs

    if strategy == "all_success":
        selected = [t for t in labeled_trajs if t["success"]]
        logger.info(f"Strategy=all_success: {len(selected)}/{len(labeled_trajs)} trajectories")
        return selected

    if strategy == "best":
        # group by instance_id, pick best trajectory per instance
        by_instance = defaultdict(list)
        for t in labeled_trajs:
            if t["success"]:
                by_instance[t["instance_id"]].append(t)

        selected = []
        for instance_id, trajs in by_instance.items():
            # sort by recall (desc), then by num_steps (asc)
            trajs.sort(key=lambda x: (-x["final_recall"], x["num_steps"]))
            selected.append(trajs[0])

        logger.info(
            f"Strategy=best: {len(selected)} trajectories "
            f"from {len(by_instance)} instances"
        )
        return selected

    raise ValueError(f"Unknown strategy: {strategy}")


def trajectory_to_sft_messages(traj: dict) -> list[dict]:
    """Convert a labeled trajectory to the messages format expected by sft_train.py.

    The existing sft_train.py expects:
        {"messages": [{"role": "system/user/assistant/tool", "content": "..."}]}

    We keep the original message list from the trajectory, filtering out
    any empty or malformed messages.
    """
    messages = traj.get("messages", [])
    if not messages:
        return []

    cleaned = []
    for msg in messages:
        role = msg.get("role", "")
        if role not in ("system", "user", "assistant", "tool"):
            continue

        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", None)

        entry = {"role": role, "content": content or ""}

        # preserve tool_calls for assistant messages
        if role == "assistant" and tool_calls:
            entry["tool_calls"] = tool_calls

        # preserve tool_call_id and name for tool messages
        if role == "tool":
            if "tool_call_id" in msg:
                entry["tool_call_id"] = msg["tool_call_id"]
            if "name" in msg:
                entry["name"] = msg["name"]

        cleaned.append(entry)

    return cleaned


def prepare_sft_data(
    progress_file: str,
    output_file: str,
    strategy: str = "best",
):
    """Main function to prepare SFT training data."""
    # load labeled trajectories
    with open(progress_file) as f:
        labeled_trajs = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(labeled_trajs)} labeled trajectories")

    # select trajectories
    selected = select_trajectories(labeled_trajs, strategy)

    # convert to SFT format
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    count = 0
    with open(output_file, "w") as f:
        for traj in selected:
            messages = trajectory_to_sft_messages(traj)
            if len(messages) < 3:  # need at least system + user + assistant
                continue
            f.write(json.dumps({"messages": messages}) + "\n")
            count += 1

    logger.info(f"Saved {count} SFT training examples to {output_file}")

    # stats
    avg_steps = sum(t["num_steps"] for t in selected) / max(len(selected), 1)
    avg_recall = sum(t["final_recall"] for t in selected) / max(len(selected), 1)
    logger.info(f"Avg steps: {avg_steps:.1f}, Avg recall: {avg_recall:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT data from trajectories")
    parser.add_argument("--progress_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument(
        "--strategy", type=str, default="best",
        choices=["all", "all_success", "best"],
    )
    args = parser.parse_args()
    prepare_sft_data(args.progress_file, args.output_file, args.strategy)


if __name__ == "__main__":
    main()
