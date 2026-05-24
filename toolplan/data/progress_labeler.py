"""
Step 2: Generate step-level progress labels from teacher trajectories.

For each step in a trajectory, computes:
  progress_t = Recall(E_{<=t}, F*) - Recall(E_{<t}, F*)

where E_{<=t} are files/entities mentioned up to step t, and F* are gold files.

Usage:
    python -m toolplan.data.progress_labeler \
        --traj_file toolplan_data/trajectories/loc_trajs.jsonl \
        --dataset "princeton-nlp/SWE-bench_Verified" \
        --output_file toolplan_data/progress_labels/progress_labels.jsonl
"""

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from typing import Optional

import numpy as np
from datasets import load_dataset

from toolplan.config import ProgressConfig, PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Gold file extraction from patches ──────────────────────────────────────────

def extract_gold_files_from_patch(patch: str) -> list[str]:
    """Extract modified file paths from a unified diff patch."""
    files = []
    for line in patch.split("\n"):
        # match lines like: diff --git a/path/to/file.py b/path/to/file.py
        if line.startswith("diff --git"):
            match = re.search(r"b/(.+)$", line)
            if match:
                fp = match.group(1)
                if fp not in files:
                    files.append(fp)
        # also match --- a/path and +++ b/path
        elif line.startswith("+++ b/"):
            fp = line[6:].strip()
            if fp and fp != "/dev/null" and fp not in files:
                files.append(fp)
    return files


def build_gold_map(dataset_name: str, split: str) -> dict[str, list[str]]:
    """Build instance_id → gold_files mapping from the dataset."""
    ds = load_dataset(dataset_name, split=split)
    gold_map = {}
    for instance in ds:
        instance_id = instance["instance_id"]
        patch = instance.get("patch", "")
        gold_files = extract_gold_files_from_patch(patch)
        gold_map[instance_id] = gold_files
    logger.info(f"Built gold map for {len(gold_map)} instances")
    return gold_map


def load_multi_level_gold(gt_file: str) -> dict[str, dict]:
    """Load file/module/entity gold from gt_location.jsonl.

    Returns: {instance_id: {"files": [...], "modules": [...], "entities": [...]}}
    Format of gold:
        modules:  "file.py:ClassName"
        entities: "file.py:ClassName.method_name"
    """
    gold_map = {}
    with open(gt_file) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            iid = d["instance_id"]
            file_changes = d.get("file_changes") or []
            files, modules, entities = [], [], []
            for fc in file_changes:
                fp = fc.get("file")
                if fp and fp not in files:
                    files.append(fp)
                changes = fc.get("changes", {}) or {}
                for m in changes.get("edited_modules", []) or []:
                    if m and m not in modules:
                        modules.append(m)
                for e in changes.get("edited_entities", []) or []:
                    if e and e not in entities:
                        entities.append(e)
            gold_map[iid] = {
                "files": files,
                "modules": modules,
                "entities": entities,
            }
    logger.info(f"Loaded multi-level gold for {len(gold_map)} instances")
    return gold_map


def _name_matches(a: str, b: str) -> bool:
    """Two qualified names match if they refer to overlapping code elements.

    Cases (all symmetric):
      - exact match:                     'Foo.bar' == 'Foo.bar'
      - parent / child via dot prefix:   'Foo' vs 'Foo.bar'
      - bare-name vs qualified suffix:   'bar' vs 'Foo.bar'
    """
    if a == b:
        return True
    if a.startswith(b + ".") or b.startswith(a + "."):
        return True
    if a.endswith("." + b) or b.endswith("." + a):
        return True
    return False


def _loc_matches(pred_loc: str, gold_loc: str) -> bool:
    """Lenient location match: same file, then name-matches per _name_matches."""
    if pred_loc == gold_loc:
        return True
    pf, _, pn = pred_loc.partition(":")
    gf, _, gn = gold_loc.partition(":")
    if pf != gf:
        return False
    # file-only entries (no ':') only match exact file names — handled by line above
    if not pn or not gn:
        return False
    return _name_matches(pn, gn)


def compute_prf(pred: set, gold: set) -> tuple[float, float, float]:
    """Compute (precision, recall, F1) with lenient name matching.

    Each gold counted as recalled if ANY pred matches it (suffix-aware).
    Each pred counted as a true positive if ANY gold matches it (suffix-aware).
    For pure file-level sets (no ':'), this reduces to exact set matching.
    """
    if not gold or not pred:
        return 0.0, 0.0, 0.0

    pred_list = list(pred)
    gold_list = list(gold)

    # File-level: any element without ':' compares exactly.
    has_qualified = any(":" in x for x in pred_list + gold_list)
    if not has_qualified:
        tp = len(pred & gold)
        p = tp / len(pred)
        r = tp / len(gold)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    # Qualified: lenient matching
    tp_recall = 0
    for g in gold_list:
        if any(_loc_matches(p_, g) for p_ in pred_list):
            tp_recall += 1
    tp_precision = 0
    for p_ in pred_list:
        if any(_loc_matches(p_, g) for g in gold_list):
            tp_precision += 1

    p = tp_precision / len(pred_list)
    r = tp_recall / len(gold_list)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


# ── File/entity extraction from tool observations ─────────────────────────────

def extract_files_from_text(text: str) -> set[str]:
    """Extract Python file paths mentioned in a text string."""
    pattern = r'[\w\./-]+\.py'
    matches = re.findall(pattern, text)
    files = set()
    for m in matches:
        # clean up leading dots or garbage
        cleaned = m.lstrip("./")
        if "/" in cleaned and cleaned.endswith(".py"):
            files.add(cleaned)
    return files


def extract_entities_from_text(text: str) -> set[str]:
    """Extract entity references (file:Class.method format) from text."""
    # Match patterns like "src/file.py:ClassName" or "src/file.py:ClassName.method"
    pattern = r'([\w\./-]+\.py):([\w\.]+)'
    matches = re.findall(pattern, text)
    entities = set()
    for file_path, entity_name in matches:
        cleaned = file_path.lstrip("./")
        if "/" in cleaned:
            entities.add(f"{cleaned}:{entity_name}")
    return entities


def extract_modules_from_text(text: str) -> set[str]:
    """Extract module references (file:ClassName, no method) from text.

    A module is the first qualified-name segment after the file path.
    For 'file.py:Foo.bar' → module 'file.py:Foo'.
    For 'file.py:foo' (top-level function/class) → module 'file.py:foo'.
    """
    pattern = r'([\w\./-]+\.py):([\w\.]+)'
    matches = re.findall(pattern, text)
    modules = set()
    for file_path, qual in matches:
        cleaned = file_path.lstrip("./")
        if "/" not in cleaned:
            continue
        first_seg = qual.split(".")[0]
        modules.add(f"{cleaned}:{first_seg}")
    return modules


def extract_tool_call_info(message: dict) -> Optional[dict]:
    """Extract tool name and arguments from an assistant message.

    Supports two formats:
    1. Native tool_calls field (OpenAI function calling format)
    2. Text-based <function=tool_name> format (Qwen3 text-mode tool calling)
    """
    # Format 1: native tool_calls
    tool_calls = message.get("tool_calls", [])
    if tool_calls:
        tc = tool_calls[0]
        func = tc.get("function", {})
        name = func.get("name", "")
        args_str = func.get("arguments", "{}")
        try:
            arguments = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            arguments = {}
        return {"tool_name": name, "arguments": arguments}

    # Format 2: text-based <function=tool_name> in content
    content = message.get("content", "") or ""
    func_match = re.search(r'<function=(\w+)>', content)
    if func_match:
        name = func_match.group(1)
        # Extract parameters
        arguments = {}
        for param_match in re.finditer(r'<parameter=(\w+)>(.*?)(?:</parameter>|$)', content, re.DOTALL):
            key = param_match.group(1)
            val = param_match.group(2).strip()
            arguments[key] = val
        return {"tool_name": name, "arguments": arguments}

    return None


# ── Step-level progress computation ───────────────────────────────────────────

def parse_trajectory_steps(messages: list[dict]) -> list[dict]:
    """Parse message list into structured steps.

    Each step = (assistant thought+tool_call, tool observation).
    Returns list of step dicts with keys:
        - step_idx: int
        - assistant_message: the assistant's message dict
        - tool_call: {tool_name, arguments} or None
        - observation: the tool response content or None
        - thought: assistant's reasoning text
    """
    steps = []
    i = 0
    step_idx = 0

    while i < len(messages):
        msg = messages[i]

        if msg.get("role") == "assistant":
            step = {
                "step_idx": step_idx,
                "assistant_message": msg,
                "tool_call": extract_tool_call_info(msg),
                "thought": msg.get("content", "") or "",
                "observation": None,
            }

            # look for the next tool/user message as observation
            if i + 1 < len(messages) and messages[i + 1].get("role") in ("tool", "user"):
                obs_msg = messages[i + 1]
                step["observation"] = obs_msg.get("content", "")
                i += 2
            else:
                i += 1

            steps.append(step)
            step_idx += 1
        else:
            i += 1

    return steps


def compute_step_progress(
    steps: list[dict],
    gold_files: list[str],
    gold_modules: list[str] = None,
    gold_entities: list[str] = None,
    step_cost_penalty: float = 0.02,
    entity_level: bool = False,
) -> list[dict]:
    """Compute progress label for each step at file/module/entity granularity.

    Returns list of dicts, one per step, including per-level progress:
        - progress_file, progress_module, progress_entity
        - recall_after_file, recall_after_module, recall_after_entity
        - files_found_cumulative, modules_found_cumulative, entities_found_cumulative
        - tool_name, tool_args, thought, observation_preview
    """
    if not gold_files:
        return []

    gold_files_set = set(gold_files)
    gold_modules_set = set(gold_modules or [])
    gold_entities_set = set(gold_entities or [])

    cum_files = set()
    cum_modules = set()
    cum_entities = set()
    results = []
    total_steps = len(steps)

    def _recall(found: set, gold: set) -> float:
        return len(found & gold) / len(gold) if gold else 0.0

    for step in steps:
        rec_f_before = _recall(cum_files, gold_files_set)
        rec_m_before = _recall(cum_modules, gold_modules_set)
        rec_e_before = _recall(cum_entities, gold_entities_set)

        new_files = set()
        new_modules = set()
        new_entities = set()
        for txt_key in ("observation", "thought"):
            txt = step.get(txt_key) or ""
            if txt:
                new_files |= extract_files_from_text(txt)
                new_modules |= extract_modules_from_text(txt)
                new_entities |= extract_entities_from_text(txt)
        if step.get("tool_call") and step["tool_call"].get("arguments"):
            args_text = json.dumps(step["tool_call"]["arguments"])
            new_files |= extract_files_from_text(args_text)
            new_modules |= extract_modules_from_text(args_text)
            new_entities |= extract_entities_from_text(args_text)

        files_new_gold = (new_files - cum_files) & gold_files_set
        modules_new_gold = (new_modules - cum_modules) & gold_modules_set
        entities_new_gold = (new_entities - cum_entities) & gold_entities_set

        cum_files |= new_files
        cum_modules |= new_modules
        cum_entities |= new_entities

        rec_f_after = _recall(cum_files, gold_files_set)
        rec_m_after = _recall(cum_modules, gold_modules_set)
        rec_e_after = _recall(cum_entities, gold_entities_set)

        progress_file = rec_f_after - rec_f_before
        progress_module = rec_m_after - rec_m_before
        progress_entity = rec_e_after - rec_e_before

        # legacy field for backward compat: file-level only
        progress = progress_file
        progress_eff = progress - step_cost_penalty

        tool_call = step.get("tool_call") or {}
        tool_name = tool_call.get("tool_name") or step.get("tool_name", "none")
        tool_args = tool_call.get("arguments") or step.get("tool_args", {})

        results.append({
            "step_idx": step["step_idx"],
            "tool_name": tool_name,
            "tool_args": tool_args,
            "thought": (step.get("thought") or "")[:500],
            "observation_preview": (step.get("observation") or "")[:500],
            # gold-only set views
            "files_found_new": sorted(files_new_gold),
            "modules_found_new": sorted(modules_new_gold),
            "entities_found_new": sorted(entities_new_gold),
            "files_found_cumulative": sorted(cum_files & gold_files_set),
            "modules_found_cumulative": sorted(cum_modules & gold_modules_set),
            "entities_found_cumulative": sorted(cum_entities & gold_entities_set),
            # per-level recall and progress
            "recall_before": round(rec_f_before, 4),
            "recall_after": round(rec_f_after, 4),
            "recall_before_file": round(rec_f_before, 4),
            "recall_after_file": round(rec_f_after, 4),
            "recall_before_module": round(rec_m_before, 4),
            "recall_after_module": round(rec_m_after, 4),
            "recall_before_entity": round(rec_e_before, 4),
            "recall_after_entity": round(rec_e_after, 4),
            "progress": round(progress, 4),
            "progress_eff": round(progress_eff, 4),
            "progress_file": round(progress_file, 4),
            "progress_module": round(progress_module, 4),
            "progress_entity": round(progress_entity, 4),
            "total_steps": total_steps,
        })

    return results


# ── Main processing ───────────────────────────────────────────────────────────

def _get_rollout_committed(
    traj_data: dict,
    traj_idx: int,
) -> tuple[set, set, set]:
    """Extract committed (final answer) files/modules/entities for a single rollout.

    The top-level traj_data may have found_files as either:
      - flat list (1 rollout): ['file1', 'file2', ...]
      - nested list (N rollouts): [[...rollout0...], [...rollout1...], ...]
    """
    def _pick(field: str) -> list:
        v = traj_data.get(field)
        if not v:
            return []
        # nested per-rollout
        if isinstance(v, list) and v and isinstance(v[0], list):
            return v[traj_idx] if traj_idx < len(v) else []
        # flat
        if isinstance(v, list):
            return v
        return []

    return (
        set(_pick("found_files")),
        set(_pick("found_modules")),
        set(_pick("found_entities")),
    )


def process_trajectory_file(
    traj_file: str,
    gold_map: dict[str, list[str]],
    cfg: ProgressConfig,
    multi_gold_map: dict[str, dict] = None,
) -> list[dict]:
    """Process all trajectories and generate progress labels.

    Returns list of labeled instances, each with:
        - instance_id, gold_files, gold_modules, gold_entities, trajectory_idx
        - success: whether trajectory found any gold file
        - final_recall, final_recall_module, final_recall_entity
        - final_f1_file, final_f1_module, final_f1_entity (committed answer F1)
        - committed_recall: |observed ∩ gold ∩ answered| / |observed ∩ gold|
        - steps: list of step-level progress dicts
    """
    results = []
    multi_gold_map = multi_gold_map or {}

    with open(traj_file) as f:
        trajectories = [json.loads(line) for line in f]

    logger.info(f"Processing {len(trajectories)} instances from {traj_file}")

    for traj_data in trajectories:
        instance_id = traj_data["instance_id"]
        gold_files = gold_map.get(instance_id, [])
        if not gold_files:
            logger.warning(f"No gold files for {instance_id}, skipping")
            continue

        ml_gold = multi_gold_map.get(instance_id, {})
        gold_modules = ml_gold.get("modules", [])
        gold_entities = ml_gold.get("entities", [])

        problem_statement = traj_data.get("meta_data", {}).get("problem_statement", "")
        trajs = traj_data.get("loc_trajs", {}).get("trajs", [])

        for traj_idx, traj in enumerate(trajs):
            messages = traj.get("messages", [])
            steps = parse_trajectory_steps(messages)

            step_progress = compute_step_progress(
                steps,
                gold_files=gold_files,
                gold_modules=gold_modules,
                gold_entities=gold_entities,
                step_cost_penalty=cfg.step_cost_penalty,
                entity_level=cfg.entity_level,
            )
            if not step_progress:
                continue

            last = step_progress[-1]
            final_recall = last["recall_after_file"]
            final_recall_module = last["recall_after_module"]
            final_recall_entity = last["recall_after_entity"]

            # Committed (final answer) per-rollout
            committed_files, committed_modules, committed_entities = _get_rollout_committed(
                traj_data, traj_idx,
            )

            # Multi-granularity F1 on committed answer
            p_f, r_f, f1_f = compute_prf(committed_files, set(gold_files))
            p_m, r_m, f1_m = compute_prf(committed_modules, set(gold_modules))
            p_e, r_e, f1_e = compute_prf(committed_entities, set(gold_entities))

            # Discovery-commitment: of the gold files we OBSERVED, how many did we ANSWER?
            observed_gold_files = set(last["files_found_cumulative"])
            denom = len(observed_gold_files)
            committed_observed_gold = observed_gold_files & committed_files
            commitment_recall_file = len(committed_observed_gold) / denom if denom > 0 else 0.0

            observed_gold_modules = set(last["modules_found_cumulative"])
            denom_m = len(observed_gold_modules)
            committed_observed_modules = observed_gold_modules & committed_modules
            commitment_recall_module = len(committed_observed_modules) / denom_m if denom_m > 0 else 0.0

            observed_gold_entities = set(last["entities_found_cumulative"])
            denom_e = len(observed_gold_entities)
            committed_observed_entities = observed_gold_entities & committed_entities
            commitment_recall_entity = len(committed_observed_entities) / denom_e if denom_e > 0 else 0.0

            # Type B failures: observed gold but NOT answered
            type_b_files = sorted(observed_gold_files - committed_files)
            type_b_modules = sorted(observed_gold_modules - committed_modules)
            type_b_entities = sorted(observed_gold_entities - committed_entities)

            results.append({
                "instance_id": instance_id,
                "trajectory_idx": traj_idx,
                "gold_files": gold_files,
                "gold_modules": gold_modules,
                "gold_entities": gold_entities,
                "success": final_recall > 0,
                "final_recall": round(final_recall, 4),
                "final_recall_file": round(final_recall, 4),
                "final_recall_module": round(final_recall_module, 4),
                "final_recall_entity": round(final_recall_entity, 4),
                "final_f1_file": round(f1_f, 4),
                "final_f1_module": round(f1_m, 4),
                "final_f1_entity": round(f1_e, 4),
                "final_precision_file": round(p_f, 4),
                "final_precision_module": round(p_m, 4),
                "final_precision_entity": round(p_e, 4),
                "committed_recall_file": round(commitment_recall_file, 4),
                "committed_recall_module": round(commitment_recall_module, 4),
                "committed_recall_entity": round(commitment_recall_entity, 4),
                "type_b_files": type_b_files,
                "type_b_modules": type_b_modules,
                "type_b_entities": type_b_entities,
                "committed_files": sorted(committed_files),
                "committed_modules": sorted(committed_modules),
                "committed_entities": sorted(committed_entities),
                "num_steps": len(step_progress),
                "steps": step_progress,
                "messages": messages,
                "first_finish_output": traj.get("first_finish_output"),
            })

    # stats
    total = len(results)
    successes = sum(1 for r in results if r["success"])
    avg_steps = sum(r["num_steps"] for r in results) / max(total, 1)
    avg_f1_file = np.mean([r["final_f1_file"] for r in results]) if results else 0.0
    avg_f1_module = np.mean([r["final_f1_module"] for r in results]) if results else 0.0
    avg_f1_entity = np.mean([r["final_f1_entity"] for r in results]) if results else 0.0
    avg_commit_f = np.mean([r["committed_recall_file"] for r in results]) if results else 0.0
    n_typeb = sum(1 for r in results if r["type_b_files"])
    logger.info(
        f"Processed {total} trajectories: "
        f"{successes} successful ({successes/max(total,1)*100:.1f}%), "
        f"avg {avg_steps:.1f} steps"
    )
    logger.info(
        f"Final F1 (committed answer): file={avg_f1_file:.4f}, "
        f"module={avg_f1_module:.4f}, entity={avg_f1_entity:.4f}"
    )
    logger.info(
        f"Commitment recall (file): {avg_commit_f:.4f}  "
        f"(Type-B failures: {n_typeb}/{total} = {n_typeb/max(total,1)*100:.1f}%)"
    )

    return results


def save_results(results: list[dict], output_file: str):
    """Save progress labels to JSONL."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Saved {len(results)} labeled trajectories to {output_file}")


def process_trajectories_inmemory(
    trajectories: list[dict],
    gold_map: dict[str, list[str]],
    cfg: 'ProgressConfig',
    multi_gold_map: dict[str, dict] = None,
) -> list[dict]:
    """In-memory version of process_trajectory_file.

    Accepts a list of trajectory dicts (loc_outputs/trajs format) instead of reading from file.
    Used by on-policy training to label freshly-collected rollouts.
    """
    results = []
    multi_gold_map = multi_gold_map or {}
    for traj_data in trajectories:
        instance_id = traj_data["instance_id"]
        gold_files = gold_map.get(instance_id, [])
        if not gold_files:
            continue
        ml_gold = multi_gold_map.get(instance_id, {})
        gold_modules = ml_gold.get("modules", [])
        gold_entities = ml_gold.get("entities", [])
        problem_statement = traj_data.get("meta_data", {}).get("problem_statement", "")
        trajs = traj_data.get("loc_trajs", {}).get("trajs", [])

        for traj_idx, traj in enumerate(trajs):
            messages = traj.get("messages", [])
            steps = parse_trajectory_steps(messages)
            step_progress = compute_step_progress(
                steps,
                gold_files=gold_files,
                gold_modules=gold_modules,
                gold_entities=gold_entities,
                step_cost_penalty=cfg.step_cost_penalty,
                entity_level=cfg.entity_level,
            )
            if not step_progress:
                continue

            last = step_progress[-1]
            final_recall = last["recall_after_file"]
            final_recall_module = last["recall_after_module"]
            final_recall_entity = last["recall_after_entity"]

            committed_files, committed_modules, committed_entities = _get_rollout_committed(
                traj_data, traj_idx,
            )
            p_f, r_f, f1_f = compute_prf(committed_files, set(gold_files))
            p_m, r_m, f1_m = compute_prf(committed_modules, set(gold_modules))
            p_e, r_e, f1_e = compute_prf(committed_entities, set(gold_entities))

            observed_gold_files = set(last["files_found_cumulative"])
            denom = len(observed_gold_files)
            commitment_recall_file = (
                len(observed_gold_files & committed_files) / denom if denom > 0 else 0.0
            )
            observed_gold_modules = set(last["modules_found_cumulative"])
            denom_m = len(observed_gold_modules)
            commitment_recall_module = (
                len(observed_gold_modules & committed_modules) / denom_m if denom_m > 0 else 0.0
            )
            observed_gold_entities = set(last["entities_found_cumulative"])
            denom_e = len(observed_gold_entities)
            commitment_recall_entity = (
                len(observed_gold_entities & committed_entities) / denom_e if denom_e > 0 else 0.0
            )

            results.append({
                "instance_id": instance_id,
                "trajectory_idx": traj_idx,
                "gold_files": gold_files,
                "gold_modules": gold_modules,
                "gold_entities": gold_entities,
                "messages": messages,
                "final_f1_file": round(f1_f, 4),
                "final_f1_module": round(f1_m, 4),
                "final_f1_entity": round(f1_e, 4),
                "committed_recall_file": round(commitment_recall_file, 4),
                "committed_recall_module": round(commitment_recall_module, 4),
                "committed_recall_entity": round(commitment_recall_entity, 4),
                "num_steps": len(step_progress),
                "steps": step_progress,
            })
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate progress labels")
    parser.add_argument("--traj_file", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--step_cost_penalty", type=float, default=0.02)
    parser.add_argument("--entity_level", action="store_true")
    parser.add_argument("--output_file", type=str, default=str(ProgressConfig().output_file))
    parser.add_argument("--gt_file", type=str, default=None,
                        help="Optional gt_location.jsonl with file_changes (modules/entities). "
                             "If provided, multi-level gold is loaded and reward is multi-granularity.")
    args = parser.parse_args()

    cfg = ProgressConfig(
        traj_file=args.traj_file,
        dataset=args.dataset,
        split=args.split,
        step_cost_penalty=args.step_cost_penalty,
        entity_level=args.entity_level,
        output_file=args.output_file,
    )

    # build gold file map (file-level)
    gold_map = build_gold_map(cfg.dataset, cfg.split)

    # multi-level gold (modules + entities) if gt_file provided
    multi_gold_map = {}
    if args.gt_file:
        multi_gold_map = load_multi_level_gold(args.gt_file)

    # process trajectories
    results = process_trajectory_file(cfg.traj_file, gold_map, cfg, multi_gold_map=multi_gold_map)

    # save
    save_results(results, cfg.output_file)


if __name__ == "__main__":
    main()
