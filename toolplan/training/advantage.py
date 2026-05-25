"""
Action-level credit assignment for code localization agents.

Computes history-aware, action-level advantages by grouping turns that share
the same recent action history (state proxy) and normalizing each turn's
discounted return within its group.

Designed for the phase-separated pipeline:
    Phase 1: Rollout (vLLM inference) -> trajectories
    Phase 2: This module -> compute advantages
    Phase 3: Advantage-weighted SFT

Usage:
    python -m toolplan.training.advantage \
        --progress_file toolplan_data/progress_labels/progress_labels.jsonl \
        --output_file toolplan_data/rl/advantages.jsonl \
        --history_length 2 \
        --gamma 0.95
"""

import argparse
import hashlib
import json
import logging
import os
from collections import defaultdict

import numpy as np

from toolplan.config import PROGRESS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# -- Action Hashing -----------------------------------------------------------

def hash_tool_call(tool_name: str, tool_args: dict) -> str:
    """Hash a tool call to a compact string for exact-match action-history grouping."""
    key = json.dumps({"tool": tool_name, "args": tool_args}, sort_keys=True)
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def to_hashable(x):
    """Convert arbitrary nested structure to hashable key (following the original implementation)."""
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        return str(x)


# -- Step Return Computation ---------------------------------------------------

def compute_step_returns(
    steps: list[dict],
    gamma: float = 0.95,
    step_cost: float = 0.02,
    final_bonus_scale: float = 1.0,
    terminal_only: bool = False,
    # multi-granularity weights (CodeScout-style: r = α_F·R_F + α_M·R_M + α_E·R_E)
    alpha_file: float = 1.0,
    alpha_module: float = 0.5,
    alpha_entity: float = 0.5,
    # discovery-commitment bonus: reward inclusion of OBSERVED gold in answer
    commitment_bonus: float = 0.5,
    # final-step F1 reward (CodeScout: multi-granularity F1)
    final_f1_file: float = None,
    final_f1_module: float = None,
    final_f1_entity: float = None,
    # discovery-commitment per granularity (sum across f/m/u; 0 if denom=0)
    final_committed_recall_file: float = None,
    final_committed_recall_module: float = None,
    final_committed_recall_entity: float = None,
) -> list[float]:
    """Compute discounted step-level returns with multi-granularity reward.

    Per-step reward:
        r_t = α_F·ΔR_F + α_M·ΔR_M + α_E·ΔR_E - step_cost

    Final-step bonus (CodeScout-style):
        r_T += final_bonus_scale · (α_F·F1_F + α_M·F1_M + α_E·F1_E)
        r_T += commitment_bonus · (α_F·DC_F + α_M·DC_M + α_E·DC_E)

    Falls back to legacy file-only reward if multi-level fields missing.
    """
    n = len(steps)
    if n == 0:
        return []

    rewards = []
    for i, step in enumerate(steps):
        # per-step progress (multi-level if available, else file-only)
        pf = step.get("progress_file", step.get("progress", 0.0))
        pm = step.get("progress_module", 0.0)
        pe = step.get("progress_entity", 0.0)
        if terminal_only:
            r = 0.0
        else:
            r = alpha_file * pf + alpha_module * pm + alpha_entity * pe - step_cost

        if i == n - 1:
            # multi-granularity F1 final bonus (preferred)
            if final_f1_file is not None:
                f1_bonus = (
                    alpha_file * (final_f1_file or 0.0)
                    + alpha_module * (final_f1_module or 0.0)
                    + alpha_entity * (final_f1_entity or 0.0)
                )
                r += final_bonus_scale * f1_bonus
            else:
                # legacy fallback: recall_after only
                r += final_bonus_scale * step.get("recall_after", 0.0)
            # discovery-commitment bonus (multi-granularity sum, weighted by alphas)
            if not terminal_only and any(x is not None for x in (
                final_committed_recall_file,
                final_committed_recall_module,
                final_committed_recall_entity,
            )):
                dc_bonus = (
                    alpha_file * (final_committed_recall_file or 0.0)
                    + alpha_module * (final_committed_recall_module or 0.0)
                    + alpha_entity * (final_committed_recall_entity or 0.0)
                )
                r += commitment_bonus * dc_bonus
        rewards.append(r)

    returns = [0.0] * n
    running = 0.0
    for t in reversed(range(n)):
        running = rewards[t] + gamma * running
        returns[t] = running

    return returns


# -- Hierarchical Advantage (observation hash based) ---------------------

def compute_advantages(
    trajectories: list[dict],
    history_length: int = 2,
    gamma: float = 0.95,
    step_cost: float = 0.02,
    alpha: float = 1.0,
    use_episode_base: bool = True,
    mode: str = "mean_norm",
    epsilon: float = 1e-6,
    # multi-granularity reward weights
    alpha_file: float = 1.0,
    alpha_module: float = 0.5,
    alpha_entity: float = 0.5,
    commitment_bonus: float = 0.5,
    final_bonus_scale: float = 1.0,
    # clustering scheme
    cluster_scheme: str = "state",  # "state" (history only, excl. current action) | "state_action" (legacy)
    terminal_only: bool = False,
) -> list[dict]:
    """Compute advantages for a group of trajectories (same issue).

    Cluster scheme:
      - "state":         cluster key = last K tool_calls EXCLUDING current step
                         → cluster contains different actions taken at the same state-history
                         → mean baseline approximates V(s) (the intended baseline)
      - "state_action":  legacy. cluster key = last K tool_calls INCLUDING current step
                         → cluster contains only same-action trajectories (variance reduction)
    """
    if not trajectories:
        return []

    # -- Extract observation hashes and returns for each trajectory --
    all_obs_hashes = []  # flat list of obs hashes across all trajectories
    all_returns = []     # flat list of returns
    traj_map = []        # (traj_idx, step_idx) for each entry

    for ti, traj in enumerate(trajectories):
        steps = traj.get("steps", [])
        if not steps:
            continue

        returns = compute_step_returns(
            steps,
            gamma=gamma,
            step_cost=step_cost,
            alpha_file=alpha_file,
            alpha_module=alpha_module,
            alpha_entity=alpha_entity,
            commitment_bonus=commitment_bonus,
            final_bonus_scale=final_bonus_scale,
            final_f1_file=traj.get("final_f1_file"),
            final_f1_module=traj.get("final_f1_module"),
            final_f1_entity=traj.get("final_f1_entity"),
            final_committed_recall_file=traj.get("committed_recall_file"),
            final_committed_recall_module=traj.get("committed_recall_module"),
            final_committed_recall_entity=traj.get("committed_recall_entity"),
            terminal_only=terminal_only,
        )

        for si, step in enumerate(steps):
            tool_name = step.get("tool_name", "none")
            tool_args = step.get("tool_args", {})
            obs_hash = hash_tool_call(tool_name, tool_args)
            all_obs_hashes.append(obs_hash)
            all_returns.append(returns[si])
            traj_map.append((ti, si))

    N = len(all_obs_hashes)
    if N == 0:
        return trajectories

    returns_arr = np.array(all_returns, dtype=np.float32)

    # -- Episode-level advantage (base group) --
    episode_advs = np.zeros(N, dtype=np.float32)
    if use_episode_base:
        episode_returns = {}
        for i, (ti, si) in enumerate(traj_map):
            if ti not in episode_returns:
                episode_returns[ti] = returns_arr[i]

        ep_rets = list(episode_returns.values())
        if len(ep_rets) > 1:
            ep_mean = np.mean(ep_rets)
            ep_std = np.std(ep_rets) if mode == "mean_std_norm" else 1.0
            for i, (ti, si) in enumerate(traj_map):
                if mode == "mean_std_norm":
                    episode_advs[i] = (episode_returns[ti] - ep_mean) / (ep_std + epsilon)
                else:
                    episode_advs[i] = episode_returns[ti] - ep_mean

    # -- Hierarchical step-level grouping --
    # Build per-trajectory action hash sequences
    traj_obs = defaultdict(list)  # ti -> list of (global_idx, action_hash)
    for i, (ti, si) in enumerate(traj_map):
        traj_obs[ti].append((i, all_obs_hashes[i]))

    # Build clusters: key = (k, history_hash_sequence), value = list of global_idx
    # cluster_scheme="state":         seq = hashes[si-k:si]      (history only; current action EXCLUDED)
    # cluster_scheme="state_action":  seq = hashes[si-k+1:si+1]  (history + current; same-action grouping)
    max_k = history_length + 1
    clusters = defaultdict(list)

    for ti, obs_list in traj_obs.items():
        hashes = [to_hashable(h) for _, h in obs_list]
        T = len(hashes)
        for si in range(T):
            global_idx = obs_list[si][0]
            for k in range(1, max_k + 1):
                if cluster_scheme == "state":
                    # k previous action hashes (history). At si=0 with k>=1, seq=() → all step-0s grouped.
                    if si - k < 0 and si > 0:
                        # not enough history for this depth; skip
                        break
                    seq = tuple(hashes[si - k: si]) if si >= k else tuple()
                else:  # state_action (legacy)
                    if si + 1 < k:
                        break
                    seq = tuple(hashes[si - k + 1: si + 1])
                clusters[(k, seq)].append(global_idx)

    # Compute per-step items: for each step, collect (k, advantage) pairs
    per_step_items = defaultdict(list)

    for (k, _key), members in clusters.items():
        if len(members) <= 1:
            continue

        member_returns = returns_arr[members]
        mean = member_returns.mean()
        std = member_returns.std()

        for j, idx in enumerate(members):
            if mode == "mean_std_norm":
                adv = (member_returns[j] - mean) / (std + epsilon)
            else:
                adv = member_returns[j] - mean
            per_step_items[idx].append((k, adv))

    # -- Aggregate advantages with adaptive weighting --
    step_advantages = np.zeros(N, dtype=np.float32)

    for i in range(N):
        items = per_step_items.get(i, [])

        if use_episode_base:
            items = [(0, episode_advs[i])] + items

        if not items:
            step_advantages[i] = 0.0
            continue

        ks = np.array([k for k, _ in items], dtype=np.float32)
        advs = np.array([a for _, a in items], dtype=np.float32)

        valid = advs != 0
        if valid.any():
            ks_v = ks[valid]
            advs_v = advs[valid]
            L = ks_v + 1
            w_raw = L ** alpha
            w = w_raw / (w_raw.sum() + epsilon)
            step_advantages[i] = (w * advs_v).sum()

    # -- Assign advantages back to trajectories --
    result = []
    for ti, traj in enumerate(trajectories):
        traj_copy = dict(traj)
        steps = traj.get("steps", [])
        advantages = []
        for si in range(len(steps)):
            idx_matches = [i for i, (t, s) in enumerate(traj_map) if t == ti and s == si]
            if idx_matches:
                advantages.append(float(step_advantages[idx_matches[0]]))
            else:
                advantages.append(0.0)
        traj_copy["step_advantages"] = advantages
        result.append(traj_copy)

    return result


# -- Advantage -> Weight conversion --------------------------------------------

def advantages_to_weights(
    advantages: list[float],
    beta: float = 1.5,
    weight_min: float = 0.1,
    weight_max: float = 3.0,
) -> list[float]:
    """Convert advantages to SFT training weights.

    w_t = clip(exp(beta * A_t), weight_min, weight_max)
    """
    weights = []
    for a in advantages:
        w = np.exp(beta * a)
        w = float(np.clip(w, weight_min, weight_max))
        weights.append(w)
    return weights


# -- Main Pipeline -------------------------------------------------------------

def process_all_trajectories(
    progress_file: str,
    history_length: int = 2,
    gamma: float = 0.95,
    step_cost: float = 0.02,
    alpha: float = 1.0,
    beta: float = 1.5,
    mode: str = "mean_norm",
    use_episode_base: bool = True,
    alpha_file: float = 1.0,
    alpha_module: float = 0.5,
    alpha_entity: float = 0.5,
    commitment_bonus: float = 0.5,
    final_bonus_scale: float = 1.0,
    cluster_scheme: str = "state",
    terminal_only: bool = False,
) -> list[dict]:
    """Process all trajectories, grouping by instance_id.

    Returns list of trajectory dicts with step_advantages and step_weights.
    """
    with open(progress_file) as f:
        all_trajs = [json.loads(line) for line in f]

    logger.info(f"Loaded {len(all_trajs)} trajectories")

    by_instance = defaultdict(list)
    for traj in all_trajs:
        by_instance[traj["instance_id"]].append(traj)

    logger.info(f"Found {len(by_instance)} unique instances")

    results = []
    n_with_anchors = 0
    total_anchors = 0

    for instance_id, trajs in by_instance.items():
        augmented = compute_advantages(
            trajs,
            history_length=history_length,
            gamma=gamma,
            step_cost=step_cost,
            alpha=alpha,
            use_episode_base=use_episode_base,
            mode=mode,
            alpha_file=alpha_file,
            alpha_module=alpha_module,
            alpha_entity=alpha_entity,
            commitment_bonus=commitment_bonus,
            final_bonus_scale=final_bonus_scale,
            terminal_only=terminal_only,
            cluster_scheme=cluster_scheme,
        )

        for traj in augmented:
            advs = traj.get("step_advantages", [])
            weights = advantages_to_weights(advs, beta=beta)
            traj["step_weights"] = weights

            nonzero = sum(1 for a in advs if abs(a) > 1e-6)
            if nonzero > 0:
                n_with_anchors += 1
                total_anchors += nonzero

            results.append(traj)

    logger.info(
        f"Computed advantages for {len(results)} trajectories. "
        f"{n_with_anchors} have non-zero step advantages "
        f"(avg {total_anchors / max(n_with_anchors, 1):.1f} anchor matches per traj)"
    )

    return results


def save_results(results: list[dict], output_file: str):
    """Save trajectories with advantages to JSONL."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Saved {len(results)} trajectories to {output_file}")

    credit_file = output_file.replace(".jsonl", "_credits.jsonl")
    with open(credit_file, "w") as f:
        for r in results:
            credits = []
            for si, (adv, w) in enumerate(
                zip(r.get("step_advantages", []), r.get("step_weights", []))
            ):
                credits.append({
                    "step_idx": si,
                    "advantage": round(adv, 6),
                    "weight": round(w, 4),
                })
            f.write(json.dumps({
                "instance_id": r["instance_id"],
                "trajectory_idx": r.get("trajectory_idx", 0),
                "credits": credits,
            }) + "\n")
    logger.info(f"Saved credits to {credit_file}")

    stats_file = output_file.replace(".jsonl", "_stats.json")
    all_advs = []
    all_weights = []
    for r in results:
        all_advs.extend(r.get("step_advantages", []))
        all_weights.extend(r.get("step_weights", []))

    if all_advs:
        stats = {
            "n_trajectories": len(results),
            "n_instances": len(set(r["instance_id"] for r in results)),
            "n_steps_total": len(all_advs),
            "advantage_mean": float(np.mean(all_advs)),
            "advantage_std": float(np.std(all_advs)),
            "advantage_min": float(np.min(all_advs)),
            "advantage_max": float(np.max(all_advs)),
            "weight_mean": float(np.mean(all_weights)),
            "weight_std": float(np.std(all_weights)),
            "n_nonzero_advantages": sum(1 for a in all_advs if abs(a) > 1e-6),
            "pct_nonzero": sum(1 for a in all_advs if abs(a) > 1e-6) / len(all_advs) * 100,
        }
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Stats: {json.dumps(stats, indent=2)}")


def main():
    parser = argparse.ArgumentParser(description="advantage computation for code localization")
    parser.add_argument("--progress_file", type=str, required=True,
                        help="Progress labels file from progress_labeler")
    parser.add_argument("--output_file", type=str,
                        default=str(PROGRESS_DIR / "advantages.jsonl"))
    parser.add_argument("--history_length", type=int, default=2,
                        help="Max history depth K for hierarchical grouping")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="Discount factor for step returns")
    parser.add_argument("--step_cost", type=float, default=0.02,
                        help="Per-step cost penalty in reward")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Weighting exponent for hierarchy levels")
    parser.add_argument("--beta", type=float, default=1.5,
                        help="Advantage-to-weight scaling factor")
    parser.add_argument("--mode", type=str, default="mean_norm",
                        choices=["mean_norm", "mean_std_norm"],
                        help="Normalization mode for advantages")
    parser.add_argument("--no_episode_base", action="store_true",
                        help="Disable episode-level base advantage")
    parser.add_argument("--alpha_file", type=float, default=1.0,
                        help="Weight for file-level reward")
    parser.add_argument("--alpha_module", type=float, default=0.5,
                        help="Weight for module-level reward")
    parser.add_argument("--alpha_entity", type=float, default=0.5,
                        help="Weight for entity (function) level reward")
    parser.add_argument("--terminal_only", action="store_true",
                        help="Only use terminal r^F1; zero per-step delta rewards, step_cost, and DC bonus")
    parser.add_argument("--commitment_bonus", type=float, default=0.5,
                        help="Bonus for committing observed gold to final answer")
    parser.add_argument("--final_bonus_scale", type=float, default=1.0,
                        help="Scaling for final-step F1 bonus")
    parser.add_argument("--cluster_scheme", type=str, default="state",
                        choices=["state", "state_action"],
                        help="state = intended proxy (cluster by history, exclude current action). "
                             "state_action = legacy (same-action grouping).")
    args = parser.parse_args()

    results = process_all_trajectories(
        progress_file=args.progress_file,
        history_length=args.history_length,
        gamma=args.gamma,
        step_cost=args.step_cost,
        alpha=args.alpha,
        beta=args.beta,
        mode=args.mode,
        use_episode_base=not args.no_episode_base,
        alpha_file=args.alpha_file,
        alpha_module=args.alpha_module,
        alpha_entity=args.alpha_entity,
        commitment_bonus=args.commitment_bonus,
        final_bonus_scale=args.final_bonus_scale,
        terminal_only=args.terminal_only,
        cluster_scheme=args.cluster_scheme,
    )

    save_results(results, args.output_file)


if __name__ == "__main__":
    main()
