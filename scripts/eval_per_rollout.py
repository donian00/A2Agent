"""Compute average P / R / F1 per rollout vs one or more GT files.

Walks a multirun base dir for run*_seed*/location/{loc_outputs.jsonl,
merged_loc_outputs_mrr.jsonl} and scores each rollout against each GT file
provided. Skips GT rows where the level's locs are empty (those instances
don't contribute to the level's average).

Usage:
    python scripts/eval_per_rollout.py \\
        --base   result_path/eval_pro_multirun \\
        --gt     evaluation/gt_location/SWE-bench_Pro/test/gt_location.jsonl \\
        [--out summary.csv]
"""
import argparse
import json
import os
from glob import glob
from statistics import mean

LEVEL_KEY = {"file": "found_files", "module": "found_modules", "function": "found_entities"}
LEVEL_K = {"file": 5, "module": 10, "function": 10}


def load_gt(path):
    """instance_id -> {level: [loc, ...]}; skip rows with file_changes=null."""
    gt = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if not d.get("file_changes"):
                continue
            iid = d["instance_id"]
            per_level = {}
            for level in LEVEL_KEY:
                locs = []
                for fc in d["file_changes"]:
                    if level == "file":
                        locs.append(fc["file"])
                    elif level == "module":
                        locs.extend(fc.get("changes", {}).get("edited_modules", []))
                    elif level == "function":
                        locs.extend(fc.get("changes", {}).get("edited_entities", []))

                per_level[level] = locs
            gt[iid] = per_level
    return gt


def clean_preds(preds, level):
    out, seen = [], set()
    for p in (preds or []):
        if isinstance(p, str) and level == "function" and p.endswith(".__init__"):
            p = p[:-len(".__init__")]
        if p not in seen:
            out.append(p); seen.add(p)
    return out


def prf1_at_k(preds, gts, k):
    if not gts:
        return None, None, None
    top = preds[:k]
    hits = sum(1 for p in top if p in gts)
    p = hits / k
    r = hits / len(gts)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def load_preds(loc_file):
    out = {}
    with open(loc_file) as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            out[d["instance_id"]] = d
    return out


def score_rollout(loc_file, gt):
    preds = load_preds(loc_file)
    rows = {}
    for level, key in LEVEL_KEY.items():
        k = LEVEL_K[level]
        Ps, Rs, F1s = [], [], []
        for iid, gt_levels in gt.items():
            gts = gt_levels.get(level, [])
            if not gts:
                continue
            d = preds.get(iid)
            if d is None:
                Ps.append(0.0); Rs.append(0.0); F1s.append(0.0)
                continue
            pl = clean_preds(d.get(key) or [], level)
            p, r, f1 = prf1_at_k(pl, gts, k)
            Ps.append(p); Rs.append(r); F1s.append(f1)
        rows[level] = {
            "N": len(Ps),
            "P": mean(Ps) if Ps else 0.0,
            "R": mean(Rs) if Rs else 0.0,
            "F1": mean(F1s) if F1s else 0.0,
        }
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Multirun base dir with run*_seed*/")
    ap.add_argument("--gt", action="append", required=True,
                    help="GT file path, optionally suffixed ':<label>' (repeatable). "
                         "Example: --gt path/to/gt.jsonl:old --gt path/to/gt_v2.jsonl:new")
    ap.add_argument("--out", default=None, help="Optional CSV output")
    args = ap.parse_args()

    # Parse --gt arguments
    gt_specs = []
    for spec in args.gt:
        if ":" in spec and not os.path.exists(spec):
            path, label = spec.rsplit(":", 1)
        else:
            path = spec
            label = os.path.basename(spec).replace(".jsonl", "")
        gts = load_gt(path)
        gt_specs.append((label, path, gts))
        print(f"[GT] {label:6s}  {len(gts)} labeled instances  ({path})")

    runs = sorted(glob(os.path.join(args.base, "run*_seed*")))
    print(f"\nFound {len(runs)} runs under {args.base}\n")

    rows = []
    for run_dir in runs:
        run_id = os.path.basename(run_dir)
        # Prefer merged_loc_outputs_mrr if present (matches eval script convention)
        loc = os.path.join(run_dir, "location", "merged_loc_outputs_mrr.jsonl")
        if not os.path.exists(loc):
            loc = os.path.join(run_dir, "location", "loc_outputs.jsonl")
        if not os.path.exists(loc):
            print(f"[skip] {run_id}: no loc file")
            continue

        for gt_label, _, gt in gt_specs:
            scores = score_rollout(loc, gt)
            for level, s in scores.items():
                rows.append({
                    "run": run_id, "gt": gt_label, "level": level,
                    "N": s["N"], "P": s["P"], "R": s["R"], "F1": s["F1"],
                })

    # Pretty print as a table per GT, per level
    print(f"{'run':24s}  {'gt':6s}  {'level':9s}   N    P       R       F1")
    print("-" * 76)
    for r in rows:
        print(f"{r['run']:24s}  {r['gt']:6s}  {r['level']:9s}  {r['N']:3d}  "
              f"{r['P']:.4f}  {r['R']:.4f}  {r['F1']:.4f}")

    # Per-(GT, level) averages across runs
    print()
    print("Average across rollouts:")
    print(f"{'gt':6s}  {'level':9s}    avg_P    avg_R    avg_F1")
    print("-" * 50)
    from collections import defaultdict
    grp = defaultdict(list)
    for r in rows:
        grp[(r["gt"], r["level"])].append((r["P"], r["R"], r["F1"]))
    for (gt_label, level), vals in sorted(grp.items()):
        Ps, Rs, F1s = zip(*vals)
        print(f"{gt_label:6s}  {level:9s}  {mean(Ps):.4f}  {mean(Rs):.4f}  {mean(F1s):.4f}")

    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["run", "gt", "level", "N", "P", "R", "F1"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nWrote CSV: {args.out}")


if __name__ == "__main__":
    main()
