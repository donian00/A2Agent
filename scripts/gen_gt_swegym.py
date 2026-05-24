"""Generate evaluation/gt_location/SWE-Gym/train/gt_location.jsonl for SWE-Gym.

For each instance:
  - clone (or local-clone from cache) the repo at base_commit
  - parse the patch hunks
  - map deleted/added lines to (class, function) → edited_modules / edited_entities
  - write {instance_id, file_changes, repo, base_commit, problem_statement, patch}

Run:
    python scripts/gen_gt_swegym.py \
        --dataset SWE-Gym/SWE-Gym --split train \
        --output_dir evaluation/gt_location \
        --repo_base_dir playground/swegym_gt \
        --num_processes 16
"""
import argparse
import json
import logging
import logging.handlers
import os
import shutil
import subprocess
import sys
import uuid
from queue import Empty

import torch.multiprocessing as mp
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.benchmark.gen_oracle_locations import extract_module_from_patch
from util.benchmark.setup_repo import setup_repo


def _worker(rank, queue, log_queue, lock, output_file, repo_playground_root, max_edit_file_num):
    qh = logging.handlers.QueueHandler(log_queue)
    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(qh)
    logger.setLevel(logging.WARNING)

    n_done = n_err = 0
    while True:
        try:
            instance = queue.get_nowait()
        except Empty:
            break

        playground = os.path.join(repo_playground_root, str(uuid.uuid4()))
        try:
            os.makedirs(playground, exist_ok=True)
            repo_dir = setup_repo(
                instance_data=instance,
                repo_base_dir=playground,
                dataset=None, split=None,
            )
            file_changes = extract_module_from_patch(
                instance, repo_dir,
                max_edit_file_num=max_edit_file_num,
                logger=logger, rank=rank,
            )
            with lock:
                with open(output_file, "a") as f:
                    f.write(json.dumps({
                        "instance_id": instance["instance_id"],
                        "file_changes": file_changes,  # may be None for filtered cases
                        "repo": instance["repo"],
                        "base_commit": instance.get("base_commit"),
                        "problem_statement": instance.get("problem_statement", ""),
                        "patch": instance.get("patch", ""),
                    }) + "\n")
            n_done += 1
        except FileNotFoundError as e:
            logger.warning(f"rank {rank} FileNotFoundError: {e}")
            n_err += 1
        except subprocess.CalledProcessError as e:
            logger.warning(f"rank {rank} CalledProcessError: {e}")
            n_err += 1
        except Exception as e:
            logger.warning(f"rank {rank} Error on {instance.get('instance_id')}: {type(e).__name__}: {e}")
            n_err += 1
        finally:
            if os.path.exists(playground):
                shutil.rmtree(playground, ignore_errors=True)

    print(f"[rank {rank}] done: {n_done} ok, {n_err} errors")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="SWE-Gym/SWE-Gym")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--output_dir", type=str, default="evaluation/gt_location")
    p.add_argument("--repo_base_dir", type=str, default="playground/swegym_gt")
    p.add_argument("--num_processes", type=int, default=16)
    p.add_argument("--max_edit_file_num", type=int, default=10,
                   help="Skip instances editing more than this many files.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    out_subdir = os.path.join(args.output_dir, args.dataset.split("/")[-1], args.split)
    os.makedirs(out_subdir, exist_ok=True)
    output_file = os.path.join(out_subdir, "gt_location.jsonl")

    logger.info(f"Loading dataset {args.dataset}/{args.split} ...")
    ds = load_dataset(args.dataset, split=args.split)
    logger.info(f"Total instances: {len(ds)}")

    # Skip instances we've already processed
    done = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["instance_id"])
    logger.info(f"Already done: {len(done)}")

    todo = [d for d in ds if d["instance_id"] not in done]
    logger.info(f"To process: {len(todo)}")
    if not todo:
        logger.info("Nothing to do.")
        return

    os.makedirs(args.repo_base_dir, exist_ok=True)

    manager = mp.Manager()
    queue = manager.Queue()
    log_queue = manager.Queue()
    lock = manager.Lock()
    for d in todo:
        queue.put(dict(d))

    queue_listener = logging.handlers.QueueListener(log_queue, *logging.getLogger().handlers)
    queue_listener.start()

    nprocs = min(args.num_processes, len(todo))
    logger.info(f"Spawning {nprocs} workers ...")
    mp.spawn(
        _worker,
        nprocs=nprocs,
        args=(queue, log_queue, lock, output_file, args.repo_base_dir, args.max_edit_file_num),
        join=True,
    )
    queue_listener.stop()

    # Final stats
    n = sum(1 for _ in open(output_file))
    n_with_changes = 0
    n_modules = 0
    n_entities = 0
    with open(output_file) as f:
        for line in f:
            d = json.loads(line)
            fcs = d.get("file_changes")
            if fcs:
                n_with_changes += 1
                for fc in fcs:
                    chg = fc.get("changes", {}) or {}
                    n_modules += len(chg.get("edited_modules", []) or [])
                    n_entities += len(chg.get("edited_entities", []) or [])

    logger.info(f"Wrote {n} lines to {output_file}")
    logger.info(f"  with file_changes: {n_with_changes}")
    logger.info(f"  total edited_modules: {n_modules}, total edited_entities: {n_entities}")


if __name__ == "__main__":
    main()
