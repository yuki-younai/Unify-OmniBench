"""Merge multi-worker shard results into the parent results directory.

Usage:
    python script/merge_shards.py \\
        --result-dir results/omnivideobench/Qwen2.5-Omni-7B_vllm_norm \\
        --num-shards 2 \\
        --dataset omnivideobench \\
        --cleanup                # remove shard_N/ dirs after merge
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

from unify_omnibench.eval.report import write_summary
from unify_omnibench.utils.io import load_jsonl, rewrite_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description="Merge multi-worker shard results")
    p.add_argument("--result-dir", required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--cleanup", action="store_true",
                   help="remove shard_N/ directories after merge")
    args = p.parse_args()

    parent = args.result_dir
    if not os.path.isdir(parent):
        sys.exit(f"result dir not found: {parent}")

    items_path = os.path.join(parent, "items.jsonl")
    failed_path = os.path.join(parent, "failed.jsonl")

    # Load existing parent results first (from a previous run), then
    # merge in new shard results.  New records override old by uid so
    # that partial re-runs (retry failed samples) don't discard the
    # already-completed results.
    if os.path.exists(items_path):
        existing = load_jsonl(items_path)
        print(f"  [merge] loaded {len(existing)} existing records from "
              f"parent items.jsonl")
    else:
        existing = []

    new_recs = []
    for i in range(args.num_shards):
        shard_path = os.path.join(parent, f"shard_{i}", "items.jsonl")
        if not os.path.exists(shard_path):
            print(f"  [merge] shard_{i}: SKIP (no items.jsonl)")
            continue
        recs = load_jsonl(shard_path)
        new_recs.extend(recs)
        print(f"  [merge] shard_{i}: {len(recs)} records")

    if not new_recs and not existing:
        sys.exit("no shard data found — did all workers fail?")

    if not new_recs:
        print("  [merge] all shards empty — reusing existing parent "
              f"results ({len(existing)} records)")

    # dedup: new records override old ones by uid
    seen = {}
    for rec in existing:
        uid = rec.get("uid")
        if uid is not None:
            seen[uid] = rec
    for rec in new_recs:
        uid = rec.get("uid")
        if uid is not None:
            seen[uid] = rec
    deduped = list(seen.values())
    updated = sum(1 for r in new_recs if r.get("uid") in seen)

    rewrite_jsonl(items_path, deduped)

    failed = [r for r in deduped if r.get("error")]
    if failed:
        rewrite_jsonl(failed_path, failed)
    elif os.path.exists(failed_path):
        os.remove(failed_path)

    summary = write_summary(items_path, out_dir=parent, dataset_name=args.dataset)
    print(f"  [merge] total={summary['total']} accuracy={summary['accuracy']:.2%}")

    # Merge per-shard trajectory files (Agent ReAct mode) into the parent
    # trajectories/ dir BEFORE cleanup — otherwise --cleanup's rmtree()
    # below silently deletes every shard's trajectories/*.json / *.html
    # with no trace, since items.jsonl only stores the parsed answer/
    # history, not the saved-trajectory file paths.
    traj_out = os.path.join(parent, "trajectories")
    n_traj = 0
    for i in range(args.num_shards):
        shard_traj = os.path.join(parent, f"shard_{i}", "trajectories")
        if not os.path.isdir(shard_traj):
            continue
        os.makedirs(traj_out, exist_ok=True)
        for fname in os.listdir(shard_traj):
            shutil.copy2(os.path.join(shard_traj, fname), os.path.join(traj_out, fname))
            n_traj += 1
    if n_traj:
        print(f"  [merge] trajectories: copied {n_traj} files from shards into {traj_out}")

    if args.cleanup:
        for i in range(args.num_shards):
            d = os.path.join(parent, f"shard_{i}")
            if os.path.isdir(d):
                shutil.rmtree(d)
                print(f"  [cleanup] removed {d}")


if __name__ == "__main__":
    main()
