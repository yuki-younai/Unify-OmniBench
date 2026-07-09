"""Merge multi-worker shard results into the parent results directory.

Usage:
    python script/merge_shards.py \\
        --result-dir results/omnivideobench/Qwen2.5-Omni-7B_vllm_norm \\
        --num-shards 2 \\
        --dataset omnivideobench
"""
from __future__ import annotations

import argparse
import os
import sys

from unify_omnibench.eval.report import write_summary
from unify_omnibench.utils.io import load_jsonl, rewrite_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description="Merge multi-worker shard results")
    p.add_argument("--result-dir", required=True,
                   help="parent results directory (e.g. results/dataset/model_backend_mode/)")
    p.add_argument("--num-shards", type=int, required=True,
                   help="total number of shards/workers")
    p.add_argument("--dataset", required=True,
                   help="dataset name (for summary)")
    args = p.parse_args()

    parent = args.result_dir
    if not os.path.isdir(parent):
        sys.exit(f"result dir not found: {parent}")

    items_path = os.path.join(parent, "items.jsonl")
    failed_path = os.path.join(parent, "failed.jsonl")

    all_items = []
    for i in range(args.num_shards):
        shard_path = os.path.join(parent, f"shard_{i}", "items.jsonl")
        if not os.path.exists(shard_path):
            print(f"  [merge] shard_{i}: SKIP (no items.jsonl)")
            continue
        recs = load_jsonl(shard_path)
        all_items.extend(recs)
        print(f"  [merge] shard_{i}: {len(recs)} records")

    if not all_items:
        sys.exit("no shard data found — did all workers fail?")

    # dedup (each shard is already compacted internally; this is a safety net)
    seen = {}
    for rec in all_items:
        uid = rec.get("uid")
        if uid is not None:
            seen[uid] = rec
    deduped = list(seen.values())
    dropped = len(all_items) - len(deduped)
    if dropped:
        print(f"  [merge] dropped {dropped} duplicate record(s) across shards")

    rewrite_jsonl(items_path, deduped)

    failed = [r for r in deduped if r.get("error")]
    if failed:
        rewrite_jsonl(failed_path, failed)
    elif os.path.exists(failed_path):
        os.remove(failed_path)

    summary = write_summary(items_path, out_dir=parent, dataset_name=args.dataset)
    print(f"  [merge] total={summary['total']} accuracy={summary['accuracy']:.2%}")


if __name__ == "__main__":
    main()
