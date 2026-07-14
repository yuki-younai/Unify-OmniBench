"""Check whether a benchmark result has already been completed successfully.

Exit codes:
    0 ─ already done (summary.json exists, failed<=10, accuracy>0)
    1 ─ needs to run (no summary.json, or too many failures, or accuracy==0)
    2 ─ error (file exists but JSON is malformed)

Usage:
    python script/check_completed.py --result-dir results/worldsense/Qwen2.5-Omni-7B_vllm_norm
"""
import argparse
import json
import os
import sys


def main() -> None:
    p = argparse.ArgumentParser(description="Check if benchmark result is complete")
    p.add_argument("--result-dir", required=True)
    args = p.parse_args()

    summary_json = os.path.join(args.result_dir, "summary.json")
    if not os.path.isfile(summary_json):
        sys.exit(1)

    try:
        with open(summary_json) as f:
            s = json.load(f)
    except (json.JSONDecodeError, OSError):
        sys.exit(2)

    failed = s.get("failed", 0)
    acc = s.get("accuracy", -1)

    if failed <= 10 and acc > 0:
        print(f"{acc:.1%}")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
