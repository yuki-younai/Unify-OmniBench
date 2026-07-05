"""Summary + per-bucket breakdown report writer."""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Dict, List

from ..utils.io import atomic_write_json, atomic_write_text, load_jsonl

# Fields used for breakdown analysis (best-effort; missing fields are skipped)
_BREAKDOWN_KEYS = (
    "task_type",
    "question_type",
    "audio_type",
    "video_type",
    "video_category",
    "video_duration",
    "duration_bucket",
    "modality_mode",
)


def _bucket_duration(sec) -> str:
    try:
        s = float(sec)
    except (TypeError, ValueError):
        return ""
    if s <= 0:
        return ""
    if s < 60:
        return "0-1min"
    if s < 5 * 60:
        return "1-5min"
    if s < 15 * 60:
        return "5-15min"
    if s < 30 * 60:
        return "15-30min"
    if s < 60 * 60:
        return "30-60min"
    return ">=60min"


def write_summary(items_path: str, out_dir: str, dataset_name: str) -> Dict[str, Any]:
    items = load_jsonl(items_path)
    total = len(items)
    failed = sum(1 for x in items if x.get("error"))
    parsed_failed = sum(1 for x in items if not x.get("error") and not x.get("parsed_answer"))
    valid = [x for x in items if not x.get("error") and x.get("parsed_answer")]
    correct = sum(1 for x in valid if x.get("is_correct"))

    by: Dict[str, Dict[str, int]] = defaultdict(lambda: {"n": 0, "c": 0})
    for x in valid:
        meta = x.get("meta") or {}
        # Auto-add duration bucket if duration_s present
        if meta.get("duration_s") is not None and "duration_bucket" not in meta:
            meta = dict(meta)
            meta["duration_bucket"] = _bucket_duration(meta["duration_s"])
        for key in _BREAKDOWN_KEYS:
            v = meta.get(key)
            if v is None or v == "":
                continue
            bk = f"{key}={v}"
            by[bk]["n"] += 1
            by[bk]["c"] += int(bool(x.get("is_correct")))

    accuracy = (correct / len(valid)) if valid else 0.0
    summary: Dict[str, Any] = {
        "dataset": dataset_name,
        "total": total,
        "failed": failed,
        "parse_failed": parsed_failed,
        "valid": len(valid),
        "correct": correct,
        "accuracy": accuracy,
        "breakdown": {
            k: {
                "n": v["n"],
                "c": v["c"],
                "acc": (v["c"] / v["n"]) if v["n"] else 0.0,
            }
            for k, v in by.items()
        },
    }
    atomic_write_json(os.path.join(out_dir, "summary.json"), summary)

    # Markdown report
    lines: List[str] = [
        f"# Summary: {dataset_name}",
        "",
        f"- total = **{total}**",
        f"- valid = **{len(valid)}** (parsed)",
        f"- failed (errors) = {failed}",
        f"- parse-failed (no letter extracted) = {parsed_failed}",
        f"- **accuracy = {accuracy:.2%}**  ({correct}/{len(valid)})",
        "",
        "## Breakdown",
        "",
        "| key | acc | correct / total |",
        "|---|---:|---:|",
    ]
    for k, v in sorted(summary["breakdown"].items()):
        lines.append(f"| `{k}` | {v['acc']:.2%} | {v['c']} / {v['n']} |")
    atomic_write_text(os.path.join(out_dir, "summary.md"), "\n".join(lines) + "\n")
    return summary


def write_leaderboard(run_dirs: List[str], out_path: str) -> str:
    """Aggregate multiple `summary.json` files into one comparison markdown."""
    rows = []
    for d in run_dirs:
        sp = os.path.join(d, "summary.json")
        if not os.path.exists(sp):
            continue
        import json
        with open(sp, "r", encoding="utf-8") as f:
            s = json.load(f)
        rows.append(
            {
                "run": os.path.basename(d.rstrip("/")),
                "dataset": s.get("dataset", "?"),
                "accuracy": s.get("accuracy", 0.0),
                "valid": s.get("valid", 0),
                "failed": s.get("failed", 0),
                "total": s.get("total", 0),
            }
        )
    rows.sort(key=lambda r: (-r["accuracy"], r["run"]))
    lines = [
        "# Leaderboard",
        "",
        "| run | dataset | accuracy | valid | failed | total |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['dataset']} | {r['accuracy']:.2%} "
            f"| {r['valid']} | {r['failed']} | {r['total']} |"
        )
    atomic_write_text(out_path, "\n".join(lines) + "\n")
    return out_path
