#!/usr/bin/env python3
"""聚合 results/ 下所有 summary.json → results/summary.md

透视表格式（每个 benchmark 一张表，行=模型，列=后端）：
    | Model | transformer | vllm | openai |
    |---|---|---|---|
    | Qwen2.5-Omni-7B | 62.0% | 61.5% | 55.1% |

Usage:
    python3 script/aggregate_results.py
"""

import json
import os
import re
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def parse_run_name(run_name: str):
    """Parse '{model_name}_{backend}_{mode}' → (model, backend, mode)."""
    # e.g. "Qwen2.5-Omni-7B_vllm_norm" → ("Qwen2.5-Omni-7B", "vllm", "norm")
    # Handle quickcheck suffix: "Qwen2.5-Omni-7B_vllm_norm_quickcheck"
    if run_name.endswith("_quickcheck"):
        run_name = run_name[:-len("_quickcheck")]

    # Split from right: mode, backend, remaining = model
    parts = run_name.rsplit("_", 2)
    if len(parts) == 3:
        model, backend, mode = parts
    elif len(parts) == 2:
        model, backend = parts
        mode = "norm"
    else:
        model = run_name
        backend = "?"
        mode = "?"
    return model, backend, mode


def load_all_summaries():
    """Scan results/ → {(benchmark, model, backend, mode): summary_dict}."""
    all_data = {}
    if not os.path.isdir(RESULTS_DIR):
        return all_data
    for bench in sorted(os.listdir(RESULTS_DIR)):
        bench_dir = os.path.join(RESULTS_DIR, bench)
        if not os.path.isdir(bench_dir):
            continue
        for run in sorted(os.listdir(bench_dir)):
            run_dir = os.path.join(bench_dir, run)
            sp = os.path.join(run_dir, "summary.json")
            if not os.path.isfile(sp):
                continue
            try:
                summary = json.load(open(sp, encoding="utf-8"))
            except (json.JSONDecodeError, IOError):
                continue
            model, backend, mode = parse_run_name(run)
            all_data[(bench, model, backend, mode)] = summary
    return all_data


def build_markdown(data: dict) -> str:
    lines = [
        "# Unify-OmniBench 评测汇总",
        f"",
        f"> 自动生成, 最后更新: `{__import__('time').strftime('%Y-%m-%d %H:%M')}`",
        "",
    ]

    # ── 收集所有 benchmark / model / backend ──────────────────
    benchmarks = sorted(set(k[0] for k in data))
    all_models = sorted(set(k[1] for k in data), reverse=True)
    all_backends = sorted(set(k[2] for k in data))
    all_modes = sorted(set(k[3] for k in data))

    # ── 总表：行=模型×backend，列=benchmark ──────────────────
    # 收集所有 row keys: "模型 backend"
    row_keys = sorted(set(
        f"{model} {back}"
        for (b, model, back, m) in data
        if m == "norm"
    ))
    # 按模型分组排序：7B 在前
    def _row_rank(rk):
        m, b = rk.split(" ", 1)
        return (0 if "7B" in m else 1, m, b)
    row_keys = sorted(row_keys, key=_row_rank)

    lines.append("## 总表")
    lines.append("")
    header = "| Model / Backend | " + " | ".join(bench for bench in benchmarks) + " |"
    sep = "|" + "---|" * (len(benchmarks) + 1)
    lines.append(header)
    lines.append(sep)

    for rk in row_keys:
        model, back = rk.split(" ", 1)
        cells = [f"**{rk}**"]
        for bench in benchmarks:
            key = (bench, model, back, "norm")
            row = data.get(key)
            if row:
                cells.append(f"{row['accuracy']:.1%}")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # ── 详细表（原样保留） ──────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## 详细记录")
    lines.append("")
    for bench in benchmarks:
        lines.append(f"### {bench}")
        lines.append("")
        lines.append("| Model | Backend | Mode | Accuracy | Valid | Failed | Parse Fail | Total |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")

        for key in sorted(data):
            b, model, back, mode = key
            if b != bench:
                continue
            s = data[key]
            acc = s.get("accuracy", 0)
            lines.append(
                f"| {model} | {back} | {mode} | {acc:.2%} "
                f"| {s.get('valid', 0)} | {s.get('failed', 0)} "
                f"| {s.get('parse_failed', 0)} | {s.get('total', 0)} |"
            )
        lines.append("")

    return "\n".join(lines)


def main():
    data = load_all_summaries()
    if not data:
        print("No summary.json files found under results/")
        return

    md = build_markdown(data)
    out_path = os.path.join(RESULTS_DIR, "summary.md")

    # 写之前备份旧版本
    if os.path.exists(out_path):
        bak = out_path + ".bak"
        with open(out_path) as f_in, open(bak, "w") as f_out:
            f_out.write(f_in.read())

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"✅ {out_path}")

    # 顺便打印简要统计
    benchmarks = sorted(set(k[0] for k in data))
    for bench in benchmarks:
        print(f"\n[{bench}]")
        for model in sorted(set(k[1] for k in data), reverse=True):
            line_parts = [f"  {model}"]
            for back in sorted(set(k[2] for k in data)):
                key = (bench, model, back, "norm")
                if key in data:
                    line_parts.append(f"{back}={data[key]['accuracy']:.1%}")
            print("  ".join(line_parts))


if __name__ == "__main__":
    main()
