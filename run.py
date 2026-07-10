"""Unify-OmniBench entry point.

Dataset paths & per-bench modality live in ``unify_omnibench/config/datasets/``.
Backend defaults live in ``unify_omnibench/config/models/``.
Decoding defaults (max_new_tokens/temperature) are hardcoded in
``unify_omnibench/config/__init__.py::get_generation_cfg`` and overridden via
``--max-new-tokens`` / ``--temperature`` / ``--top-p``.

Usage:
    python run.py --backend openai --dataset daily_omni \
        --model-path gpt-4o --model-name gpt-4o
    python run.py --backend echo   --dataset daily_omni --model-name echo

Results are saved to: ``results/<dataset>/<model_name>_<backend>_<mode>/``

断点续跑/失败重测是自动的、无需任何参数：同一个 run_dir 再跑一次，已成功的样本
会被跳过，失败/未解析出答案的样本会自动重新推理（见 ``Runner.run()``）。
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict

# Trigger dataset / model registrations.
from unify_omnibench import datasets  # noqa: F401
from unify_omnibench import models    # noqa: F401
from unify_omnibench.prompt.templates import _USER_PROMPT_COT  # noqa: F401
from unify_omnibench.config import (
    concurrency_for,
    get_agent_cfg,
    get_dataset_cfg,
    get_generation_cfg,
    get_model_cfg,
    list_backends,
    list_datasets,
)
from unify_omnibench.core.registry import build_dataset, build_model
from unify_omnibench.runner import Runner
from unify_omnibench.utils.io import atomic_write_yaml
from unify_omnibench.utils.logging import get_logger

log = get_logger("run")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unify-OmniBench evaluation runner")
    p.add_argument("--backend", required=True, choices=list_backends(),
                   help="backend yaml under config/models/")
    p.add_argument("--dataset", required=True, choices=list_datasets(),
                   help="dataset yaml under config/datasets/")
    p.add_argument("--model-path", default="",
                   help="loadable model id / local path / HF repo "
                        "(empty = use config/models/<backend>.yaml default)")
    p.add_argument("--model-name", default="",
                   help="short name for the results directory "
                        "(default: backend name)")
    p.add_argument("--mode", default="norm", choices=("norm", "cot"),
                   help="inference mode: norm (direct answer) | cot (chain-of-thought)")
    p.add_argument("--run-mode", default="direct", choices=("direct", "react"),
                   dest="run_mode",
                   help="evaluation mode: direct (single-shot) | react (multi-turn agent)")

    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--temperature", type=float, default=None,
                   help="override default temperature (0.0)")
    p.add_argument("--top-p", type=float, default=None, dest="top_p",
                   help="override default top_p")
    p.add_argument("--max-new-tokens", type=int, default=None, dest="max_new_tokens",
                   help="override default max_new_tokens (10); single source of "
                        "truth for token budget — --mode no longer auto-sets this")
    p.add_argument("--vllm-gpu-mem", type=float, default=None,
                   help="override vLLM gpu_memory_utilization (default: 0.95)")
    p.add_argument("--api-url", default="",
                   help="override base_url for api backends (e.g. http://localhost:8001/v1)")
    p.add_argument("--api-key", default="",
                   help="API key (empty = local vLLM / non-empty = cloud API)")
    p.add_argument("--limit", type=int, default=None,
                   help="only evaluate the first N (post-filter) pending samples "
                        "— for quick sanity checks without waiting for a full run")
    p.add_argument("--task-type", default=None,
                   help="only evaluate samples whose meta.task_type matches this "
                        "value (e.g. 'Event Sequence'); combine with --limit for a "
                        "fast, targeted re-check of a specific failure category")
    p.add_argument("--shard-id", type=int, default=None,
                   help="0-based shard index (0..num_shards-1), used with "
                        "--num-shards for multi-worker parallel eval")
    p.add_argument("--num-shards", type=int, default=None,
                   help="total number of shards, used with --shard-id")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    dataset_cfg = get_dataset_cfg(args.dataset)
    model_cfg = get_model_cfg(args.backend, model_path=args.model_path)
    # dataset_config.yaml 可以按数据集覆盖 use_audio_in_video（优先于 backend yaml
    # 的全局默认值），例如 omnivideobench 需要 True 以对齐官方参考实现。
    if "use_audio_in_video" in dataset_cfg:
        model_cfg["use_audio_in_video"] = dataset_cfg["use_audio_in_video"]
    # 同理，数据集级别的抽帧参数（fps/max_frames/min_frames）覆盖 backend 的全局默认值。
    if "video" in dataset_cfg:
        video_override = dict(dataset_cfg["video"] or {})
        model_cfg_video = dict(model_cfg.get("video") or {})
        model_cfg_video.update(video_override)
        model_cfg["video"] = model_cfg_video
        # 同步 flat 字段，作为 qwen25omni.py::_cap_video_frames 的事后裁剪安全网
        if "max_frames" in video_override:
            model_cfg["max_frames"] = int(video_override["max_frames"])
    if args.vllm_gpu_mem is not None and args.backend == "vllm":
        model_cfg["gpu_memory_utilization"] = args.vllm_gpu_mem
    if args.backend == "vllm":
        # max_num_seqs（vLLM 引擎真实并发上限）直接跟 --workers 走同一个值，
        # 跟下面 concurrency.batch_size 天然一致，不用单独维护。
        model_cfg["max_num_seqs"] = args.workers
    if args.api_url:
        model_cfg["base_url"] = args.api_url
    if args.api_key:
        model_cfg["api_key"] = args.api_key

    model_name = args.model_name or args.backend
    # results/<dataset>/<model_name>_<backend>_<mode>/；--limit/--task-type 抽查
    # 加 _quickcheck 后缀，写到独立目录，不混进完整评测的 items.jsonl。
    # shard worker 写到 shard_N/ 子目录，跑完由合并脚本聚合成父目录。
    run_dir_name = f"{model_name}_{args.backend}_{args.mode}"
    if args.limit or args.task_type:
        run_dir_name += "_quickcheck"
    run_dir = os.path.join("results", args.dataset, run_dir_name)
    if args.shard_id is not None and args.num_shards is not None:
        run_dir = os.path.join(run_dir, f"shard_{args.shard_id}")

    gen_cfg = get_generation_cfg()
    if args.temperature is not None:
        gen_cfg["temperature"] = args.temperature
    if args.top_p is not None:
        gen_cfg["top_p"] = args.top_p
    if args.max_new_tokens is not None:
        gen_cfg["max_new_tokens"] = args.max_new_tokens

    # prompt 全部由 dataset_config.yaml 定义（--mode cot 只换 user prompt 文案）。
    prompt_template = dataset_cfg.get("prompt_template")
    system_prompt = dataset_cfg.get("system_prompt")
    if args.mode == "cot":
        prompt_template = _USER_PROMPT_COT

    cfg: Dict[str, Any] = {
        "run_name": f"{args.dataset}/{model_name}",
        "run_dir": run_dir,
        "modality_mode": dataset_cfg.get("modality", "av"),
        "dataset": dataset_cfg,
        "model": model_cfg,
        "generation": gen_cfg,
        "prompt_template": prompt_template,
        "system_prompt": system_prompt,
        "infer_mode": args.mode,
        "run_mode": args.run_mode,
        "react": get_agent_cfg(args.dataset),

        # react 环境变量覆盖（从 eval_react.sh 传入）
        "_react_env_overrides": {
            k: os.environ[k] for k in ("MAX_STEPS_OVERRIDE",)
            if k in os.environ
        },
        "limit": args.limit,
        "task_type_filter": args.task_type,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "concurrency": {
            "mode": concurrency_for(args.backend),
            "max_workers": args.workers,
            "batch_size": args.workers,
        },
    }

    log.info("run_dir = %s", run_dir)
    os.makedirs(run_dir, exist_ok=True)
    atomic_write_yaml(os.path.join(run_dir, "run_config.yaml"), cfg)

    ds = build_dataset(cfg["dataset"])
    md = build_model(cfg["model"])
    runner = Runner(ds, md, cfg)
    runner.run()


if __name__ == "__main__":
    main()
