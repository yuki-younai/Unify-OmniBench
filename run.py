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

Results are saved to: ``results/<dataset>/<model_name>/``
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
    p.add_argument("--resume", action="store_true",
                   help="reuse existing results/<dataset>/<model_name>/ dir "
                        "and skip already-finished uids")
    p.add_argument("--rerun-failed", action="store_true",
                   help="re-run only failed / unparsed items in the existing "
                        "results dir")
    p.add_argument("--limit", type=int, default=None,
                   help="only evaluate the first N (post-filter) pending samples "
                        "— for quick sanity checks without waiting for a full run")
    p.add_argument("--task-type", default=None,
                   help="only evaluate samples whose meta.task_type matches this "
                        "value (e.g. 'Event Sequence'); combine with --limit for a "
                        "fast, targeted re-check of a specific failure category")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    dataset_cfg = get_dataset_cfg(args.dataset)
    model_cfg = get_model_cfg(args.backend, model_path=args.model_path)
    # dataset_config.yaml can set use_audio_in_video per dataset (flat field,
    # sibling of data_file/modality) — it wins over the backend yaml's global
    # default. e.g. omnivideobench needs True to match ITS OWN reference
    # implementation (OmniVideoBench/eval/qwenomni_eval.py hardcodes True),
    # while daily_omni/omnibench align with the transformer baseline (False).
    if "use_audio_in_video" in dataset_cfg:
        model_cfg["use_audio_in_video"] = dataset_cfg["use_audio_in_video"]
    if args.vllm_gpu_mem is not None and args.backend == "vllm":
        model_cfg["gpu_memory_utilization"] = args.vllm_gpu_mem
    if args.api_url:
        model_cfg["base_url"] = args.api_url
    if args.api_key:
        model_cfg["api_key"] = args.api_key

    model_name = args.model_name or args.backend
    # run_dir: results/<dataset>/<model_name>_<backend>_<mode>/
    # NOTE: --limit/--task-type get a "_quickcheck" suffix so they write to an
    # isolated results dir — never mixed into (or resumed from) the full run's
    # items.jsonl, which would corrupt the full run's resume bookkeeping / summary.
    run_dir_name = f"{model_name}_{args.backend}_{args.mode}"
    if args.limit or args.task_type:
        run_dir_name += "_quickcheck"
    run_dir = os.path.join("results", args.dataset, run_dir_name)

    gen_cfg = get_generation_cfg()
    if args.temperature is not None:
        gen_cfg["temperature"] = args.temperature
    if args.top_p is not None:
        gen_cfg["top_p"] = args.top_p
    if args.max_new_tokens is not None:
        gen_cfg["max_new_tokens"] = args.max_new_tokens

    # Resolve prompt_template.
    # NOTE: --mode no longer auto-overrides max_new_tokens (that used to force
    # 1024 for "cot"). Token budget is now controlled ONLY by --max-new-tokens
    # (falling back to the hardcoded default of 10 when not passed), so the
    # two concerns — prompt wording vs. token budget — are independent knobs.
    prompt_template = dataset_cfg.get("prompt_template")
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
        "infer_mode": args.mode,
        "limit": args.limit,
        "task_type_filter": args.task_type,
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
    if args.rerun_failed:
        runner.rerun_failed()
    else:
        runner.run()


if __name__ == "__main__":
    main()
