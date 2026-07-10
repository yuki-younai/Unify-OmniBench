"""YAML-driven configuration loader.

Layout::

    config/
      dataset_config.yaml      # single file: {root, datasets: {key: {data_file, modality}}}
      models/<key>.yaml        # one file per backend

To add a new benchmark, add an entry under ``datasets:`` in
``dataset_config.yaml`` — no Python change needed. All ``data_file`` /
``media_root`` are resolved relative to the single ``root`` path.

解码默认参数（``max_new_tokens`` / ``temperature``）硬编码在
:func:`get_generation_cfg`，通过 ``run.py`` 的 ``--max-new-tokens`` /
``--temperature`` / ``--top-p`` 覆盖（对应 ``eval.sh`` 的
``MAX_NEW_TOKENS`` / ``TEMPERATURE`` / ``TOP_P``）。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import yaml

_CFG_DIR = os.path.dirname(os.path.abspath(__file__))
_DATASET_CONFIG_FILE = os.path.join(_CFG_DIR, "dataset_config.yaml")
_MODELS_DIR = os.path.join(_CFG_DIR, "models")

# Hardcoded decoding defaults (no longer backed by a YAML file).
_DEFAULT_GENERATION_CFG: Dict[str, Any] = {
    "max_new_tokens": 10,
    "temperature": 0.0,
}

# Env-var name per backend for injecting API keys.
_API_KEY_ENV: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
}


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_dataset_config() -> Dict[str, Any]:
    if not os.path.exists(_DATASET_CONFIG_FILE):
        return {"root": "", "datasets": {}}
    return _load_yaml(_DATASET_CONFIG_FILE)


def _list_keys(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(folder)
        if f.endswith((".yaml", ".yml")) and not f.startswith("_")
    )


# ------------------------------------------------------------- public API
def list_datasets() -> List[str]:
    return sorted(_load_dataset_config().get("datasets", {}).keys())


def list_backends() -> List[str]:
    return _list_keys(_MODELS_DIR)


def get_dataset_cfg(key: str) -> Dict[str, Any]:
    """Build a dataset's runtime config from ``dataset_config.yaml``.

    Resolves ``data_file`` to an absolute path under ``root``, and passes
    ``root`` through as ``media_root`` (used by ``UnifiedAdapter`` to
    resolve each Sample's relative video/audio/image paths).
    """
    cfg_all = _load_dataset_config()
    root = cfg_all.get("root", "")
    entries = cfg_all.get("datasets", {})
    if key not in entries:
        raise KeyError(
            f"Unknown dataset '{key}'. Available: {list_datasets()}"
        )
    entry = dict(entries[key])
    data_file = entry.get("data_file", "")
    entry["data_file"] = os.path.join(root, data_file) if root and not os.path.isabs(data_file) else data_file
    entry.setdefault("media_root", root)
    entry.setdefault("name", key)
    return entry


def _detect_gpu_count() -> int:
    """Return the number of visible GPUs.

    Respects ``CUDA_VISIBLE_DEVICES``:
    - comma-separated list → len of tokens (e.g. ``0,1,2,3`` → 4)
    - empty → 1
    - not set → tries ``torch.cuda.device_count()``, falls back to 1.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        # CUDA_VISIBLE_DEVICES="0,1,2,3" or "0,1" or "0" or "GPU-xxx,GPU-yyy"
        return len(visible.split(","))

    try:
        import torch
        n = torch.cuda.device_count()
        if n > 0:
            return n
    except Exception:
        pass
    return 1


def get_model_cfg(backend: str, model_path: str = "") -> Dict[str, Any]:
    """Load a backend's YAML config.

    For ``vllm`` the ``tensor_parallel_size`` is auto-set to the number of
    visible GPUs (via ``CUDA_VISIBLE_DEVICES`` or ``torch.cuda.device_count``).

    Args:
        backend: yaml filename under ``config/models/`` (without extension).
        model_path: optional override of the loadable id/path.
    """
    path = os.path.join(_MODELS_DIR, f"{backend}.yaml")
    if not os.path.exists(path):
        raise KeyError(
            f"Unknown backend '{backend}'. Available: {list_backends()}"
        )
    cfg = _load_yaml(path)

    if model_path:
        # qwen_omni uses ``model_name_or_path``; others use ``model``.
        if "model_name_or_path" in cfg:
            cfg["model_name_or_path"] = model_path
        elif "base_url" in cfg:
            # API backends (openai, gemini, etc.): ``model`` is an identifier
            # sent in the API request (e.g. "Qwen2.5-Omni-7B"), NOT a local
            # filesystem path. Overwriting it with ``model_path`` would
            # send a full path like "/data/models/Qwen...-7B" as the model
            # name in the HTTP call — the server doesn't know that name.
            pass
        else:
            cfg["model"] = model_path

    # auto-detect tensor_parallel_size for vllm
    if backend == "vllm":
        cfg["tensor_parallel_size"] = _detect_gpu_count()

    env_key = _API_KEY_ENV.get(backend)
    if env_key:
        cfg["api_key"] = os.environ.get(env_key, "")

    return cfg


def concurrency_for(backend: str) -> str:
    """Read ``concurrency_mode`` from the backend's YAML (default: thread)."""
    cfg = get_model_cfg(backend)
    return cfg.get("concurrency_mode", "thread")


def get_generation_cfg() -> Dict[str, Any]:
    """Return decoding defaults (max_new_tokens/temperature).

    No YAML backing anymore — this is a plain hardcoded default; callers
    (``run.py``) override individual keys via CLI flags.
    """
    return dict(_DEFAULT_GENERATION_CFG)


def get_agent_cfg(bench: str) -> Dict[str, Any]:
    """Load agent ReAct config from config/agent.yaml, merged per-benchmark."""
    path = os.path.join(_CFG_DIR, "agent.yaml")
    if not os.path.exists(path):
        return {}
    raw = _load_yaml(path)
    defaults = raw.get("default", {})
    bench_cfg = raw.get("benchmarks", {}).get(bench, {})
    merged = dict(defaults)
    _deep_update(merged, bench_cfg)
    return merged


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


__all__ = [
    "list_datasets",
    "list_backends",
    "get_dataset_cfg",
    "get_model_cfg",
    "concurrency_for",
    "get_generation_cfg",
    "get_agent_cfg",
]
