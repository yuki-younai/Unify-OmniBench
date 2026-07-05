"""YAML config loader with simple ${ENV} expansion and _base_ inheritance."""
from __future__ import annotations

import os
import re
from typing import Any, Dict

import yaml

_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)(?::([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} or ${VAR:default} in strings."""
    if isinstance(value, str):
        def _sub(m):
            var = m.group(1)
            default = m.group(2) or ""
            return os.environ.get(var, default)
        return _ENV_RE.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _deep_update(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_update(a[k], v)
        else:
            a[k] = v
    return a


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config. Supports a single-level `_base_: path/to/base.yaml`."""
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base = cfg.pop("_base_", None)
    if base:
        base_path = base if os.path.isabs(base) else os.path.join(os.path.dirname(path), base)
        with open(base_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f) or {}
        cfg = _deep_update(base_cfg, cfg)

    cfg = _expand_env(cfg)
    return cfg
