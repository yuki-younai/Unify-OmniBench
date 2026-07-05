"""Global registries for dataset adapters and models."""
from __future__ import annotations

from typing import Any, Callable, Dict, Type

DATASET_REGISTRY: Dict[str, Type] = {}
MODEL_REGISTRY: Dict[str, Type] = {}


def register_dataset(name: str) -> Callable[[Type], Type]:
    def deco(cls: Type) -> Type:
        if name in DATASET_REGISTRY:
            raise ValueError(f"Dataset '{name}' already registered: {DATASET_REGISTRY[name]}")
        cls.name = name
        DATASET_REGISTRY[name] = cls
        return cls
    return deco


def register_model(name: str) -> Callable[[Type], Type]:
    def deco(cls: Type) -> Type:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered: {MODEL_REGISTRY[name]}")
        cls.name = name
        MODEL_REGISTRY[name] = cls
        return cls
    return deco


def build_dataset(cfg: Dict[str, Any]):
    name = cfg["name"]
    if name not in DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {sorted(DATASET_REGISTRY)}"
        )
    return DATASET_REGISTRY[name](cfg)


def build_model(cfg: Dict[str, Any]):
    name = cfg["name"]
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](cfg)
