"""Unify-OmniBench: Unified evaluation framework for multimodal Omni benchmarks."""

__version__ = "0.1.0"

from .core.types import Sample, MediaRef, InferenceRequest, InferenceResult  # noqa: F401
from .core.registry import (  # noqa: F401
    register_dataset,
    register_model,
    build_dataset,
    build_model,
    DATASET_REGISTRY,
    MODEL_REGISTRY,
)
