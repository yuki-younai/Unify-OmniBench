"""Unified prompt & media layer.

All model backends should go through this package so that new benchmarks
(e.g. pure-image, pure-audio, text-only) work without modifying model code.
"""
from .media import (
    ModalityClass,
    classify_media,
    filter_media,
    media_description,
    visual_label,
)
from .templates import PromptTemplate

__all__ = [
    "ModalityClass",
    "classify_media",
    "filter_media",
    "media_description",
    "visual_label",
    "PromptTemplate",
]
