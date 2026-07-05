"""Trivial echo model — for smoke-testing the framework without any LLM."""
from __future__ import annotations

import random
from typing import Any, Dict

from ..core.registry import register_model
from ..core.types import InferenceRequest
from .base import BaseModel


@register_model("echo")
class EchoModel(BaseModel):
    """Always returns a configurable letter (default ``"A"``).

    Useful for:
      * end-to-end dry-runs (data loading, parser, report)
      * unit tests of Runner / resume / concurrency
    """
    supports_modalities = ("video", "audio", "image", "text")
    is_thread_safe = True
    supports_batch = False

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.fixed = cfg.get("fixed_answer", "A")
        self.random_mode = bool(cfg.get("random", False))
        self.delay = float(cfg.get("delay_s", 0.0))

    def generate(self, req: InferenceRequest) -> str:
        if self.delay:
            import time
            time.sleep(self.delay)
        if self.random_mode:
            return random.choice(["A", "B", "C", "D"])
        return self.fixed
