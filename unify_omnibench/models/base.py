"""BaseModel interface — implement ``generate`` (and optionally ``generate_batch``)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from ..core.types import InferenceRequest, Modality


class BaseModel(ABC):
    """Concrete models must:
      * Inherit and implement :meth:`generate`.
      * Optionally implement :meth:`generate_batch` for vLLM-style backends.
      * Set ``is_thread_safe`` / ``supports_batch`` to tell :class:`Runner`
        which concurrency mode is safe.
    """
    name: str = ""
    supports_modalities: Tuple[Modality, ...] = ()
    is_thread_safe: bool = False
    supports_batch: bool = False

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def load(self) -> None:
        """Heavy initialization. Called once before any generate()."""
        return None

    def close(self) -> None:
        """Optional cleanup."""
        return None

    @abstractmethod
    def generate(self, req: InferenceRequest) -> str: ...

    def generate_batch(self, reqs: List[InferenceRequest]) -> List[str]:
        return [self.generate(r) for r in reqs]

    # -------- helpers for child classes
    @staticmethod
    def format_choices(choices) -> str:
        if isinstance(choices, list):
            return "\n".join(str(c) for c in choices)
        return str(choices)

    @classmethod
    def default_prompt(cls, question: str, choices) -> str:
        return (
            "Answer the multiple-choice question. Reply with one capital letter "
            "(A/B/C/D) only — no other text.\n\n"
            f"Question: {question}\n"
            f"Options:\n{cls.format_choices(choices)}"
        )
