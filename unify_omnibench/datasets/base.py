"""Abstract base class for dataset adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator

from ..core.types import Sample


class BaseDatasetAdapter(ABC):
    """Subclasses must:
      * iterate :class:`Sample` instances via ``__iter__``
      * provide a length via ``__len__``
    Register with :func:`unify_omnibench.core.registry.register_dataset`.
    """
    name: str = ""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    @abstractmethod
    def __iter__(self) -> Iterator[Sample]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    # ---- helpers
    def make_uid(self, *parts) -> str:
        return f"{self.name}:" + ":".join(str(p) for p in parts)
