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

    # -------- helpers for qwen25omni / vllm local backends --------
    @staticmethod
    def parse_video_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Extract ``{fps, max_frames, min_frames, ...}`` from cfg,
        filtering out None values.  Shared by qwen25omni & vllm_runner.
        These keys are merged into the video content block by
        :func:`filter_media` and read by ``smart_nframes`` at decode time.
        """
        v = dict(cfg.get("video") or {})
        return {
            k: val for k, val in {
                "fps":          v.get("fps"),
                "max_frames":   v.get("max_frames"),
                "min_frames":   v.get("min_frames"),
                "min_pixels":   v.get("min_pixels"),
                "max_pixels":   v.get("max_pixels"),
                "total_pixels": v.get("total_pixels"),
            }.items() if val is not None
        }
