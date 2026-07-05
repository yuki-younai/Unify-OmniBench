"""Core data types shared across datasets / models / runner."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional, Tuple

Modality = Literal["video", "audio", "image", "text"]


@dataclass
class MediaRef:
    """Reference to a media file on disk (or remote, future)."""
    kind: Modality                       # "video" | "audio" | "image"
    path: str                            # absolute path preferred
    mime: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Sample:
    """Unified multimodal QA sample."""
    uid: str
    dataset: str
    question: str
    choices: List[str]
    answer: Optional[str] = None         # ground-truth letter (A/B/C/D)
    media: List[MediaRef] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class InferenceRequest:
    """Carries a sample + per-run modality / prompt / generation settings."""
    sample: Sample
    modality_mode: str = "av"            # "av" | "visual" | "audio" | "text"
    prompt_template: Optional[str] = None
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """One per sample. Persisted to items.jsonl."""
    uid: str
    dataset: str
    question: str = ""
    choices: List[str] = field(default_factory=list)
    raw_output: str = ""
    parsed_answer: Optional[str] = None
    correct_answer: Optional[str] = None
    is_correct: bool = False
    latency_s: float = 0.0
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
