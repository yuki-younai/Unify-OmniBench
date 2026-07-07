"""Model-agnostic media helpers.

All benchmark- and model-specific code should go through this module so that
adding a new benchmark (e.g. a pure-image or pure-audio dataset) doesn't
require touching any model backend.

Key concepts
------------
* **ModalityClass** — a set of flags describing what media types a sample contains.
* **filter_media** — drop media entries that shouldn't be fed to the model
  for a given *modality_mode* (av / visual / audio / text).
* **visual_label** — the word that should appear in the prompt ("video" or
  "image") depending on what's actually in the sample.
* **media_description** — the full "given …" phrase for the prompt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from ..core.types import MediaRef, Sample


@dataclass
class ModalityClass:
    """Flags describing what media types are present in a sample."""

    has_video: bool = False
    has_image: bool = False
    has_audio: bool = False

    _present: Set[str] = field(default_factory=set, repr=False)

    def __bool__(self) -> bool:
        return bool(self._present)

    @property
    def visual_label(self) -> str:
        """Human-readable label for the visual media type."""
        if self.has_video:
            return "video"
        if self.has_image:
            return "image"
        return "media"

    @property
    def has_visual(self) -> bool:
        return self.has_video or self.has_image


def classify_media(sample: Sample) -> ModalityClass:
    """Inspect the sample's media list and return a :class:`ModalityClass`."""
    mc = ModalityClass()
    for m in sample.media:
        mc._present.add(m.kind)
        if m.kind == "video":
            mc.has_video = True
        elif m.kind == "image":
            mc.has_image = True
        elif m.kind == "audio":
            mc.has_audio = True
    return mc


def visual_label(sample: Sample) -> str:
    """Return ``"video"``, ``"image"`` or ``"media"`` depending on content."""
    return classify_media(sample).visual_label


def media_description(sample: Sample, modality_mode: str) -> str:
    """Build the ``"given …"`` phrase for the prompt.

    Examples
    --------
    * ``av`` with video+audio  → ``"given video and audio together"``
    * ``av`` with image+audio  → ``"given image and audio together"``
    * ``visual`` with image    → ``"given image"``
    * ``audio``               → ``"given audio"``
    * ``text``                → ``"given text description"``
    """
    mc = classify_media(sample)

    if modality_mode == "audio":
        return "given audio"
    if modality_mode == "text":
        return "given text description"
    if modality_mode in ("all", "av"):
        if mc.has_audio and mc.has_visual:
            return f"given {mc.visual_label} and audio together"
        return f"given {mc.visual_label}"
    # "visual" or anything else
    return f"given {mc.visual_label}"


def filter_media(
    sample: Sample,
    modality_mode: str,
    *,
    extra_skip_kinds: tuple = (),
    video_kwargs: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Return ``{"type": kind, kind: path}`` dicts for media that should be
    passed to the model given *modality_mode*.

    Audio-only mode skips **all** visual media (video + image).
    Visual-only mode skips audio.
    ``text`` mode returns an empty list.

    *video_kwargs*: extra keys (e.g. ``fps``/``max_frames``/``min_frames``)
    merged into any ``"video"`` content block. These are read directly by
    ``qwen_omni_utils.vision_process.smart_nframes`` off the video element
    dict itself, so setting them here caps the frame budget at DECODE time
    (cheap) rather than after decode+resize (wasteful) — see
    ``dataset_config.yaml``'s ``video:`` comment for context.
    """
    visual_kinds = ("video", "image")
    result: List[Dict[str, Any]] = []
    for m in sample.media:
        if modality_mode == "text":
            break
        if modality_mode == "visual" and m.kind == "audio":
            continue
        if modality_mode == "audio" and m.kind in visual_kinds:
            continue
        if m.kind in extra_skip_kinds:
            continue
        entry: Dict[str, Any] = {"type": m.kind, m.kind: m.path}
        if m.kind == "video" and video_kwargs:
            entry.update(video_kwargs)
        result.append(entry)
    return result
