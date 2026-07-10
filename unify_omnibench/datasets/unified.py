"""Unified-format dataset adapter — reads the standardized JSON produced
by ``script/convert_*.py``.

Config example::

    data_file: /path/to/Unify-OmniBench/data/omnibench.json
    media_root: /path/to/Unify-OmniBench           # prepended to relative paths
"""
from __future__ import annotations

import json, os
from typing import Iterator

from ..core.registry import register_dataset
from ..core.types import MediaRef, Sample
from .base import BaseDatasetAdapter


_UNIFIED_FIELDS = ("id", "question", "choices", "answer", "video_path", "audio_path", "image_path",
                   "task_type", "category", "duration", "meta")


@register_dataset("omnibench")
@register_dataset("daily_omni")
@register_dataset("omnivideobench")
@register_dataset("worldsense")
@register_dataset("videomme")
class UnifiedAdapter(BaseDatasetAdapter):
    """Load any benchmark that's been converted to the unified JSON format."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.name = cfg.get("name", self.name)  # 实例属性覆盖类属性（多个装饰器共享同一个类，cls.name 取最后一次赋值）
        with open(cfg["data_file"], "r", encoding="utf-8") as f:
            self.records = json.load(f)
        self.media_root = cfg.get("media_root", "")

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[Sample]:
        for r in self.records:
            media: list[MediaRef] = []
            for kind, key in [("video", "video_path"), ("audio", "audio_path"), ("image", "image_path")]:
                rel = r.get(key)
                if rel:
                    path = os.path.join(self.media_root, rel) if self.media_root else rel
                    if os.path.exists(path):
                        media.append(MediaRef(kind=kind, path=path))
            yield Sample(
                uid=r.get("id", r.get("index", "")),
                dataset=self.name,
                question=r.get("question", ""),
                choices=r.get("choices") or [],
                answer=r.get("answer"),
                media=media,
                meta={
                    "task_type": r.get("task_type"),
                    "category": r.get("category"),
                    "duration": r.get("duration"),
                    **(r.get("meta") or {}),
                },
            )
