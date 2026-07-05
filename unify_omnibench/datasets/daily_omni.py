"""Daily-Omni adapter.

Expected layout::

  qa_file: path/to/qa.json     # list[dict]
     each item: {Question, Choice, Answer, video_id, Type, video_category, video_duration}
  video_base_dir: path/to/videos/
     each video lives in:  <video_base_dir>/<video_id>/<video_id>_video.mp4
                           <video_base_dir>/<video_id>/<video_id>_audio.wav
"""
from __future__ import annotations

import json
import os
from typing import Iterator, List

from ..core.registry import register_dataset
from ..core.types import MediaRef, Sample
from .base import BaseDatasetAdapter


# @register_dataset("daily_omni")  — replaced by datasets/unified.py
class DailyOmniAdapter(BaseDatasetAdapter):
    def __init__(self, cfg):
        super().__init__(cfg)
        with open(cfg["qa_file"], "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.video_base = cfg["video_base_dir"]
        self.require_audio = cfg.get("require_audio", True)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[Sample]:
        for idx, item in enumerate(self.data):
            vid = str(item["video_id"])
            video_path = os.path.join(self.video_base, vid, f"{vid}_video.mp4")
            audio_path = os.path.join(self.video_base, vid, f"{vid}_audio.wav")
            media: List[MediaRef] = [
                MediaRef(kind="video", path=video_path, mime="video/mp4"),
            ]
            if self.require_audio or os.path.exists(audio_path):
                media.append(MediaRef(kind="audio", path=audio_path, mime="audio/wav"))

            choices = item.get("Choice") or item.get("choices") or item.get("Choices")
            if isinstance(choices, str):
                choices = [c.strip() for c in choices.split("\n") if c.strip()]

            yield Sample(
                uid=self.make_uid(idx, vid),
                dataset=self.name,
                question=item.get("Question") or item.get("question", ""),
                choices=choices or [],
                answer=item.get("Answer") or item.get("answer"),
                media=media,
                meta={
                    "video_id": vid,
                    "task_type": item.get("Type") or item.get("type"),
                    "video_category": item.get("video_category"),
                    "video_duration": item.get("video_duration"),
                },
            )
