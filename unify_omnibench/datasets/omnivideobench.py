"""OmniVideoBench adapter.

Data format (one JSON file)::

  [
    {
      "video": "<name>",                # video file = <video_dir>/<name>.mp4
      "video_type": "...",
      "duration": "12:34" or "01:02:03",
      "questions": [
        {
          "question": "...",
          "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
          "correct_option": "A",
          "question_type": "...",
          "audio_type": "...",          # optional
        }, ...
      ]
    },
    ...
  ]
"""
from __future__ import annotations

import json
import os
from typing import Iterator, List, Optional

from ..core.registry import register_dataset
from ..core.types import MediaRef, Sample
from .base import BaseDatasetAdapter


def _mmss_to_sec(t) -> Optional[int]:
    if t is None:
        return None
    if isinstance(t, (int, float)):
        return int(t)
    parts = str(t).split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        return None
    return None


# @register_dataset("omnivideobench")  — replaced by datasets/unified.py
class OmniVideoBenchAdapter(BaseDatasetAdapter):
    def __init__(self, cfg):
        super().__init__(cfg)
        with open(cfg["data_file"], "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.video_dir = cfg["video_dir"]
        self.video_ext = cfg.get("video_ext", ".mp4")
        self._samples: List[Sample] = list(self._expand())

    def _expand(self) -> Iterator[Sample]:
        for v_idx, v in enumerate(self.data):
            vname = v["video"]
            vpath = os.path.join(self.video_dir, f"{vname}{self.video_ext}")
            dur = _mmss_to_sec(v.get("duration"))
            for q_idx, q in enumerate(v.get("questions", [])):
                yield Sample(
                    uid=self.make_uid(v_idx, vname, q_idx),
                    dataset=self.name,
                    question=q.get("question", ""),
                    choices=q.get("options", []),
                    answer=(q.get("correct_option") or q.get("answer")
                            or "").strip().upper()[:1] or None,
                    media=[
                        MediaRef(
                            kind="video", path=vpath, mime="video/mp4",
                            extra={"duration_s": dur},
                        )
                    ],
                    meta={
                        "video": vname,
                        "video_type": v.get("video_type"),
                        "question_type": q.get("question_type"),
                        "audio_type": q.get("audio_type"),
                        "duration_s": dur,
                    },
                )

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)
