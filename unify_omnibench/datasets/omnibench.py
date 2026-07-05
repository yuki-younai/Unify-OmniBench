"""OmniBench adapter.

Each record is expected to include either:
  * ``options``: list of strings   OR  ``option``: "A. ... B. ... C. ... D. ..."
  * ``image_path`` and/or ``audio_path`` (relative to ``mm_root/image`` and ``mm_root/audio``)
  * ``answer`` or ``correct answer``: the gold letter
  * ``task type`` / ``audio type`` / ``index`` (optional meta)

Supports ``.jsonl`` and ``.xlsx`` (requires pandas).
"""
from __future__ import annotations

import json
import os
import re
from typing import Iterator, List

from ..core.registry import register_dataset
from ..core.types import MediaRef, Sample
from .base import BaseDatasetAdapter

_OPT_RE = re.compile(r"(?P<L>[A-D])\s*[\.\)]\s*(?P<T>.+?)(?=\s+[A-D]\s*[\.\)]|$)", re.DOTALL)


def _parse_options(raw) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw]
    s = str(raw)
    matches = _OPT_RE.findall(s)
    if matches:
        return [f"{lt}. {tx.strip()}" for lt, tx in matches]
    # fallback: split by newline
    parts = [p.strip() for p in re.split(r"[\n;]", s) if p.strip()]
    return parts or [s]


# @register_dataset("omnibench")  — replaced by datasets/unified.py
class OmniBenchAdapter(BaseDatasetAdapter):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.records = self._load(cfg["data_file"])
        self.mm_root = cfg["mm_root"]
        self.image_subdir = cfg.get("image_subdir", "image")
        self.audio_subdir = cfg.get("audio_subdir", "audio")

    @staticmethod
    def _load(path: str):
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "data" in data:
                return data["data"]
            raise ValueError(f"Unsupported json structure in {path}")
        if path.endswith(".xlsx") or path.endswith(".xls"):
            import pandas as pd  # type: ignore
            return pd.read_excel(path).to_dict("records")
        raise ValueError(f"Unsupported OmniBench data file: {path}")

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[Sample]:
        for idx, r in enumerate(self.records):
            options = _parse_options(r.get("options") or r.get("option"))
            answer_raw = r.get("answer") or r.get("correct answer") or r.get("correct_answer")

            # OmniBench stores the *full text* of the correct option as answer.
            # Map it back to the letter index (A/B/C/D).
            if isinstance(answer_raw, str) and options:
                answer_raw = answer_raw.strip()
                found = None
                letter_idx = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                for i, opt in enumerate(options):
                    opt_text = opt.strip()
                    # Strip "A. " prefix if present for comparison
                    if opt_text.startswith(letter_idx[i] + "."):
                        opt_text = opt_text[2:].strip()
                    if opt_text == answer_raw:
                        found = letter_idx[i]
                        break
                answer = found or answer_raw.upper()[:1]
            elif isinstance(answer_raw, str):
                answer = answer_raw.strip().upper()[:1]
            else:
                answer = answer_raw

            media: List[MediaRef] = []
            img = r.get("image_path") or r.get("image")
            if img:
                ip = img if os.path.isabs(img) else os.path.join(self.mm_root, self.image_subdir, img)
                mime = "image/jpeg" if ip.lower().endswith((".jpg", ".jpeg")) else "image/png"
                media.append(MediaRef(kind="image", path=ip, mime=mime))
            aud = r.get("audio_path") or r.get("audio")
            if aud:
                ap = aud if os.path.isabs(aud) else os.path.join(self.mm_root, self.audio_subdir, aud)
                media.append(MediaRef(kind="audio", path=ap, mime="audio/wav"))

            yield Sample(
                uid=self.make_uid(r.get("index", idx)),
                dataset=self.name,
                question=r.get("question", ""),
                choices=options,
                answer=answer,
                media=media,
                meta={
                    "task_type": r.get("task type") or r.get("task_type"),
                    "audio_type": r.get("audio type") or r.get("audio_type"),
                    "index": r.get("index", idx),
                },
            )
