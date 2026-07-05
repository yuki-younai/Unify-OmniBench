"""Google Gemini model — uses ``google-genai`` SDK; thread-local client."""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional

from ...core.registry import register_model
from ...core.types import InferenceRequest, MediaRef
from ...utils.logging import get_logger
from ...utils.retry import retry
from ..base import BaseModel

log = get_logger(__name__)


@register_model("gemini")
class GeminiModel(BaseModel):
    """Gemini via ``google-genai``.

    cfg::

        name: gemini
        model: gemini-2.5-pro
        api_key: ${GEMINI_API_KEY}
        upload_poll_interval_s: 5
        upload_max_polls: 100
    """

    supports_modalities = ("video", "image", "audio", "text")
    is_thread_safe = True

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.api_key = cfg.get("api_key") or os.environ.get(
            cfg.get("api_key_env", "GEMINI_API_KEY"), ""
        )
        self.model_name = cfg["model"]
        self._local = threading.local()
        self.poll_interval = float(cfg.get("upload_poll_interval_s", 5.0))
        self.poll_max = int(cfg.get("upload_max_polls", 100))
        self.retry_kwargs = cfg.get("retry", {}) or {}
        self.system_prompt = cfg.get(
            "system_prompt",
            "You are a multimodal evaluator. Reply with a single letter A/B/C/D.",
        )

    def load(self) -> None:
        # nothing eager; per-thread client lazily created
        if not self.api_key:
            log.warning("GeminiModel: api_key is empty")

    def _client(self):
        if not hasattr(self._local, "c"):
            from google import genai
            self._local.c = genai.Client(api_key=self.api_key)
        return self._local.c

    def _upload(self, path: str):
        client = self._client()
        f = client.files.upload(file=path)
        polls = 0
        while getattr(f.state, "name", "") == "PROCESSING" and polls < self.poll_max:
            time.sleep(self.poll_interval)
            f = client.files.get(name=f.name)
            polls += 1
        if getattr(f.state, "name", "") == "FAILED":
            raise RuntimeError(f"Gemini file upload failed: {path}")
        return f

    def _select_media(self, media: List[MediaRef], modality_mode: str) -> List[MediaRef]:
        if modality_mode == "text":
            return []
        out = []
        for m in media:
            if modality_mode == "visual" and m.kind == "audio":
                continue
            if modality_mode == "audio" and m.kind == "video":
                continue
            out.append(m)
        return out

    def generate(self, req: InferenceRequest) -> str:
        from google.genai import types  # local

        s = req.sample
        prompt = req.prompt_template or self.default_prompt(s.question, s.choices)
        media = self._select_media(s.media, req.modality_mode)
        uploaded = []
        try:
            for m in media:
                uploaded.append(self._upload(m.path))

            contents = [*uploaded, prompt] if uploaded else [prompt]

            @retry(
                max_retries=int(self.retry_kwargs.get("max_retries", 4)),
                base_delay=float(self.retry_kwargs.get("base_delay", 4.0)),
            )
            def _call() -> str:
                r = self._client().models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=req.generation_kwargs.get("temperature", 0.0),
                        system_instruction=self.system_prompt,
                    ),
                )
                txt = (r.text or "").strip()
                if not txt:
                    raise RuntimeError("Empty Gemini response")
                return txt

            return _call()
        finally:
            client = self._client()
            for f in uploaded:
                try:
                    client.files.delete(name=f.name)
                except Exception:  # pragma: no cover
                    pass
