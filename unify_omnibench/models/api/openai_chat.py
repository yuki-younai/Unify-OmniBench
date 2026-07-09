"""OpenAI-compatible chat models.

Two backends registered in this module:

  ``openai_chat`` — plain ``vllm serve`` / GPT-4o / any standard
      OpenAI-compatible gateway.  Uses ONLY standard
      ChatCompletionRequest fields (temperature/max_tokens/top_p +
      documented ``mm_processor_kwargs``/``media_io_kwargs`` extras).

  ``openai_omni`` — vllm-omni ``--omni`` multi-stage server.
      vllm-omni's serving layer ignores standard temperature/max_tokens
      entirely for pipelined (Omni) models and only honors
      ``extra_body.sampling_params_list``.  This backend explicitly
      sets per-stage sampling params to match the transformer/vllm-offline
      baselines (repetition_penalty=1.0, temperature=0, etc.).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from ...core.registry import register_model
from ...core.types import InferenceRequest, MediaRef
from ...prompt.media import media_description
from ...utils.video_io import (
    encode_audio_data_url_16k,
    encode_file_base64,
    extract_qwen_native_frames_base64,
    probe_video_frame_count,
)
from ...utils.logging import get_logger
from ...utils.retry import retry
from ..base import BaseModel

log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Shared base — all media encoding / message building logic lives here
# ═══════════════════════════════════════════════════════════════════════


class _BaseOpenAIChatModel(BaseModel):
    """Shared logic for OpenAI-compatible chat backends.

    Concrete subclasses only need to override ``_init_backend_specific``
    (for backend-specific config parsing) and ``_apply_sampling_overrides``
    (for backend-specific sampling parameter injection into the request).
    """

    supports_modalities = ("video", "image", "audio", "text")
    is_thread_safe = True

    # ── __init__ (common) ──────────────────────────────────────────
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.model_name: str = cfg["model"]

        base_url = cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL", "http://localhost:8001/v1")
        api_key = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY", "EMPTY")

        self._client = OpenAI(api_key=api_key, base_url=base_url)

        vcfg = cfg.get("video", {}) or {}
        self.max_frames = vcfg.get("max_frames", 768)
        self.min_frames = int(vcfg.get("min_frames", 4))
        self.video_mode = vcfg.get("mode", "qwen_native")
        self.video_fps = vcfg.get("fps")

        acfg = cfg.get("audio", {}) or {}
        self.audio_mode = acfg.get("mode", "audio_url")
        self.use_audio_in_video = bool(cfg.get("use_audio_in_video", False))

        self.modalities = cfg.get("modalities", ["text"])
        self.retry_kwargs = cfg.get("retry", {}) or {}
        self.extra_body = (cfg.get("request") or {}).get("extra_body", {}) or {}

        self._init_backend_specific(cfg)

    def _init_backend_specific(self, cfg: Dict[str, Any]) -> None:
        """Hook for subclasses to parse backend-specific config fields."""
        pass

    def load(self) -> None:
        pass  # OpenAI client is stateless

    # ── media encoding (shared) ────────────────────────────────────
    def _media_to_content(
        self, media: List[MediaRef], modality_mode: str
    ) -> List[Dict[str, Any]]:
        """Convert filtered media refs to OpenAI content blocks."""
        out: List[Dict[str, Any]] = []
        has_video = any(mm.kind == "video" for mm in media)
        for m in media:
            if m.kind == "image":
                b64 = encode_file_base64(m.path)
                mime = m.mime or "image/jpeg"
                out.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
            elif m.kind == "video":
                if self.video_mode == "qwen_native":
                    frames, achieved_fps = extract_qwen_native_frames_base64(
                        m.path,
                        fps=float(self.video_fps or 2.0),
                        min_frames=self.min_frames,
                        max_frames=int(self.max_frames),
                    )
                    out.append({
                        "type": "video_url",
                        "video_url": {"url": f"data:video/jpeg;base64,{','.join(frames)}",
                                      "num_frames": len(frames), "fps": achieved_fps},
                    })
                else:  # "video_mp4"
                    ext = os.path.splitext(m.path)[1].lower().lstrip(".")
                    mime = f"video/{ext}" if ext else "video/mp4"
                    b64 = encode_file_base64(m.path)
                    out.append({
                        "type": "video_url",
                        "video_url": {"url": f"data:{mime};base64,{b64}"},
                    })
            elif m.kind == "audio":
                if self.audio_mode == "skip":
                    continue
                if self.use_audio_in_video and has_video and self.video_mode == "video_mp4":
                    continue
                data_url = encode_audio_data_url_16k(m.path)
                out.append({"type": "audio_url", "audio_url": {"url": data_url}})
        return out

    def _media_refs_for_mode(self, sample, modality_mode: str) -> List[MediaRef]:
        if modality_mode == "text" or not sample.media:
            return []
        return [
            m for m in sample.media
            if not (
                (modality_mode == "visual" and m.kind == "audio")
                or (modality_mode == "audio" and m.kind in ("video", "image"))
            )
        ]

    # ── message building (shared) ──────────────────────────────────
    def build_messages(self, req: InferenceRequest) -> List[Dict[str, Any]]:
        """Build OpenAI-compatible messages (system + user)."""
        s = req.sample
        desc = media_description(s, req.modality_mode)
        text_prompt = (req.prompt_template or "").format(
            media_desc=desc,
            question=s.question,
            choices="\n".join(str(c) for c in s.choices),
        )
        system = req.system_prompt or ""
        if req.modality_mode == "text" or not s.media:
            return [
                {"role": "system", "content": system},
                {"role": "user", "content": text_prompt},
            ]
        refs = self._media_refs_for_mode(s, req.modality_mode)
        content = self._media_to_content(refs, req.modality_mode)
        content.append({"type": "text", "text": text_prompt})
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

    # ── common pre-processing (video fps / frame count) ────────────
    def _build_extra_body(self, req: InferenceRequest) -> Dict[str, Any]:
        """Build extra_body with mm_processor_kwargs + media_io_kwargs."""
        extra_body: Dict[str, Any] = dict(self.extra_body or {})
        refs = self._media_refs_for_mode(req.sample, req.modality_mode)
        has_video = any(m.kind == "video" for m in refs)
        if not has_video:
            return extra_body

        has_audio = any(m.kind == "audio" for m in refs)
        mpk = dict(extra_body.get("mm_processor_kwargs") or {})
        if self.use_audio_in_video and has_audio and self.video_mode == "video_mp4":
            mpk["use_audio_in_video"] = True
        if self.video_fps is not None:
            mpk.setdefault("fps", float(self.video_fps))
        if mpk:
            extra_body["mm_processor_kwargs"] = mpk

        if self.video_mode == "video_mp4":
            video_ref = next((m for m in refs if m.kind == "video"), None)
            if video_ref is not None:
                try:
                    nframes = probe_video_frame_count(
                        video_ref.path,
                        target_fps=float(self.video_fps or 2.0),
                        min_frames=self.min_frames,
                        max_frames=int(self.max_frames),
                    )
                    miok = dict(extra_body.get("media_io_kwargs") or {})
                    vio = dict(miok.get("video") or {})
                    vio["num_frames"] = nframes
                    vio.setdefault("fps", float(self.video_fps or 2.0))
                    miok["video"] = vio
                    extra_body["media_io_kwargs"] = miok
                except Exception as e:
                    log.warning("probe_video_frame_count(%s) failed: %s", video_ref.path, e)
        return extra_body

    # ── generate (template method) ─────────────────────────────────
    def generate(self, req: InferenceRequest) -> str:
        """Template method: shared flow, subclasses override sampling."""
        messages = self.build_messages(req)
        max_retries = int(self.retry_kwargs.get("max_retries", 3))
        base_delay = float(self.retry_kwargs.get("base_delay", 2.0))
        max_tokens = req.generation_kwargs.get("max_new_tokens", 10)

        extra_body = self._build_extra_body(req)
        request_kwargs = self._apply_sampling_overrides(max_tokens, extra_body)

        @retry(max_retries=max_retries, base_delay=base_delay)
        def _call() -> str:
            completion = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                modalities=self.modalities,
                extra_body=extra_body or None,
                **request_kwargs,
            )
            content = completion.choices[0].message.content
            if not content:
                raise RuntimeError("Empty response from API")
            return content.strip()

        return _call()

    def _apply_sampling_overrides(self, max_tokens: int, extra_body: Dict[str, Any]) -> Dict[str, Any]:
        """Hook: return extra kwargs for the OpenAI SDK call (temperature etc)."""
        raise NotImplementedError("subclass must override _apply_sampling_overrides")


# ═══════════════════════════════════════════════════════════════════════
# Backend 1: standard OpenAI-compatible (plain vLLM / GPT-4o / ...)
# ═══════════════════════════════════════════════════════════════════════


@register_model("openai_chat")
class OpenAIChatModel(_BaseOpenAIChatModel):
    """Standard OpenAI-compatible chat — plain ``vllm serve``, GPT-4o, etc.

    Uses only standard ChatCompletionRequest fields (temperature, max_tokens,
    top_p).  No vllm-omni-specific ``sampling_params_list`` workaround.
    """

    def _init_backend_specific(self, cfg: Dict[str, Any]) -> None:
        self.sampling_overrides = cfg.get("sampling", {}) or {}

    def _apply_sampling_overrides(self, max_tokens: int, extra_body: Dict[str, Any]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict(
            temperature=float(self.sampling_overrides.get("temperature", 0.0)),
            top_p=float(self.sampling_overrides.get("top_p", 1.0)),
            max_tokens=max_tokens,
        )
        if "repetition_penalty" in self.sampling_overrides:
            extra_body["repetition_penalty"] = self.sampling_overrides["repetition_penalty"]
        return kwargs


# ═══════════════════════════════════════════════════════════════════════
# Backend 2: vllm-omni ``--omni`` multi-stage server
# ═══════════════════════════════════════════════════════════════════════


@register_model("openai_omni")
class OpenAIOmniChatModel(_BaseOpenAIChatModel):
    """OpenAI-compatible chat for vllm-omni ``--omni`` multi-stage servers.

    See module docstring for why sampling params go through
    ``extra_body.sampling_params_list`` instead of standard fields.
    """

    def _init_backend_specific(self, cfg: Dict[str, Any]) -> None:
        self.num_stages = int(cfg.get("num_stages", 1))
        self.sampling_overrides = cfg.get("sampling", {}) or {}

    def _apply_sampling_overrides(self, max_tokens: int, extra_body: Dict[str, Any]) -> Dict[str, Any]:
        sp = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "max_tokens": max_tokens,
            **self.sampling_overrides,
        }
        extra_body.setdefault("sampling_params_list", [dict(sp) for _ in range(self.num_stages)])
        # Do NOT send temperature/max_tokens via standard request fields —
        # vllm-omni ignores them for Omni models anyway.
        return {}

