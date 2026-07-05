"""Qwen2.5-Omni local Transformers backend.

Conversation construction is provided by :func:`build_messages` so both
the Transformers and vLLM backends produce **identical** inputs.  Media
handling delegates to :mod:`unify_omnibench.prompt.media` so new image-based
or audio-only benchmarks work without changing model code.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ...core.registry import register_model
from ...core.types import InferenceRequest, Sample
from ...prompt.media import filter_media, media_description
from ...prompt.templates import PromptTemplate, QWEN_OMNI_DEFAULT
from ...utils.logging import get_logger
from ..base import BaseModel

log = get_logger(__name__)


def _try_import_process_mm_info():
    try:
        from qwen_omni_utils import process_mm_info  # type: ignore
        return process_mm_info
    except Exception:
        return None


# ── shared prompt / media construction ──────────────────────────────────
def build_messages(
    sample: Sample,
    modality_mode: str,
    prompt_template: Optional[PromptTemplate] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Build the Qwen2.5-Omni conversation (system + user).

    Returns ``(conversation, use_audio_in_video)``.  Always returns
    ``use_audio_in_video=False`` to match the original Daily-Omni evaluation
    (use the separate .wav audio file, ignore the video's audio track).
    """
    template = prompt_template or QWEN_OMNI_DEFAULT
    desc = media_description(sample, modality_mode)
    user_content = filter_media(sample, modality_mode)

    prompt = template.render(
        media_desc=desc,
        question=sample.question,
        choices=sample.choices,
    )
    user_content.append({"type": "text", "text": prompt})

    conversation = [
        {"role": "system", "content": [
            {"type": "text", "text": template.system or ""},
        ]},
        {"role": "user", "content": user_content},
    ]
    return conversation, False  # use_audio_in_video always False


@register_model("transformers_qwen25omni")
class Qwen25OmniModel(BaseModel):
    """Local Qwen2.5-Omni via 🤗 Transformers.

    cfg::

        name: transformers_qwen25omni
        model_name_or_path: Qwen/Qwen2.5-Omni-7B
        device: auto
        attn_implementation: flash_attention_2     # or "eager"
        torch_dtype: bfloat16
        max_frames: 256
        prompt_template: null                      # optional user-prompt override
        system_prompt: null                        # optional system-prompt override
    """

    supports_modalities = ("video", "audio", "image", "text")
    is_thread_safe = False
    supports_batch = True  # enable batch via generate_batch() below

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.model_path: str = cfg["model_name_or_path"]
        self.device_map = cfg.get("device", "auto")
        self.attn_impl = cfg.get("attn_implementation", "flash_attention_2")
        self.torch_dtype_str = cfg.get("torch_dtype", "bfloat16")
        self.max_frames = int(cfg.get("max_frames", 256))
        self.prompt_template = PromptTemplate.from_config(cfg, QWEN_OMNI_DEFAULT)
        self.model = None
        self.processor = None
        self._process_mm_info = None

    def load(self) -> None:
        import torch  # noqa: F401
        from transformers import (  # type: ignore
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor,
        )
        dtype_map = {
            "bfloat16": __import__("torch").bfloat16,
            "float16": __import__("torch").float16,
            "float32": __import__("torch").float32,
        }
        torch_dtype = dtype_map.get(self.torch_dtype_str, __import__("torch").bfloat16)

        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=self.device_map,
            attn_implementation=self.attn_impl,
            enable_audio_output=False,   # 评测不需要 TTS，只加载 Thinker
        )
        # 显式指定 use_fast=True：HF 的 Qwen2VLImageProcessor 新版本默认已经改成
        # fast processor，且官方明确警告 fast/slow 两种实现的输出会有细微差异
        # ("This is a breaking change and may produce slightly different outputs")。
        # vLLM 服务端(openai backend 打的那个服务)日志确认它用的是 fast 处理器；
        # 这里显式对齐成同一种，避免两条路径各自依赖所在环境 transformers 版本的
        # 隐式默认值，从而产生看不见的图像预处理差异。
        try:
            self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path, use_fast=True)
        except TypeError:
            # 装的 transformers 版本太老，不认识 use_fast 参数 —— 直接退回默认加载
            log.warning("Qwen2_5OmniProcessor.from_pretrained() 不支持 use_fast 参数，使用默认加载方式")
            self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)
        self._process_mm_info = _try_import_process_mm_info()
        if self._process_mm_info is None:
            log.warning(
                "qwen_omni_utils.process_mm_info not found; "
                "install with `pip install qwen-omni-utils` if you need video/audio inputs."
            )

    def _resolve_template(self, req: InferenceRequest) -> PromptTemplate:
        """Resolve the prompt template: benchmark override > model config > default."""
        if req.prompt_template:
            return PromptTemplate(user=req.prompt_template, system=self.prompt_template.system)
        return self.prompt_template

    def generate(self, req: InferenceRequest) -> str:
        import torch
        template = self._resolve_template(req)
        conv, _ = build_messages(req.sample, req.modality_mode, template)

        text = self.processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False
        )
        if self._process_mm_info is not None:
            audios, images, videos = self._process_mm_info(conv, use_audio_in_video=False)
        else:
            audios, images, videos = None, None, None

        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=False,
        )
        # float tensors → device + dtype; int tensors → device only
        for key, value in list(inputs.items()):
            if isinstance(value, torch.Tensor):
                if value.is_floating_point():
                    inputs[key] = value.to(device=self.model.device, dtype=self.model.dtype)
                else:
                    inputs[key] = value.to(device=self.model.device)

        max_tokens = req.generation_kwargs.get("max_new_tokens", 10)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                use_audio_in_video=False,
                return_audio=False,
                max_new_tokens=max_tokens,
                num_beams=1,
                do_sample=False,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        in_len = inputs["input_ids"].shape[1]
        gen_ids = out[0][:, in_len:] if isinstance(out, tuple) else out[:, in_len:]
        decoded = self.processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return (decoded[0] if decoded else "").strip()

    def generate_batch(self, reqs: List[InferenceRequest]) -> List[str]:
        """Batch inference: process multiple samples together for higher GPU utilization."""
        import torch
        if not reqs:
            return []

        texts, all_audios, all_images, all_videos = [], [], [], []
        for req in reqs:
            template = self._resolve_template(req)
            conv, _ = build_messages(req.sample, req.modality_mode, template)
            texts.append(
                self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            )
            if self._process_mm_info is not None:
                a, im, v = self._process_mm_info(conv, use_audio_in_video=False)
            else:
                a, im, v = None, None, None
            all_audios.append(a[0] if a else None)
            all_images.append(im[0] if im else None)
            all_videos.append(v[0] if v else None)

        audios  = [x for x in all_audios  if x is not None] or None
        images  = [x for x in all_images  if x is not None] or None
        videos  = [x for x in all_videos  if x is not None] or None

        inputs = self.processor(
            text=texts, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=False,
        )
        for key, value in list(inputs.items()):
            if isinstance(value, torch.Tensor):
                if value.is_floating_point():
                    inputs[key] = value.to(device=self.model.device, dtype=self.model.dtype)
                else:
                    inputs[key] = value.to(device=self.model.device)

        max_tokens = reqs[0].generation_kwargs.get("max_new_tokens", 10) if reqs else 10
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                use_audio_in_video=False,
                return_audio=False,
                max_new_tokens=max_tokens,
                num_beams=1,
                do_sample=False,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        gen_ids = out[0] if isinstance(out, tuple) else out
        in_len = inputs["input_ids"].shape[1]
        gen_ids = gen_ids[:, in_len:]
        decoded = self.processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return [s.strip() for s in decoded]
