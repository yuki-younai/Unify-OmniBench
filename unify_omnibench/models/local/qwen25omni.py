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
    use_audio_in_video: bool = False,
    video_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Build the Qwen2.5-Omni conversation (system + user).

    Returns ``(conversation, use_audio_in_video)`` (the latter just echoes
    back the *use_audio_in_video* argument for caller convenience).

    Default ``use_audio_in_video=False`` matches the original Daily-Omni /
    OmniBench evaluation (audio sent as a separate stream, e.g. a .wav
    file — filtered in via ``filter_media`` based on the Sample's actual
    ``MediaRef`` list, independent of this flag). Datasets whose audio
    lives ONLY inside the video container (e.g. OmniVideoBench, which has
    no separate audio MediaRef) must pass ``use_audio_in_video=True`` —
    otherwise ``qwen_omni_utils.process_mm_info`` has no audio source at
    all to extract from and the model receives zero audio signal.

    *video_kwargs*: optional ``{"fps": ..., "max_frames": ..., "min_frames":
    ...}`` dict forwarded to :func:`~unify_omnibench.prompt.media.filter_media`,
    which merges it into the ``"video"`` content block so ``qwen_omni_utils``
    caps the sampled frame count at DECODE time (see dataset_config.yaml's
    ``video:`` field) instead of decoding a much larger default budget and
    throwing most of it away afterwards.
    """
    template = prompt_template or QWEN_OMNI_DEFAULT
    desc = media_description(sample, modality_mode)
    user_content = filter_media(sample, modality_mode, video_kwargs=video_kwargs)

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
    return conversation, use_audio_in_video


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
        # video 抽帧参数：优先读 dataset_config.yaml 合并进来的 ``video`` 字典
        # ({"fps", "max_frames", "min_frames"})，在 build_messages() 里注入到
        # video content block，让 qwen_omni_utils 在解码源头就按这个预算采样，
        # 而不是解码完默认的 768 帧再事后裁剪（见 dataset_config.yaml 注释）。
        video_cfg = dict(cfg.get("video") or {})
        self.video_kwargs: Dict[str, Any] = {
            k: v for k, v in {
                "fps": video_cfg.get("fps"),
                "max_frames": video_cfg.get("max_frames"),
                "min_frames": video_cfg.get("min_frames"),
                # 像素预算三元组 —— WorldSense (VLMEvalKit::Qwen2VLChat) 需要，
                # qwen_omni_utils.vision_process.smart_resize 直接读取这些
                # per-element 键来决定每帧 resize 到的目标像素数。
                "min_pixels": video_cfg.get("min_pixels"),
                "max_pixels": video_cfg.get("max_pixels"),
                "total_pixels": video_cfg.get("total_pixels"),
            }.items() if v is not None
        }
        # 仍保留事后裁剪作为安全网（例如装的 qwen_omni_utils 版本忽略了
        # per-element max_frames 字段时，至少不会把超量帧喂给 processor）
        self.max_frames = int(video_cfg.get("max_frames") or cfg.get("max_frames", 256))
        # 每数据集可通过 dataset_config.yaml::use_audio_in_video 覆盖（run.py 合并进
        # model_cfg）。True = 从视频容器自带音轨按时间戳提取音频（OmniVideoBench 需要，
        # 因为它没有独立音频文件）；False = 音频完全依赖 Sample.media 里的独立 MediaRef
        # （Daily-Omni / OmniBench 的场景，process_mm_info 不会触碰视频的音轨）。
        self.use_audio_in_video = bool(cfg.get("use_audio_in_video", False))
        self.prompt_template = PromptTemplate.from_config(cfg, QWEN_OMNI_DEFAULT)
        self.model = None
        self.processor = None
        self._process_mm_info = None

    def load(self) -> None:
        import logging as _logging
        import warnings as _warnings
        # qwen_omni_utils.vision_process 用 logger.warning() 打印
        # "The given max_pixels[...] exceeds limit[...]"，这是 per-element 动态
        # 像素预算计算的正常提示（total_pixels 摊薄后的上限低于我们配置的
        # max_pixels 时必然触发），logging 模块没有内置去重，几乎每个视频样本
        # 都会打一遍 —— 压掉，不影响正确性，只是噪音。
        _logging.getLogger("qwen_omni_utils").setLevel(_logging.ERROR)
        # librosa 的 __audioread_load FutureWarning 本应只打一次（Python
        # warnings 模块默认按 (message, lineno) 去重），但期间其它库反复调用
        # warnings.filterwarnings() 会让去重缓存失效，导致重复出现 —— 显式过滤掉。
        _warnings.filterwarnings("ignore", category=FutureWarning, module="librosa.*")

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

    def _cap_video_frames(self, videos, max_frames: int):
        """Uniformly downsample each video's frames so none exceeds *max_frames*.

        Mirrors the inline ``MAX_FRAMES=256`` uniform downsampling in
        OmniVideoBench's official ``eval/qwenomni_eval.py::process_multimedia_input``.
        """
        import numpy as np
        if videos is None or max_frames <= 0:
            return videos
        capped = []
        for v in videos:
            if len(v) <= max_frames:
                capped.append(v)
                continue
            step = len(v) / max_frames
            indices = [int(i * step) for i in range(max_frames)]
            capped.append(v[indices])
        return capped

    def generate(self, req: InferenceRequest) -> str:
        import torch
        template = self._resolve_template(req)
        conv, _ = build_messages(
            req.sample, req.modality_mode, template,
            use_audio_in_video=self.use_audio_in_video,
            video_kwargs=self.video_kwargs,
        )

        text = self.processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False
        )
        if self._process_mm_info is not None:
            audios, images, videos = self._process_mm_info(
                conv, use_audio_in_video=self.use_audio_in_video
            )
            # 安全网：即便解码源头的 max_frames 生效，这里再做一次事后裁剪也无害
            # （len(v) <= max_frames 时是 no-op）；如果源头没生效（例如装的
            # qwen_omni_utils 版本忽略了 per-element 字段），这里仍能兜底防止
            # 喂给 processor 的帧数超预算。
            videos = self._cap_video_frames(videos, self.max_frames)
        else:
            audios, images, videos = None, None, None

        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=self.use_audio_in_video,
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
                use_audio_in_video=self.use_audio_in_video,
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
            conv, _ = build_messages(
                req.sample, req.modality_mode, template,
                use_audio_in_video=self.use_audio_in_video,
                video_kwargs=self.video_kwargs,
            )
            texts.append(
                self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            )
            if self._process_mm_info is not None:
                a, im, v = self._process_mm_info(conv, use_audio_in_video=self.use_audio_in_video)
                # 事后裁剪安全网（见 generate() 里同一处的说明）
                v = self._cap_video_frames(v, self.max_frames)
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
            return_tensors="pt", padding=True, use_audio_in_video=self.use_audio_in_video,
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
                use_audio_in_video=self.use_audio_in_video,
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
