"""Qwen2.5-Omni vLLM batch backend.

Uses the *Omni-Bench-Work*same** prompt / media construction as the Transformers backend
(:func:`~.local.qwen25omni.build_messages`). Only the generation engine
differs. See ``docs/Unify-OmniBench-v0.1.0-dev.md`` for the full debugging
history behind the choices below.

Environment variables:
    VLLM_USE_V1                          — do NOT set this to 0. vLLM has
        fully removed the V0 engine; forcing V0 raises a ValueError at
        LLMEngine init. Leave unset.
    VLLM_WORKER_MULTIPROC_METHOD=spawn   — avoid CUDA re-init in forked workers
    VLLM_DISABLE_PROGRESS_BAR=1          — suppress vLLM internal tqdm bars

``use_audio_in_video`` (per-dataset, via ``dataset_config.yaml``):
Daily-Omni/OmniBench keep it ``False`` (independent audio file, sent as
its own ``multi_modal_data`` entry); OmniVideoBench/WorldSense set it
``True`` (audio comes only from the video's own track, no separate audio
file attached). Interleaved mode requires vllm >= the version that merged
upstream PR #33605 (pinned vllm==0.17.0 here is past that fix; regression
test: ``tests/test_qwen_omni_vllm.py``).
"""
from __future__ import annotations

import gc
from typing import Any, Dict, List

from ...core.registry import register_model
from ...core.types import InferenceRequest
from ...utils.logging import get_logger
from ...utils.retry import retry
from ..base import BaseModel
from ..local.qwen25omni import build_messages
from qwen_omni_utils import process_mm_info  # type: ignore

log = get_logger(__name__)


@register_model("vllm")
class VLLMModel(BaseModel):
    supports_modalities = ("video", "audio", "image", "text")
    is_thread_safe = False
    supports_batch = True

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.model_path: str = cfg["model"]
        self.tp_size = int(cfg.get("tensor_parallel_size", 1))
        self.gpu_util = float(cfg.get("gpu_memory_utilization", 0.95))
        self.max_num_seqs = int(cfg.get("max_num_seqs", 1))
        self.dtype = cfg.get("dtype", "bfloat16")
        self.max_model_len = int(cfg.get("max_model_len", 32768))
        self.seed = int(cfg.get("seed", 1234))
        self.enforce_eager = bool(cfg.get("enforce_eager", True))
        self.enable_prefix_caching = bool(cfg.get("enable_prefix_caching", False))
        # Disables vLLM's multi-modal processor cache — see vllm.yaml's
        # mm_processor_cache_gb comment / docs/Unify-OmniBench-v0.1.0-dev.md for why.
        self.mm_processor_cache_gb = float(cfg.get("mm_processor_cache_gb", 0))
        self.video_kwargs = self.parse_video_kwargs(cfg)
        self.use_audio_in_video = bool(cfg.get("use_audio_in_video", False))
        self.llm = None
        self.processor = None

    def load(self) -> None:
        import os as _os
        import logging as _logging
        import warnings as _warnings
        _os.environ.setdefault("VLLM_DISABLE_PROGRESS_BAR", "1")
        _logging.getLogger("vllm").setLevel(_logging.ERROR)
        _logging.getLogger("qwen_omni_utils").setLevel(_logging.ERROR)
        _warnings.filterwarnings("ignore", category=FutureWarning, module="librosa.*")

        from transformers import Qwen2_5OmniProcessor  # type: ignore
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase  # type: ignore

        # Compat shim: some transformers builds removed
        # ``all_special_tokens_extended``, which older vLLM tokenizer-caching
        # code still reads at engine/worker startup. Also applied globally
        # in sitecustomize.py so vLLM's spawned worker subprocesses pick it
        # up too (this in-process patch alone wouldn't reach them).
        if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
            PreTrainedTokenizerBase.all_special_tokens_extended = property(
                lambda self: self.all_special_tokens
            )

        from vllm import LLM  # type: ignore

        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            tensor_parallel_size=self.tp_size,
            gpu_memory_utilization=self.gpu_util,
            max_num_seqs=self.max_num_seqs,
            max_model_len=self.max_model_len,
            dtype=self.dtype,
            seed=self.seed,
            limit_mm_per_prompt={"image": 1, "video": 1, "audio": 1},
            enforce_eager=self.enforce_eager,
            enable_prefix_caching=self.enable_prefix_caching,
            mm_processor_cache_gb=self.mm_processor_cache_gb,
        )
        # use_fast=True to match the transformer backend / vllm-omni server
        # (avoids silent fast/slow image-processor output differences).
        try:
            self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path, use_fast=True)
        except TypeError:
            log.warning("Qwen2_5OmniProcessor.from_pretrained() 不支持 use_fast 参数，使用默认加载方式")
            self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)

    # -----------------------------------------------------------------
    def _make_sampling_params(self, max_tokens: int):
        from vllm import SamplingParams  # type: ignore
        return SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=max_tokens)

    def generate(self, req: InferenceRequest) -> str:
        """Single-sample inference. Wrapped in a short retry to self-heal
        occasional vLLM V1-engine state hiccups (same request repeated
        succeeds immediately after)."""
        import io as _io
        import contextlib as _ctxlib

        sp = self._make_sampling_params(req.generation_kwargs.get("max_new_tokens", 10))
        vllm_in = self._build_one(req)

        @retry(max_retries=3, base_delay=0.5, jitter=0.5)
        def _call():
            with _ctxlib.redirect_stderr(_io.StringIO()):
                return self.llm.generate([vllm_in], sampling_params=sp)

        try:
            outs = _call()
            out0 = outs[0] if outs else None
            result = out0.outputs[0].text if (out0 and out0.outputs) else ""
        finally:
            del vllm_in
            gc.collect()
        return result

    def generate_batch(self, reqs: List[InferenceRequest]) -> List[str]:
        """Real vLLM continuous-batching call: submits ALL prompts in
        ``reqs`` to a SINGLE ``llm.generate()`` call so vLLM's async
        scheduler can overlap their prefill/decode internally. Requires
        ``max_num_seqs`` (config/models/vllm.yaml) to be raised to actually
        get concurrent scheduling — it caps how many sequences run at once
        regardless of how many prompts are submitted per call.

        No internal try/except: if the whole call raises (e.g. a poisoned
        sample killing the shared engine), the exception propagates and
        ``Runner._run_batched()`` falls back to per-sample retries for this
        chunk — no results lost, just slower for that chunk.
        """
        import io as _io
        import contextlib as _ctxlib

        if not reqs:
            return []

        vllm_ins = [self._build_one(r) for r in reqs]
        max_tokens_list = [r.generation_kwargs.get("max_new_tokens", 10) for r in reqs]
        if len(set(max_tokens_list)) == 1:
            sp: Any = self._make_sampling_params(max_tokens_list[0])
        else:
            # vLLM accepts a list of SamplingParams matching prompts 1:1.
            sp = [self._make_sampling_params(mt) for mt in max_tokens_list]

        try:
            with _ctxlib.redirect_stderr(_io.StringIO()):
                outs = self.llm.generate(vllm_ins, sampling_params=sp)
            if len(outs) != len(vllm_ins):
                # Should never happen per vLLM's API contract; fail loudly
                # rather than silently misalign zip() downstream.
                raise RuntimeError(
                    f"vLLM returned {len(outs)} outputs for {len(vllm_ins)} inputs"
                )
            return [
                (out.outputs[0].text if (out and out.outputs) else "")
                for out in outs
            ]
        finally:
            del vllm_ins
            gc.collect()

    def _build_one(self, req: InferenceRequest) -> Dict[str, Any]:
        conv, _ = build_messages(
            req.sample, req.modality_mode,
            user_template=req.prompt_template or "",
            system=req.system_prompt or "",
            use_audio_in_video=self.use_audio_in_video,
            video_kwargs=self.video_kwargs,
        )
        text = self.processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False
        )
        mm_data: Dict[str, Any] = {}
        audios, images, videos = process_mm_info(
            conv, use_audio_in_video=self.use_audio_in_video
        )
        if images:
            mm_data["image"] = images[0] if len(images) == 1 else images
        if videos:
            mm_data["video"] = videos[0] if len(videos) == 1 else videos
        if audios:
            mm_data["audio"] = audios[0] if len(audios) == 1 else audios
        return {
            "prompt": text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": {"use_audio_in_video": self.use_audio_in_video},
        }
