"""Qwen2.5-Omni vLLM batch backend.

Uses the **same** prompt / media construction as the Transformers backend
(:func:`~.local.qwen25omni.build_messages`).  Only the generation
engine differs.

Environment variables:
    VLLM_USE_V1                          — [2026-07-02] Do **NOT** set this
        to 0 on the pinned ``vllm==0.11.0`` (see env_init.sh). V0 has been
        fully removed upstream (official V1 guide: "我们已完全弃用 V0" /
        RFC #18571), and ``vllm/v1/engine/llm_engine.py``'s ``__init__``
        actively asserts ``envs.VLLM_USE_V1`` is truthy and raises::

            ValueError: Using V1 LLMEngine, but envs.VLLM_USE_V1=False.
            This should not happen. ...

        (confirmed by actually hitting this in the real
        ``agentomni`` conda env). An earlier revision of this file set
        ``os.environ.setdefault("VLLM_USE_V1", "0")`` here, mirroring
        Daily-Omni/test_model/Qwen2.5-Omni/testmodel.py::load_vllm_backend
        and FutureOmni/eval/infer_vllm.py — those were written against an
        older vLLM where V0/V1 coexisted and this was a genuinely valid
        workaround; on 0.11.0 there is no V0 to fall back to, so it just
        breaks engine init. Leave this env var **unset** entirely.
        The actual, version-relevant constraint is on V1 itself (confirmed
        in vLLM's own official Qwen2.5-Omni offline-inference example
        docs): "V1 engine does not support interleaved modalities yet" —
        i.e. ``use_audio_in_video=True`` (audio extracted from the video's
        own track, interleaved with video tokens) is simply unsupported
        under V1 (their own example asserts ``not envs.VLLM_USE_V1`` before
        allowing it — which would ALSO now raise the same ValueError above
        if you tried to force V0 to use it). This is why ``_build_one()``
        below sends video and audio as SEPARATE ``multi_modal_data``
        entries with ``use_audio_in_video=False`` (vLLM's
        officially-supported "mixed_modalities" pattern, no V1
        restriction) instead of the interleaved form — this also happens
        to match the transformer reference backend's encoding.
    VLLM_WORKER_MULTIPROC_METHOD=spawn   — avoid CUDA re-init in forked workers
    VLLM_DISABLE_PROGRESS_BAR=1          — suppress vLLM internal tqdm bars
"""
from __future__ import annotations

import gc
from typing import Any, Dict, List

from ...core.registry import register_model
from ...core.types import InferenceRequest
from ...prompt.templates import PromptTemplate, QWEN_OMNI_DEFAULT
from ...utils.logging import get_logger
from ...utils.retry import retry
from ..base import BaseModel
from ..local.qwen25omni import build_messages, _try_import_process_mm_info

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
        # [2026-07-02] enforce_eager / enable_prefix_caching: kept (harmless,
        # mirrors vllm_stage_textonly.yaml for the same model on the
        # openai/vllm-omni-server path) but these did NOT fix the ~50%
        # deterministic failure rate observed in a real run — see
        # mm_processor_cache_gb below for the confirmed actual root cause.
        self.enforce_eager = bool(cfg.get("enforce_eager", True))
        self.enable_prefix_caching = bool(cfg.get("enable_prefix_caching", False))
        # [2026-07-02 CONFIRMED ROOT CAUSE] Real full traceback (after
        # bumping runner.py's traceback.format_exc(limit=3) -> no limit)
        # showed the actual crash is NOT in llm.py::_validate_and_add_requests
        # itself (that frame is just the outer entrypoint) but deep inside
        # vllm/model_executor/models/qwen2_5_omni_thinker.py::
        # _maybe_apply_prompt_updates:
        #     use_audio_in_video = (all(
        #         item["use_audio_in_video"].data
        #         for item in mm_kwargs["video"]
        #     ))
        #     TypeError: 'NoneType' object is not subscriptable
        # i.e. one of the items in mm_kwargs["video"] is None. Cross-checked
        # against this repo's OWN vllm-omni fork of this exact function
        # (vllm-omni/vllm_omni/model_executor/models/qwen2_5_omni/
        # qwen2_5_omni_thinker.py::_maybe_apply_prompt_updates) which
        # explicitly guards against this:
        #     video_items = [item for item in mm_kwargs["video"] if item is not None]
        # The pinned vllm==0.11.0 does NOT have this guard. Root cause: vLLM's
        # multi-modal PROCESSOR cache (mm_processor_cache_gb, DEFAULT=4GB,
        # i.e. ON by default!) — a content-hash-keyed LRU cache of already-
        # processed mm features, completely separate from KV-cache
        # `enable_prefix_caching` above. When the SAME video (same content
        # hash) is submitted again in a later request (very common here:
        # Daily-Omni has many questions per video), the cache returns a
        # `None` placeholder for that item to signal "already cached,
        # reuse it" — and 0.11.0's use_audio_in_video auto-detection forgot
        # to skip Nones before subscripting. Confirmed by the failure
        # pattern: the exact same video succeeds on first use, then fails
        # intermittently on later reuse across different questions.
        # Fix: disable this cache entirely (per-sample sequential inference
        # here gains ~nothing from it anyway — each video is only ever
        # decoded once per question, and re-decoding on a genuine repeat is
        # far cheaper than getting wrong/crashed answers).
        self.mm_processor_cache_gb = float(cfg.get("mm_processor_cache_gb", 0))
        self.prompt_template = PromptTemplate.from_config(cfg, QWEN_OMNI_DEFAULT)
        self.llm = None
        self.processor = None
        self._process_mm_info = None

    def load(self) -> None:
        # Must be set BEFORE importing vllm — its internal modules
        # (LLMEngine, EngineCore, etc.) read this env var at import time,
        # not at engine-init time. Setting it in eval.sh's shell export
        # is NOT sufficient when vllm is a late import inside this method.
        import os as _os
        _os.environ.setdefault("VLLM_DISABLE_PROGRESS_BAR", "1")
        # Also suppress vLLM's own logger (INFO-level "Adding requests:",
        # "Processed prompts:" lines printed directly to stderr, separate
        # from the tqdm bars above). This is the only way to keep the
        # runner's OWN progress bar readable when max_num_seqs=1 makes
        # every single request trigger a one-item engine batch.
        import logging as _logging
        _logging.getLogger("vllm").setLevel(_logging.WARNING)

        from transformers import Qwen2_5OmniProcessor  # type: ignore
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
        # 显式 use_fast=True，跟 transformer 后端 / vllm-omni 在线服务保持一致
        # （HF 新版 Qwen2VLImageProcessor 默认已切到 fast，且官方警告 fast/slow
        # 两种实现输出会有细微差异，避免三条路径各自依赖隐式默认值）
        try:
            self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path, use_fast=True)
        except TypeError:
            log.warning("Qwen2_5OmniProcessor.from_pretrained() 不支持 use_fast 参数，使用默认加载方式")
            self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)
        self._process_mm_info = _try_import_process_mm_info()

    # -----------------------------------------------------------------
    def _resolve_template(self, req: InferenceRequest) -> PromptTemplate:
        if req.prompt_template:
            return PromptTemplate(user=req.prompt_template, system=self.prompt_template.system)
        return self.prompt_template

    def generate(self, req: InferenceRequest) -> str:
        """Single-sample inference with immediate memory release.

        [2026-07-02] Wrapped the actual ``self.llm.generate()`` call in a
        short retry. Observed in a real run: ~4% of samples (51/1197, spread
        across totally unrelated videos/categories/task_types — i.e. not a
        data problem) fail with the exact same
        ``TypeError: 'NoneType' object is not subscriptable`` at
        ``vllm/entrypoints/llm.py::_validate_and_add_requests`` (see
        ``failed.jsonl`` from that run). This looks like an intermittent
        vLLM V1-engine state hiccup from calling ``.generate()`` repeatedly
        on the same long-lived ``LLM`` instance in a loop (one request at a
        time, as Runner._run_sequential does) rather than anything wrong
        with our inputs — successful samples immediately before/after a
        failure use the exact same code path. A couple of quick retries
        self-heals most of these without needing a separate
        ``--rerun-failed`` pass after the whole dataset finishes.
        """
        import gc
        from vllm import SamplingParams  # type: ignore

        max_tokens = req.generation_kwargs.get("max_new_tokens", 10)
        sp = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=max_tokens)
        vllm_in = self._build_one(req)

        @retry(max_retries=3, base_delay=0.5, jitter=0.5)
        def _call():
            return self.llm.generate([vllm_in], sampling_params=sp)

        try:
            outs = _call()
            out0 = outs[0] if outs else None
            result = out0.outputs[0].text if (out0 and out0.outputs) else ""
        finally:
            # explicitly free the decoded tensors so they don't accumulate
            del vllm_in
            gc.collect()
        return result

    def generate_batch(self, reqs: List[InferenceRequest]) -> List[str]:
        return [self.generate(r) for r in reqs]

    def _build_one(self, req: InferenceRequest) -> Dict[str, Any]:
        import torch
        template = self._resolve_template(req)
        conv, _ = build_messages(req.sample, req.modality_mode, template)
        text = self.processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False
        )
        mm_data: Dict[str, Any] = {}
        if self._process_mm_info is not None:
            # [2026-07-02] Changed from True -> False. Two independent reasons:
            #   1. Correctness/consistency: the transformer reference backend
            #      ALWAYS uses use_audio_in_video=False (see
            #      ../local/qwen25omni.py::build_messages — "matches the
            #      original Daily-Omni evaluation": audio is sent as a
            #      separate 16kHz wav, NOT extracted from the video's own
            #      audio track). True changes the video's temporal position
            #      encoding vs. the transformer path.
            #   2. Actually load-bearing on the pinned vllm==0.11.0: per
            #      vLLM's OWN official Qwen2.5-Omni offline-inference example
            #      docs, "V1 engine does not support interleaved modalities
            #      yet" — use_audio_in_video=True IS the interleaved case and
            #      their own example asserts `not envs.VLLM_USE_V1` before
            #      allowing it. Since V0 has been fully removed by 0.11.0
            #      (see module docstring), V1 is the ONLY engine available,
            #      so use_audio_in_video=True would hit this unsupported path
            #      — most likely the real cause behind the previously-cited
            #      "vLLM stateful bug where a separate audio tensor corrupts
            #      the input_preprocessor for subsequent requests". Sending
            #      video+audio as separate multi_modal_data entries (this
            #      function's actual behavior) is vLLM's officially-supported
            #      "mixed_modalities" pattern, with no V1 restriction.
            audios, images, videos = self._process_mm_info(
                conv, use_audio_in_video=False
            )
            # [2026-07-02] Unwrap single-item lists to bare objects.
            # qwen_omni_utils.process_mm_info() always returns a *list* per
            # modality (even for exactly one image/video/audio, e.g.
            # videos == [tensor]) — mirroring the HF processor's expected
            # input shape. But vLLM's OWN official multi-modal examples
            # (vllm-omni/examples/offline_inference/qwen2_5_omni/end2end.py
            # ::get_mixed_modalities_query) pass a SINGLE bare object per
            # modality when there is only one item — no list wrapper:
            #   "audio": AudioAsset(...).audio_and_sample_rate,   # tuple
            #   "image": convert_image_mode(...),                 # PIL Image
            #   "video": VideoAsset(...).np_ndarrays,              # ndarray
            # A list value makes vLLM treat it as "multiple items of this
            # modality" and run it through its per-item content-hash /
            # multi_modal_uuids bookkeeping path — which is where the
            # intermittent (and, per real-run evidence, actually
            # deterministic-per-request-but-content-independent)
            # "TypeError: 'NoneType' object is not subscriptable" in
            # vllm/entrypoints/llm.py has been showing up. Daily-Omni /
            # OmniBench / OmniVideoBench in this repo all only ever have
            # exactly one video + one audio (+ optionally one image) per
            # sample, so always unwrapping is safe and matches the
            # official single-item usage pattern exactly.
            if images:
                mm_data["image"] = images[0] if len(images) == 1 else images
            if videos:
                mm_data["video"] = videos[0] if len(videos) == 1 else videos
            if audios:
                mm_data["audio"] = audios[0] if len(audios) == 1 else audios
        return {
            "prompt": text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": {"use_audio_in_video": False},
        }
