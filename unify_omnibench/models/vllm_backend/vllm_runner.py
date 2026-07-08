"""Qwen2.5-Omni vLLM batch backend.

Uses the **same** prompt / media construction as the Transformers backend
(:func:`~.local.qwen25omni.build_messages`).  Only the generation
engine differs.

Environment variables:
    VLLM_USE_V1                          — [2026-07-02] Do **NOT** set this
        to 0. V0 has been fully removed upstream since vllm==0.11.0
        (official V1 guide: "我们已完全弃用 V0" / RFC #18571), and
        ``vllm/v1/engine/llm_engine.py``'s ``__init__`` actively asserts
        ``envs.VLLM_USE_V1`` is truthy and raises::

            ValueError: Using V1 LLMEngine, but envs.VLLM_USE_V1=False.
            This should not happen. ...

        (confirmed by actually hitting this in the real
        ``agentomni`` conda env). An earlier revision of this file set
        ``os.environ.setdefault("VLLM_USE_V1", "0")`` here, mirroring
        Daily-Omni/test_model/Qwen2.5-Omni/testmodel.py::load_vllm_backend
        and FutureOmni/eval/infer_vllm.py — those were written against an
        older vLLM where V0/V1 coexisted and this was a genuinely valid
        workaround; on any 0.11.0+ version there is no V0 to fall back to,
        so it just breaks engine init. Leave this env var **unset** entirely.
        Interleaved ``use_audio_in_video=True`` used to crash V1 on the
        previously pinned ``vllm==0.11.0`` (missing support — upstream issue
        #25473, fixed for Qwen2.5-Omni by PR #33605). Now pinned to
        ``vllm==0.17.0`` (see env_init.sh), past that fix; verified via
        ``tests/test_qwen_omni_vllm.py``'s interleaved scenario.
        ``use_audio_in_video`` is read from ``self.cfg`` (per-dataset
        override via ``dataset_config.yaml``) — Daily-Omni/OmniBench keep it
        ``False`` (independent audio file, sent as its own
        ``multi_modal_data`` entry); OmniVideoBench/WorldSense set it
        ``True`` (audio comes only from the video's own track, no separate
        audio file attached).
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
        # [2026-07-07] video 抽帧参数：跟 qwen25omni.py 用同一套 dataset_config.yaml
        # ``video`` 字典，在 _build_one() 里注入进 video content block，让
        # qwen_omni_utils 在解码源头就按这个预算采样（而不是解码默认的 768 帧再
        # 事后丢弃——这曾是 OmniVideoBench 长视频跑得比官方慢很多的主因）。
        video_cfg = dict(cfg.get("video") or {})
        self.video_kwargs: Dict[str, Any] = {
            k: v for k, v in {
                "fps": video_cfg.get("fps"),
                "max_frames": video_cfg.get("max_frames"),
                "min_frames": video_cfg.get("min_frames"),
                # 像素预算三元组 —— WorldSense (VLMEvalKit::Qwen2VLChat) 需要，
                # 跟 qwen25omni.py 用同一套键名，保持两个后端行为一致。
                "min_pixels": video_cfg.get("min_pixels"),
                "max_pixels": video_cfg.get("max_pixels"),
                "total_pixels": video_cfg.get("total_pixels"),
            }.items() if v is not None
        }
        # [2026-07-07] 按数据集覆盖（见 dataset_config.yaml::use_audio_in_video）。
        # [2026-07-08] 之前记录过 "V1 engine does not support interleaved
        # modalities yet" 的风险提示——已确认修复（升级 vllm 后过了上游
        # PR #33605，见本文件顶部 VLLM_USE_V1 说明），交织模式在 V1 上已验证
        # 可正常工作，不再是需要规避的高风险路径。
        self.use_audio_in_video = bool(cfg.get("use_audio_in_video", False))
        self.prompt_template = PromptTemplate.from_config(cfg, QWEN_OMNI_DEFAULT)
        self.llm = None
        self.processor = None
        self._process_mm_info = None

    def load(self) -> None:
        import os as _os
        import logging as _logging
        import warnings as _warnings
        _os.environ.setdefault("VLLM_DISABLE_PROGRESS_BAR", "1")
        _logging.getLogger("vllm").setLevel(_logging.ERROR)
        # 同 qwen25omni.py::load() 里的说明：压掉 qwen_omni_utils 的
        # per-element max_pixels 提示和 librosa 的重复 FutureWarning。
        _logging.getLogger("qwen_omni_utils").setLevel(_logging.ERROR)
        _warnings.filterwarnings("ignore", category=FutureWarning, module="librosa.*")

        from transformers import Qwen2_5OmniProcessor  # type: ignore
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase  # type: ignore

        # [2026-07-07] Version-compat shim: the transformers build pinned for
        # Qwen2.5-Omni support (a newer/dev commit, see the model.py comment
        # about `pip install git+...@3a1ead0...`) removed the legacy
        # ``all_special_tokens_extended`` property that pinned vllm==0.11.0's
        # own ``get_cached_tokenizer()`` (transformers_utils/tokenizer.py)
        # still reads at LLM-load time:
        #   AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended
        # This is a genuine cross-package version mismatch, not anything
        # wrong with our config. ``all_special_tokens_extended`` historically
        # differs from ``all_special_tokens`` only in that it may contain
        # ``AddedToken`` objects instead of plain strings — irrelevant for
        # vLLM's use here (just snapshotting special tokens for its tokenizer
        # cache), so falling back to ``all_special_tokens`` is a safe,
        # semantically-equivalent shim. Patched on the base class so it
        # covers whichever concrete tokenizer class vLLM instantiates.
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
        import io as _io
        import contextlib as _ctxlib
        from vllm import SamplingParams  # type: ignore

        max_tokens = req.generation_kwargs.get("max_new_tokens", 10)
        sp = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=max_tokens)
        vllm_in = self._build_one(req)

        @retry(max_retries=3, base_delay=0.5, jitter=0.5)
        def _call():
            # vLLM V1 engine prints "Adding requests" / "Processed prompts"
            # tqdm bars to stderr on every single request; suppress them.
            with _ctxlib.redirect_stderr(_io.StringIO()):
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
        """[2026-07-08] REAL vLLM continuous-batching call.

        Previously this just looped ``self.generate()`` one request at a
        time — functionally identical to ``Runner._run_sequential()`` (see
        ``config/models/vllm.yaml``'s ``concurrency_mode: sequential``,
        which was set precisely BECAUSE the old ``generate_batch()`` gave
        zero throughput benefit over sequential — no point picking "batch"
        mode when it wasn't actually batching anything). This version
        submits ALL prompts in ``reqs`` to ``self.llm.generate()`` in a
        SINGLE call, letting vLLM's async V1 scheduler interleave/overlap
        their prefill+decode steps internally (PagedAttention + continuous
        batching across concurrently in-flight sequences — this is vLLM's
        actual reason for existing; one-request-at-a-time throws that away).

        Verified via ``MODE=batch python tests/test_qwen_omni_vllm.py``:
        results for a given video/prompt are identical whether processed
        via this real-batch path or the single-request ``generate()``
        path, and ``self.llm.generate()`` is confirmed to return outputs in
        the SAME order as the input prompt list (vLLM's documented
        behavior), so the ``zip(batch, raws)`` alignment in
        ``Runner._run_batched()`` is safe.

        IMPORTANT — ``max_num_seqs`` must be raised for this to actually
        help: it caps how many sequences the engine schedules
        *concurrently* regardless of how many prompts are submitted in one
        ``.generate()`` call. With the default ``max_num_seqs=1`` (see
        ``config/models/vllm.yaml``), submitting N prompts here still gets
        processed one-at-a-time internally — no real speedup, just fewer
        Python-level calls. Set ``max_num_seqs`` >= ``concurrency.batch_size``
        to get genuine concurrent scheduling (start small, e.g. 4 — each
        multi-modal request is memory-heavy, watch for OOM).

        Error handling: intentionally NO internal try/except here. If the
        whole call raises (e.g. an ``EngineDeadError`` — a single poisoned
        sample can kill the shared engine for ALL in-flight requests, same
        risk that exists in sequential mode too, just realized inside one
        Python call instead of N), the exception propagates to
        ``Runner._run_batched()``, which already falls back to per-sample
        ``_infer_one()`` for this chunk (see runner.py) — so one bad sample
        degrades gracefully to slower-but-isolated retries instead of
        silently losing the whole batch.
        """
        import io as _io
        import contextlib as _ctxlib
        from vllm import SamplingParams  # type: ignore

        if not reqs:
            return []

        vllm_ins = [self._build_one(r) for r in reqs]
        max_tokens_list = [r.generation_kwargs.get("max_new_tokens", 10) for r in reqs]
        if len(set(max_tokens_list)) == 1:
            # Common case (one dataset run = one generation config): a
            # single shared SamplingParams object is enough.
            sp: Any = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=max_tokens_list[0])
        else:
            # Per-request max_tokens differ — vLLM accepts a list of
            # SamplingParams matching the prompts list 1:1.
            sp = [
                SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=mt)
                for mt in max_tokens_list
            ]

        try:
            with _ctxlib.redirect_stderr(_io.StringIO()):
                outs = self.llm.generate(vllm_ins, sampling_params=sp)
            if len(outs) != len(vllm_ins):
                # Defensive: should never happen per vLLM's API contract,
                # but a length mismatch would silently misalign zip() in
                # Runner._run_batched() — fail loudly instead so it falls
                # back to per-sample instead of returning wrong answers.
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
        mm_data: Dict[str, Any] = {}
        if self._process_mm_info is not None:
            # [2026-07-07] use_audio_in_video 现在按数据集从 self.use_audio_in_video
            # 读取（见 __init__ 里的说明），不再硬编码 False。
            # [2026-07-08] True 时的交织路径（audio 从视频容器提取，与video token
            # 交织排列）之前在 vLLM V1 上不受支持，现已随版本升级确认修复。False
            # 时 video/audio 作为两个独立的 multi_modal_data 条目发送（vLLM 官方
            # 支持的 mixed_modalities 模式），适用于有独立音频文件的场景
            # （Daily-Omni/OmniBench）。
            audios, images, videos = self._process_mm_info(
                conv, use_audio_in_video=self.use_audio_in_video
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
            "mm_processor_kwargs": {"use_audio_in_video": self.use_audio_in_video},
        }
