"""Unified evaluation Runner: orchestrates dataset → model → persist → report."""
from __future__ import annotations

import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set

from .utils.progress import ProgressManager
from .core.types import InferenceRequest, InferenceResult, Sample
from .eval.parser import choices_to_index2ans, extract_choice_letter
from .eval.report import write_summary
from .utils.io import (
    append_jsonl,
    atomic_write_yaml,
    load_jsonl,
    rewrite_jsonl,
)
from .utils.logging import get_logger

log = get_logger(__name__)


class Runner:
    """Orchestrate evaluation: load data → submit to model → parse → persist → summarize.

    Concurrency modes:
      * ``sequential`` — one-by-one (default for local GPU models)
      * ``thread``     — ``ThreadPoolExecutor`` (default for API models)
      * ``batch``      — ``model.generate_batch`` (default for vLLM)

    Mode auto-selection (when ``cfg.concurrency.mode == "auto"``):
      - ``model.supports_batch`` -> batch
      - ``model.is_thread_safe`` -> thread
      - else                     -> sequential
    """

    def __init__(self, dataset, model, cfg: Dict[str, Any]):
        self.dataset = dataset
        self.model = model
        self.cfg = cfg
        self.run_dir: str = cfg["run_dir"]
        os.makedirs(self.run_dir, exist_ok=True)
        self.items_path = os.path.join(self.run_dir, "items.jsonl")
        self.failed_path = os.path.join(self.run_dir, "failed.jsonl")
        # snapshot run config
        atomic_write_yaml(os.path.join(self.run_dir, "run_config.yaml"), cfg)

    # ------------------------------------------------------------------ public
    def run(self) -> Dict[str, Any]:
        log.info("Loading model: %s", self.model.name)
        self.model.load()

        # 断点重测：清理上一次（可能被中断的）运行留下的重复/过期记录，同时让
        # write_summary() 的统计不被同一个 uid 的多条历史记录污染。之后
        # _load_done_uids() 天然会把所有失败/未解析的样本重新纳入 pending——
        # 不需要单独的 --rerun-failed 步骤，每次跑都会自动重测之前失败的样本。
        self._compact_items()

        all_samples: List[Sample] = list(self.dataset)
        log.info("Dataset '%s' loaded: %d samples", self.dataset.name, len(all_samples))

        done_uids = self._load_done_uids()
        retry_uids = self._previously_failed_uids()
        if done_uids:
            log.info("Resume: skipping %d already-finished samples", len(done_uids))
        pending = [s for s in all_samples if s.uid not in done_uids]
        n_retry = sum(1 for s in pending if s.uid in retry_uids)
        if n_retry:
            log.info(
                "Auto-retry: %d/%d pending samples previously failed or "
                "unparsed — retrying them automatically (no --rerun-failed "
                "needed)", n_retry, len(pending),
            )

        # optional task_type filter — for quickly re-checking a specific
        # failure category (e.g. "Event Sequence") without evaluating the
        # whole dataset
        task_type_filter = self.cfg.get("task_type_filter")
        if task_type_filter:
            pending = [
                s for s in pending
                if str((s.meta or {}).get("task_type", "")) == task_type_filter
            ]
            log.info("Filtered to task_type=%r: %d pending samples", task_type_filter, len(pending))

        # apply optional sample limit / shuffling for debugging
        limit = self.cfg.get("limit")
        if limit:
            pending = pending[: int(limit)]
            log.info("Limited to first %d pending samples", len(pending))

        mode = self._resolve_mode()
        log.info("Concurrency mode = %s", mode)

        if not pending:
            log.info("Nothing to do — all samples already evaluated.")
        elif mode == "thread":
            self._run_threaded(pending)
        elif mode == "batch":
            self._run_batched(pending)
        elif mode == "sequential":
            self._run_sequential(pending)
        else:
            raise ValueError(f"Unknown concurrency mode: {mode}")

        # 这一轮新写入的成功记录跟旧的失败记录（同一个 uid）此时同时存在于
        # items.jsonl 里——run() 开头的那次 compact 只清理了"上一轮遗留"的重复，
        # 清不掉"这一轮刚产生"的重复，必须在算 summary 前再 compact 一次，
        # 否则重跑过的样本会被计两遍，total/accuracy 都会算错。
        if pending:
            self._compact_items()

        summary = write_summary(
            self.items_path,
            out_dir=self.run_dir,
            dataset_name=self.cfg["dataset"]["name"],
        )
        log.info(
            "Done. accuracy=%.2f%% valid=%d failed=%d total=%d",
            summary["accuracy"] * 100,
            summary["valid"],
            summary["failed"],
            summary["total"],
        )
        try:
            self.model.close()
        except Exception as e:  # pragma: no cover
            log.warning("model.close() raised: %s", e)
        return summary

    # ---------------------------------------------------------------- internals
    def _compact_items(self) -> None:
        """Rewrite ``items.jsonl`` keeping only the LATEST record per uid.

        ``_persist()`` only ever appends. Since every :meth:`run` call
        automatically re-attempts any uid that previously errored or failed
        to parse (see ``_load_done_uids``), a sample retried across multiple
        resume cycles would otherwise leave BOTH its stale failed record(s)
        AND its new record in ``items.jsonl`` — inflating ``total`` and
        skewing ``accuracy`` in :func:`write_summary`. Compacting (last
        occurrence wins, since the file is append-ordered) keeps the
        bookkeeping accurate no matter how many times a sample has been
        retried, and regenerates ``failed.jsonl`` to match — a uid that
        succeeded on retry must stop showing up there.
        """
        items = load_jsonl(self.items_path)
        if not items:
            return
        latest: Dict[str, Dict[str, Any]] = {}
        for rec in items:
            uid = rec.get("uid")
            if uid is not None:
                latest[uid] = rec  # later occurrence overwrites earlier one
        if len(latest) == len(items):
            return  # nothing stale to drop
        deduped = list(latest.values())
        log.info("Compacted items.jsonl: dropped %d stale duplicate record(s) "
                  "from earlier retries", len(items) - len(deduped))
        rewrite_jsonl(self.items_path, deduped)
        still_failed = [r for r in deduped if r.get("error")]
        if still_failed:
            rewrite_jsonl(self.failed_path, still_failed)
        elif os.path.exists(self.failed_path):
            # 全部重测成功——不留一个空的 failed.jsonl，直接删掉文件本身
            os.remove(self.failed_path)
            log.info("All previously-failed samples now pass — removed %s", self.failed_path)

    def _resolve_mode(self) -> str:
        mode = self.cfg.get("concurrency", {}).get("mode", "auto")
        if mode != "auto":
            return mode
        if getattr(self.model, "supports_batch", False):
            return "batch"
        if getattr(self.model, "is_thread_safe", False):
            return "thread"
        return "sequential"

    def _load_done_uids(self) -> Set[str]:
        done: Set[str] = set()
        for rec in load_jsonl(self.items_path):
            # a sample is "done" only if it completed without error AND parsed something
            if rec.get("error"):
                continue
            if rec.get("parsed_answer") is None:
                continue
            done.add(rec["uid"])
        return done

    def _previously_failed_uids(self) -> Set[str]:
        """uids that already have a record in ``items.jsonl`` but did NOT
        finish successfully (errored, or the answer couldn't be parsed) —
        exactly the ones ``_load_done_uids()`` excludes, so they'll show up
        in ``pending`` again. Only used for the "auto-retry: N samples" log
        line below — a way to actually SEE that retry is happening, since
        those uids are otherwise indistinguishable from never-attempted
        ones once mixed into ``pending``."""
        failed: Set[str] = set()
        for rec in load_jsonl(self.items_path):
            uid = rec.get("uid")
            if uid is not None and (rec.get("error") or rec.get("parsed_answer") is None):
                failed.add(uid)
        return failed

    def _build_req(self, sample: Sample) -> InferenceRequest:
        return InferenceRequest(
            sample=sample,
            modality_mode=self.cfg.get("modality_mode", "av"),
            prompt_template=self.cfg.get("prompt_template"),
            generation_kwargs=self.cfg.get("generation", {}) or {},
        )

    @staticmethod
    def _build_result(sample: Sample, raw: str, latency_s: float,
                       error: Optional[str] = None,
                       extra_meta: Optional[Dict[str, Any]] = None) -> InferenceResult:
        """Shared result construction for both sequential/threaded and
        batched inference paths, so answer-parsing / correctness logic
        only lives in one place."""
        parsed = None if error else extract_choice_letter(
            raw, index2ans=choices_to_index2ans(sample.choices)
        )
        meta = dict(sample.meta or {})
        if extra_meta:
            meta.update(extra_meta)
        return InferenceResult(
            uid=sample.uid,
            dataset=sample.dataset,
            question=sample.question,
            choices=list(sample.choices),
            raw_output=raw,
            parsed_answer=parsed,
            correct_answer=sample.answer,
            is_correct=(parsed is not None and sample.answer is not None
                        and parsed.upper() == str(sample.answer).strip().upper()),
            latency_s=latency_s,
            error=error,
            meta=meta,
        )

    def _infer_one(self, sample: Sample) -> InferenceResult:
        req = self._build_req(sample)
        t0 = time.time()
        try:
            raw = self.model.generate(req)
            return self._build_result(sample, raw, time.time() - t0)
        except Exception as e:
            log.warning("Sample %s failed: %s", sample.uid, e)
            # Full traceback (not just the outer frame, which for e.g. vLLM
            # is often just the generic entrypoint) kept in meta to help
            # diagnose intermittent multimodal-inference failures.
            return self._build_result(
                sample, "", time.time() - t0,
                error=f"{type(e).__name__}: {e}",
                extra_meta={"_traceback": traceback.format_exc()},
            )

    def _persist(self, res: InferenceResult) -> None:
        rec = res.to_dict()
        append_jsonl(self.items_path, rec)
        if res.error:
            append_jsonl(self.failed_path, rec)

    # ----- run modes
    def _run_sequential(self, pending: List[Sample]) -> None:
        with ProgressManager(len(pending), desc="Sequential") as pm:
            for s in pending:
                res = self._infer_one(s)
                self._persist(res)
                pm.update(is_failed=bool(res.error), is_correct=bool(res.is_correct))

    def _run_threaded(self, pending: List[Sample]) -> None:
        W = int(self.cfg.get("concurrency", {}).get("max_workers", 8))
        with ProgressManager(len(pending), desc=f"Thread x{W}") as pm, \
                ThreadPoolExecutor(max_workers=W) as pool:
            futures = [pool.submit(self._infer_one, s) for s in pending]
            for fut in as_completed(futures):
                res = fut.result()
                self._persist(res)
                pm.update(is_failed=bool(res.error), is_correct=bool(res.is_correct))

    def _run_batched(self, pending: List[Sample]) -> None:
        B = int(self.cfg.get("concurrency", {}).get("batch_size", 8))
        with ProgressManager(len(pending), desc=f"Batch x{B}") as pm:
            for i in range(0, len(pending), B):
                batch = pending[i:i + B]
                reqs = [self._build_req(s) for s in batch]
                t0 = time.time()
                try:
                    raws = self.model.generate_batch(reqs)
                except Exception as e:
                    log.warning("Batch %d failed (%s); falling back to per-sample.", i, e)
                    for s in batch:
                        res = self._infer_one(s)
                        self._persist(res)
                        pm.update(is_failed=bool(res.error), is_correct=bool(res.is_correct))
                    continue
                # split the batch's wall-clock time evenly across samples —
                # not exact per-sample timing, but far better than the
                # previous hardcoded 0.0
                latency = (time.time() - t0) / max(len(batch), 1)
                for s, raw in zip(batch, raws):
                    res = self._build_result(s, raw or "", latency)
                    self._persist(res)
                    pm.update(is_failed=bool(res.error), is_correct=bool(res.is_correct))
