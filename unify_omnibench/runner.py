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
        if self.cfg.get("run_mode") == "react":
            from .agent.react_evaluator import ReActEvaluator
            evaluator = ReActEvaluator(self.dataset, self.model, self.cfg)
            return evaluator.run()

        log.info("Loading model: %s", self.model.name)
        self.model.load()

        # 清理上一轮遗留的重复/过期记录（失败样本会自动重跑，见 _scan_items）。
        self._compact_items()

        # 断点续跑：先读本 shard 的 items.jsonl；若被 merge+cleanup 删了，
        # 从父目录的 items.jsonl 恢复（按 shard_id 过滤出本 worker 负责的 uid）。
        items = self._load_resume_items()

        all_samples: List[Sample] = list(self.dataset)
        log.info("Dataset '%s' loaded: %d samples", self.dataset.name, len(all_samples))

        done_uids, retry_uids, correct_uids = self._scan_items(items)
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

        # 多 worker 分片：每个 worker 只负责 uid 的 MD5 % num_shards 命中归属的样本。
        # 注意：不能用 Python 内置 hash()——它默认每进程不同（PYTHONHASHSEED），
        # 会导致同一 uid 在不同 worker 进程算出不同 hash、同时落入多个 shard。
        shard_id = self.cfg.get("shard_id")
        num_shards = self.cfg.get("num_shards")
        if shard_id is not None and num_shards is not None:
            import hashlib
            pending = [
                s for s in pending
                if int(hashlib.md5(s.uid.encode()).hexdigest(), 16) % int(num_shards) == int(shard_id)
            ]
            log.info("Shard %d/%d: %d pending samples after shard filter",
                     int(shard_id), int(num_shards), len(pending))

        # apply optional sample limit / shuffling for debugging
        limit = self.cfg.get("limit")
        if limit:
            pending = pending[: int(limit)]
            log.info("Limited to first %d pending samples", len(pending))

        mode = self._resolve_mode()
        log.info("Concurrency mode = %s", mode)

        # 进度条基线：让重试/断点续跑的进度条接着全量数据集的进度往下走。
        bar_total, bar_initial, bar_correct = self._progress_baseline(
            all_samples, done_uids, correct_uids)

        if not pending:
            log.info("Nothing to do — all samples already evaluated.")
        elif mode == "thread":
            self._run_threaded(pending, bar_total, bar_initial, bar_correct)
        elif mode == "batch":
            self._run_batched(pending, bar_total, bar_initial, bar_correct)
        elif mode == "sequential":
            self._run_sequential(pending, bar_total, bar_initial, bar_correct)
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
        to parse (see ``_scan_items``), a sample retried across multiple
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

    def _load_resume_items(self) -> List[Dict[str, Any]]:
        """Load items.jsonl for resume. If shard file is missing (cleaned by
        merge --cleanup), fall back to parent dir and filter by shard."""
        items = load_jsonl(self.items_path)
        if items:
            return items
        shard_id = self.cfg.get("shard_id")
        num_shards = self.cfg.get("num_shards")
        if shard_id is None or num_shards is None:
            return items
        parent_items = os.path.join(os.path.dirname(self.run_dir), "items.jsonl")
        if not os.path.exists(parent_items):
            return items
        import hashlib
        items = load_jsonl(parent_items)
        items = [r for r in items
                 if int(hashlib.md5((r.get("uid") or "").encode()).hexdigest(), 16)
                 % int(num_shards) == int(shard_id)]
        log.info("Resume from parent items.jsonl: %d records for shard %d/%d",
                 len(items), int(shard_id), int(num_shards))
        return items

    @staticmethod
    def _scan_items(items: List[Dict[str, Any]]) -> tuple[Set[str], Set[str], Set[str]]:
        """One pass over items.jsonl -> (done_uids, retry_uids, correct_uids).

        done:    finished without error and an answer was parsed -> skip.
        retry:   has a record but not done -> stays in pending, retried
                 automatically next run (no --rerun-failed needed).
        correct: subset of done that was also correct (for progress baseline).
        """
        done: Set[str] = set()
        retry: Set[str] = set()
        correct: Set[str] = set()
        for rec in items:
            uid = rec.get("uid")
            if uid is None:
                continue
            if rec.get("error") or rec.get("parsed_answer") is None:
                retry.add(uid)
            else:
                done.add(uid)
                if rec.get("is_correct"):
                    correct.add(uid)
        return done, retry, correct

    def _progress_baseline(self, all_samples: List[Sample], done_uids: Set[str],
                            correct_uids: Set[str]):
        """(bar_total, initial_completed, initial_correct) so a resumed run's
        progress bar continues the dataset-wide count instead of restarting
        at 0. Falls back to per-invocation bar (None, 0, 0) when there's no
        well-defined dataset-wide baseline to resume against (quickcheck or
        shard mode — each shard only processes a subset of the full dataset).
        """
        if self.cfg.get("task_type_filter") or self.cfg.get("limit") \
                or self.cfg.get("shard_id") is not None:
            return None, 0, 0
        return len(all_samples), len(done_uids), len(correct_uids)

    def _build_req(self, sample: Sample) -> InferenceRequest:
        return InferenceRequest(
            sample=sample,
            modality_mode=self.cfg.get("modality_mode", "av"),
            prompt_template=self.cfg.get("prompt_template"),
            system_prompt=self.cfg.get("system_prompt"),
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

    def _persist_and_update(self, res: InferenceResult, pm: ProgressManager) -> None:
        """Shared tail of every inference path below: persist the result,
        then advance the progress bar with it — kept in one place so the
        four call sites (sequential / threaded / batch-success /
        batch-fallback) can't drift out of sync."""
        self._persist(res)
        pm.update(is_failed=bool(res.error), is_correct=bool(res.is_correct))

    # ----- run modes
    # ``bar_total``/``bar_initial``/``bar_correct`` come from
    # ``_progress_baseline()`` — ``bar_total=None`` means "no dataset-wide
    # baseline available" (quickcheck runs), fall back to len(pending).
    def _run_sequential(self, pending: List[Sample], bar_total: Optional[int] = None,
                         bar_initial: int = 0, bar_correct: int = 0) -> None:
        with ProgressManager(bar_total or len(pending), desc="Sequential",
                              initial=bar_initial, initial_correct=bar_correct) as pm:
            for s in pending:
                self._persist_and_update(self._infer_one(s), pm)

    def _run_threaded(self, pending: List[Sample], bar_total: Optional[int] = None,
                       bar_initial: int = 0, bar_correct: int = 0) -> None:
        W = int(self.cfg.get("concurrency", {}).get("max_workers", 8))
        with ProgressManager(bar_total or len(pending), desc=f"Thread x{W}",
                              initial=bar_initial, initial_correct=bar_correct) as pm, \
                ThreadPoolExecutor(max_workers=W) as pool:
            futures = [pool.submit(self._infer_one, s) for s in pending]
            for fut in as_completed(futures):
                self._persist_and_update(fut.result(), pm)

    def _run_batched(self, pending: List[Sample], bar_total: Optional[int] = None,
                      bar_initial: int = 0, bar_correct: int = 0) -> None:
        B = int(self.cfg.get("concurrency", {}).get("batch_size", 8))
        with ProgressManager(bar_total or len(pending), desc=f"Batch x{B}",
                              initial=bar_initial, initial_correct=bar_correct) as pm:
            for i in range(0, len(pending), B):
                batch = pending[i:i + B]
                reqs = [self._build_req(s) for s in batch]
                t0 = time.time()
                try:
                    raws = self.model.generate_batch(reqs)
                except Exception as e:
                    log.warning("Batch %d failed (%s); falling back to per-sample.", i, e)
                    for s in batch:
                        self._persist_and_update(self._infer_one(s), pm)
                    continue
                # split the batch's wall-clock time evenly across samples —
                # not exact per-sample timing, but far better than the
                # previous hardcoded 0.0
                latency = (time.time() - t0) / max(len(batch), 1)
                for s, raw in zip(batch, raws):
                    self._persist_and_update(self._build_result(s, raw or "", latency), pm)
