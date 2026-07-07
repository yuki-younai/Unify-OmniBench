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

        all_samples: List[Sample] = list(self.dataset)
        log.info("Dataset '%s' loaded: %d samples", self.dataset.name, len(all_samples))

        done_uids = self._load_done_uids()
        if done_uids:
            log.info("Resume: skipping %d already-finished samples", len(done_uids))
        pending = [s for s in all_samples if s.uid not in done_uids]

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

    def rerun_failed(self) -> Dict[str, Any]:
        """Re-evaluate samples that errored or failed-to-parse on the previous run.

        Implementation: filter out failed records from ``items.jsonl``, then call
        :meth:`run`. The base dataset adapter will re-emit them; resume logic will
        skip the already-successful ones.
        """
        items = load_jsonl(self.items_path)
        keep = [
            x for x in items
            if not x.get("error") and x.get("parsed_answer")
        ]
        dropped = len(items) - len(keep)
        log.info("rerun-failed: dropping %d failed/unparsed items from items.jsonl", dropped)
        rewrite_jsonl(self.items_path, keep)
        # clear failed.jsonl too (it will be regenerated)
        if os.path.exists(self.failed_path):
            os.remove(self.failed_path)
        return self.run()

    # ---------------------------------------------------------------- internals
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

    def _build_req(self, sample: Sample) -> InferenceRequest:
        return InferenceRequest(
            sample=sample,
            modality_mode=self.cfg.get("modality_mode", "av"),
            prompt_template=self.cfg.get("prompt_template"),
            generation_kwargs=self.cfg.get("generation", {}) or {},
        )

    def _infer_one(self, sample: Sample) -> InferenceResult:
        req = self._build_req(sample)
        t0 = time.time()
        try:
            raw = self.model.generate(req)
            parsed = extract_choice_letter(raw, index2ans=choices_to_index2ans(sample.choices))
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
                latency_s=time.time() - t0,
                meta=dict(sample.meta or {}),
            )
        except Exception as e:
            log.warning("Sample %s failed: %s", sample.uid, e)
            return InferenceResult(
                uid=sample.uid,
                dataset=sample.dataset,
                question=sample.question,
                choices=list(sample.choices),
                raw_output="",
                parsed_answer=None,
                correct_answer=sample.answer,
                is_correct=False,
                latency_s=time.time() - t0,
                error=f"{type(e).__name__}: {e}",
                # [2026-07-02] Was limit=3 — too shallow to see where inside
                # vLLM's multimodal input processing a NoneType subscript
                # error actually originates (the outer frame is always
                # llm.py::_validate_and_add_requests / _add_request, which
                # itself doesn't raise TypeError — the real culprit is a
                # deeper, currently-invisible frame). Full trace needed to
                # pinpoint root cause; can be reverted once diagnosed.
                meta={**dict(sample.meta or {}), "_traceback": traceback.format_exc()},
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
                try:
                    raws = self.model.generate_batch(reqs)
                except Exception as e:
                    log.warning("Batch %d failed (%s); falling back to per-sample.", i, e)
                    for s in batch:
                        res = self._infer_one(s)
                        self._persist(res)
                        pm.update(is_failed=bool(res.error), is_correct=bool(res.is_correct))
                    continue
                for s, raw in zip(batch, raws):
                    parsed = extract_choice_letter(
                        raw, index2ans=choices_to_index2ans(s.choices)
                    )
                    res = InferenceResult(
                        uid=s.uid,
                        dataset=s.dataset,
                        question=s.question,
                        choices=list(s.choices),
                        raw_output=raw or "",
                        parsed_answer=parsed,
                        correct_answer=s.answer,
                        is_correct=(parsed is not None and s.answer is not None
                                    and parsed.upper() == str(s.answer).strip().upper()),
                        meta=dict(s.meta or {}),
                    )
                    self._persist(res)
                    pm.update(is_failed=bool(res.error), is_correct=bool(res.is_correct))
