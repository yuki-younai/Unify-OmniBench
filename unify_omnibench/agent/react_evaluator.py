"""ReAct Agent evaluation loop — orchestrates multi-turn interaction."""
from __future__ import annotations

import os
import re
import shutil
import time
import traceback
from typing import Any, Dict, List, Optional, Set

from ..core.types import InferenceRequest, InferenceResult, MediaRef, Sample
from ..eval.report import write_summary
from ..utils.io import append_jsonl, load_jsonl, rewrite_jsonl
from ..utils.logging import get_logger
from ..utils.progress import ProgressManager
from .action_parser import ParsedAction, parse_action_json, validate_action
from .prompt import build_system_prompt, build_user_prompt
from .tools import ToolRegistry, ToolResult, VideoEnv
from .trajectory_viz import save_trajectory

log = get_logger(__name__)


class ReActEvaluator:
    """Run Agent ReAct evaluation for a single dataset → model pair.

    Uses existing model backends unchanged — each turn calls
    ``model.generate(req)`` just like the direct mode.
    """

    def __init__(self, dataset, model, cfg: Dict[str, Any]):
        self.dataset = dataset
        self.model = model
        self.cfg = cfg
        self.run_dir: str = cfg["run_dir"]
        os.makedirs(self.run_dir, exist_ok=True)
        self.items_path = os.path.join(self.run_dir, "items.jsonl")
        self.failed_path = os.path.join(self.run_dir, "failed.jsonl")
        self.trajectory_dir = os.path.join(self.run_dir, "trajectories")
        self.react_cfg = dict(cfg.get("react", {}) or {})

        # env var override (from eval_react.sh MAX_STEPS_OVERRIDE)
        env_overrides = cfg.get("_react_env_overrides") or {}
        if env_overrides.get("MAX_STEPS_OVERRIDE"):
            self.react_cfg["max_steps"] = int(env_overrides["MAX_STEPS_OVERRIDE"])

        self.max_steps = int(self.react_cfg.get("max_steps", 32))
        self.gen_kwargs: Dict[str, Any] = dict(
            self.react_cfg.get("generation", {}) or {},
        )
        # 也从顶层 generation 读取 max_new_tokens（eval.sh 的 MAX_NEW_TOKENS）
        top_gen = cfg.get("generation", {}) or {}
        self.gen_kwargs.setdefault("max_new_tokens",
                                   top_gen.get("max_new_tokens", 4096))
        # 注入 react 配置到工具注册表，让工具读取 max_frames_len 等上限
        ToolRegistry.configure(self.react_cfg)

    def run(self) -> Dict[str, Any]:
        log.info("Loading model: %s", self.model.name)
        self.model.load()

        # 上一轮重跑遗留的重复记录先清一遍（同 uid 保留最后一次），
        # 避免总数/准确率被算重（同 Runner._compact_items 的用途）。
        self._compact_items()

        all_samples: List[Sample] = list(self.dataset)
        log.info("Dataset '%s' loaded: %d samples", self.dataset.name, len(all_samples))

        # 断点续跑：优先读本 shard 的 items.jsonl；若被上一轮 merge+cleanup
        # 删了，从父目录的 items.jsonl 恢复（按 shard_id 过滤出本 worker
        # 负责的 uid）——同 Runner._load_resume_items。
        items = self._load_resume_items()
        done_uids, retry_uids = self._scan_items(items)
        if done_uids:
            log.info("Resume: skipping %d already-finished samples", len(done_uids))
        pending = [s for s in all_samples if s.uid not in done_uids]
        n_retry = sum(1 for s in pending if s.uid in retry_uids)
        if n_retry:
            log.info(
                "Auto-retry: %d/%d pending samples previously failed or "
                "unparsed — retrying them automatically", n_retry, len(pending),
            )

        # 可选 task_type 过滤（快速复查某个失败类别）
        task_type_filter = self.cfg.get("task_type_filter")
        if task_type_filter:
            pending = [
                s for s in pending
                if str((s.meta or {}).get("task_type", "")) == task_type_filter
            ]
            log.info("Filtered to task_type=%r: %d pending samples", task_type_filter, len(pending))

        # 多 worker 分片：跟 Runner 用同一套 md5(uid)%num_shards 算法，
        # 保证同一个 uid 在 direct/react 两种模式下落到同一个 shard。
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

        # 可选 --limit（抽查用）
        limit = self.cfg.get("limit")
        if limit:
            pending = pending[: int(limit)]
            log.info("Limited to first %d pending samples", len(pending))

        # 进度条基线：分片/限量/task_type 过滤时没有"全量数据集"的统一基线，
        # 退化为按本次调用范围显示（同 Runner._progress_baseline）。
        sharded_or_filtered = (
            task_type_filter or limit or shard_id is not None
        )
        if sharded_or_filtered:
            bar_total, bar_initial, bar_correct, bar_failed = len(pending), 0, 0, 0
        else:
            prior_items = self._all_items()
            bar_total = len(all_samples)
            bar_initial = len(done_uids)
            bar_correct = sum(1 for r in prior_items if r.get("is_correct"))
            bar_failed = sum(1 for r in prior_items if r.get("error"))

        if not pending:
            log.info("Nothing to do — all samples already evaluated.")
        else:
            with ProgressManager(
                total=bar_total, desc=f"ReAct[{self.dataset.name}]",
                initial=bar_initial, initial_correct=bar_correct,
                initial_failed=bar_failed,
            ) as pbar:
                for s in pending:
                    # Defense-in-depth: _evaluate_one() already catches
                    # per-tool-call and per-model-call exceptions, but an
                    # unforeseen fatal error (e.g. a vLLM engine-level crash
                    # on malformed multimodal input) must NOT be allowed to
                    # kill this GPU worker process — that would silently
                    # abandon every remaining sample in the shard (see
                    # docs/AGENT_REACT_DESIGN.md for a real incident: a
                    # near-zero-length get_audio request crashed all 8
                    # workers with an uncaught vLLM ValueError).
                    try:
                        result = self._evaluate_one(s)
                    except Exception as e:
                        log.warning("Sample %s crashed the agent loop: %s", s.uid, e)
                        result = self._error_result(
                            s.uid, f"FATAL: {type(e).__name__}: {e}",
                            {"_traceback": traceback.format_exc()},
                            sample_meta=s.meta,
                        )
                    self._persist(result)
                    pbar.update(is_failed=bool(result.get("error")),
                                is_correct=bool(result.get("is_correct")))

        self.model.close()

        # 这一轮新写入的成功记录跟旧的失败记录（同一个 uid）此时同时存在于
        # items.jsonl 里，算 summary 前再 compact 一次，否则重跑过的样本会
        # 被计两遍（同 Runner.run() 末尾的二次 compact）。
        if pending:
            self._compact_items()

        summary = write_summary(
            self.items_path, out_dir=self.run_dir,
            dataset_name=self.cfg["dataset"]["name"],
        )
        log.info(
            "Done. accuracy=%.2f%% valid=%d failed=%d total=%d",
            summary["accuracy"] * 100, summary["valid"],
            summary["failed"], summary["total"],
        )
        return summary

    # ── single-sample agent loop ──────────────────────────────────

    def _evaluate_one(self, sample: Sample) -> Dict[str, Any]:
        video = self._find_video(sample.media)
        if video is None:
            return self._error_result(sample.uid, "no video in sample media",
                                       sample_meta=sample.meta)

        env = VideoEnv(video.path)
        try:
            return self._run_agent_loop(sample, env)
        finally:
            # VideoEnv.tmp_dir (holding this sample's extracted frames/
            # audio/clips) is never removed anywhere else — clean it up
            # unconditionally (success, tool error, or an escaped fatal
            # exception) to avoid unbounded disk growth over a full run.
            shutil.rmtree(env.tmp_dir, ignore_errors=True)

    def _run_agent_loop(self, sample: Sample, env: VideoEnv) -> Dict[str, Any]:
        meta = env.meta()

        # dynamic step: longer videos get proportionally more exploration steps
        max_steps = self.max_steps
        if self.react_cfg.get("dynamic_step", True):
            clip_len = self.react_cfg.get("max_clip_len", 60)
            max_steps = min(max_steps, 5 + int(meta["duration"] / max(clip_len, 1)))

        # build initial conversation
        system = build_system_prompt()
        user = build_user_prompt(sample.question, sample.choices, meta)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ]
        # [NOTICE] remaining-steps hint, injected before every model call —
        # mirrors OmniAgent's video_env.py::_append_step_notice (called once
        # in reset() for the first decision, then again after each step).
        self._append_step_notice(messages, step_count=0, max_steps=max_steps)

        history: List[Dict[str, Any]] = []
        parsed = ParsedAction(action_type="")
        t_start = time.time()

        for step in range(1, max_steps + 1):
            # 1) call model
            raw = self._call_model(messages)

            # 2) parse action
            parsed = parse_action_json(raw)
            history.append({"step": step, "raw": raw, "parsed_type": parsed.action_type})

            # 3) answer → done
            if parsed.action_type == "answer":
                answer = parsed.answer_content or ""
                correct = (
                    sample.answer is not None
                    and answer.strip().upper() == str(sample.answer).strip().upper()
                )
                # Snapshot current conversation for trajectory saving
                messages_snapshot = _deep_copy_messages(messages)
                # 保留 sample.meta（task_type/category 等）供 write_summary
                # 的 breakdown 统计使用 —— 同 Runner._build_result 的做法。
                result_meta = dict(sample.meta or {})
                result_meta.update({
                    "steps": step, "history": history,
                    "agent_confidence": parsed.confidence,
                    "duration": meta.get("duration", 0),
                })
                return {
                    "uid": sample.uid, "dataset": sample.dataset,
                    "question": sample.question, "choices": list(sample.choices),
                    "raw_output": answer, "parsed_answer": answer,
                    "correct_answer": sample.answer,
                    "is_correct": correct,
                    "latency_s": time.time() - t_start,
                    "error": None,
                    "meta": result_meta,
                    "_messages": messages_snapshot,
                }

            # 4) execute tool
            tool = ToolRegistry.get(parsed.action_type)
            if tool is None:
                err_txt = f"UNKNOWN_TOOL: '{parsed.action_type}'. Available: " + \
                          ", ".join(t.name for t in ToolRegistry.all())
                history.append({"step": step, "error": err_txt})
                messages.append({"role": "user", "content": [
                    {"type": "text", "text": f"[ERROR] {err_txt}"}
                ]})
                self._append_step_notice(messages, step_count=step, max_steps=max_steps)
                continue

            # validate required fields BEFORE execution (matches OmniAgent's
            # video_env.py::_parse_action strict field checks) — reject
            # malformed calls up front instead of silently defaulting to 0.
            val_err = validate_action(parsed.action_type, parsed.action)
            if val_err:
                history.append({"step": step, "error": val_err})
                messages.append({"role": "user", "content": [
                    {"type": "text", "text": f"[ERROR] INVALID_ACTION: {val_err}"}
                ]})
                self._append_step_notice(messages, step_count=step, max_steps=max_steps)
                continue

            try:
                result = tool.execute(parsed.action, env)
            except Exception as e:
                log.warning("tool %s failed: %s", parsed.action_type, e)
                messages.append({"role": "user", "content": [
                    {"type": "text", "text": f"[ERROR] {e}"}
                ]})
                history.append({"step": step, "error": str(e)})
                self._append_step_notice(messages, step_count=step, max_steps=max_steps)
                continue

            # 5) append assistant response + tool observation
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": [{"type": "text", "text": result.observation}]})

            # add new media to user content
            for m in result.media:
                if m.kind == "image":
                    messages[-1]["content"].append({"type": "image", "image": m.path})
                elif m.kind == "audio":
                    messages[-1]["content"].append({"type": "audio", "audio": m.path})
                elif m.kind == "video":
                    messages[-1]["content"].append({"type": "video", "video": m.path})

            # memory consolidation: replace old assistant media with placeholder
            self._consolidate_memory(messages)
            self._append_step_notice(messages, step_count=step, max_steps=max_steps)

        # max steps reached without answer
        messages_snapshot = _deep_copy_messages(messages)
        return self._error_result(
            sample.uid, f"max_steps={self.max_steps} reached without answer",
            {"steps": self.max_steps, "history": history,
             "duration": meta.get("duration", 0)},
            _messages=messages_snapshot,
            sample_meta=sample.meta,
        )

    # ── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _append_step_notice(messages: List[Dict[str, Any]], step_count: int,
                             max_steps: int) -> None:
        """Append a "[NOTICE] Step X/Y. Z steps remaining." hint to the
        LAST user message, so the model always knows its remaining budget.

        Mirrors OmniAgent's ``video_env.py::_append_step_notice``:
        - ``remain <= 1`` (i.e. this is the model's last chance) →
          "[NOTICE] FINAL STEP! You MUST provide your answer now."
        - otherwise → normal countdown text.

        Appended AFTER ``_consolidate_memory()`` runs for that turn, so it
        naturally gets discarded (not accumulated) the next time that
        message's media/content block gets collapsed — no separate
        "strip old notices" pass needed.
        """
        remain = max_steps - step_count
        if remain <= 1:
            text = "\n[NOTICE] FINAL STEP! You MUST provide your answer now."
        else:
            text = f"\n[NOTICE] Step {step_count}/{max_steps}. {remain} steps remaining."
        if messages and messages[-1]["role"] == "user":
            content = messages[-1].get("content")
            if isinstance(content, list):
                content.append({"type": "text", "text": text})
            else:
                messages[-1]["content"] = [
                    {"type": "text", "text": str(content or "")},
                    {"type": "text", "text": text},
                ]

    def _call_model(self, messages: List[Dict[str, Any]]) -> str:
        """Call the model backend with pre-built conversation messages."""
        req = InferenceRequest(
            sample=self._messages_to_sample(messages),
            modality_mode="text",
            generation_kwargs=self.gen_kwargs,
            messages=messages,
        )
        raw = self.model.generate(req)
        return raw

    @staticmethod
    def _messages_to_sample(messages: List[Dict[str, Any]]) -> Sample:
        """Pack conversation into a Sample so backends can process it."""
        last = messages[-1]["content"]
        text = last if isinstance(last, str) else next(
            (b["text"] for b in last if b.get("type") == "text"), ""
        )
        return Sample(uid="agent", dataset="agent", question=text, choices=[], media=[])

    @staticmethod
    def _find_video(media: List[MediaRef]) -> Optional[MediaRef]:
        for m in media:
            if m.kind == "video":
                return m
        return None

    def _consolidate_memory(self, messages: List[Dict[str, Any]]) -> None:
        """Bound context growth: keep only the freshest tool-call media.

        Mirrors OmniAgent's ``video_env.py::_replace_old_media``:
        - Assistant messages: replace with a text placeholder (raw JSON not
          needed after the turn — history already records it).
        - User messages: only the LAST user message (this turn's tool
          observation) keeps its image/audio/video blocks; earlier user
          messages have their media blocks stripped down to a text
          placeholder that preserves the original header text and any
          timestamps (e.g. "Frames 10.0s-20.0s ... [MEDIA OMITTED ...]")
          so the model can still refer back to what/when it saw something.

        Without this, images/audio/video from every past ``get_frames`` /
        ``get_audio`` / ``get_clip`` call would accumulate forever and
        blow past vLLM's ``limit_mm_per_prompt`` (or OOM on long videos).
        """
        media_kinds = ("image", "audio", "video")
        suffix = "[MEDIA OMITTED - Refer to your Observation]"

        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                last_user_idx = i
                break

        for i, msg in enumerate(messages):
            if msg["role"] == "assistant":
                if isinstance(msg["content"], list) and len(msg["content"]) > 1:
                    messages[i]["content"] = [{"type": "text", "text": suffix}]
                continue
            if msg["role"] == "user" and i != last_user_idx:
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                has_image = any(
                    isinstance(b, dict) and b.get("type") == "image" for b in content
                )
                has_other_media = any(
                    isinstance(b, dict) and b.get("type") in ("video", "audio")
                    for b in content
                )
                if not (has_image or has_other_media):
                    continue

                # preserve the original header text (e.g. "[Frames 10.0s-20.0s ...]")
                header = "Media content"
                if content and isinstance(content[0], dict) and content[0].get("type") == "text":
                    header = content[0].get("text", header).strip()

                if has_image:
                    timestamps: List[str] = []
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "text":
                            timestamps.extend(re.findall(r"(\d+(?:\.\d+)?)s", b.get("text", "")))
                    ts_str = ", ".join(f"{float(t):.2f}s" for t in timestamps)
                    new_text = f"{header} Timestamps: [{ts_str}] {suffix}"
                else:
                    new_text = f"{header} {suffix}"

                messages[i]["content"] = [{"type": "text", "text": new_text}]

    # ── persistence / resume (mirrors Runner's equivalents) ──────────

    def _all_items(self) -> List[Dict[str, Any]]:
        return load_jsonl(self.items_path)

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
        parent_items_path = os.path.join(os.path.dirname(self.run_dir), "items.jsonl")
        if not os.path.exists(parent_items_path):
            return items
        import hashlib
        items = load_jsonl(parent_items_path)
        items = [r for r in items
                 if int(hashlib.md5((r.get("uid") or "").encode()).hexdigest(), 16)
                 % int(num_shards) == int(shard_id)]
        log.info("Resume from parent items.jsonl: %d records for shard %d/%d",
                  len(items), int(shard_id), int(num_shards))
        return items

    @staticmethod
    def _scan_items(items: List[Dict[str, Any]]) -> "tuple[Set[str], Set[str]]":
        """One pass over items.jsonl -> (done_uids, retry_uids).

        done:  finished without error and an answer was parsed -> skip.
        retry: has a record but not done -> stays pending, retried
               automatically next run (no extra flag needed).
        """
        done: Set[str] = set()
        retry: Set[str] = set()
        for rec in items:
            uid = rec.get("uid")
            if uid is None:
                continue
            if rec.get("error") or rec.get("parsed_answer") is None:
                retry.add(uid)
            else:
                done.add(uid)
        return done, retry

    def _compact_items(self) -> None:
        """Rewrite items.jsonl keeping only the LATEST record per uid.

        Same purpose as Runner._compact_items: a sample retried across
        resume cycles would otherwise leave both its stale failed record(s)
        AND its new record in items.jsonl, inflating totals / skewing
        accuracy in write_summary().
        """
        items = load_jsonl(self.items_path)
        if not items:
            return
        latest: Dict[str, Dict[str, Any]] = {}
        for rec in items:
            uid = rec.get("uid")
            if uid is not None:
                latest[uid] = rec
        if len(latest) == len(items):
            return
        deduped = list(latest.values())
        log.info("Compacted items.jsonl: dropped %d stale duplicate record(s)",
                  len(items) - len(deduped))
        rewrite_jsonl(self.items_path, deduped)
        still_failed = [r for r in deduped if r.get("error")]
        if still_failed:
            rewrite_jsonl(self.failed_path, still_failed)
        elif os.path.exists(self.failed_path):
            os.remove(self.failed_path)

    def _persist(self, rec: Dict[str, Any]) -> None:
        # strip internal _messages before writing to items.jsonl
        msgs = rec.pop("_messages", None)
        append_jsonl(self.items_path, rec)
        if rec.get("error"):
            append_jsonl(self.failed_path, rec)
        # save trajectory (OpenAI JSON + HTML) if messages exist
        if msgs:
            save_trajectory(rec, msgs, self.trajectory_dir)

    def _error_result(self, uid: str, error: str,
                       extra_meta: Optional[Dict[str, Any]] = None,
                       _messages: Optional[List[Dict[str, Any]]] = None,
                       sample_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        meta = dict(sample_meta or {})
        meta.update(extra_meta or {})
        return {
            "uid": uid, "dataset": "", "question": "", "choices": [],
            "raw_output": "", "parsed_answer": None, "correct_answer": None,
            "is_correct": False, "latency_s": 0.0, "error": error,
            "meta": meta,
            "_messages": _messages or [],
        }


def _deep_copy_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deep-copy conversation messages for trajectory persistence."""
    copied = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_content = []
            for block in content:
                if isinstance(block, dict):
                    new_content.append(dict(block))
                else:
                    new_content.append(block)
            copied.append({"role": msg.get("role", "user"), "content": new_content})
        else:
            copied.append({"role": msg.get("role", "user"), "content": str(content or "")})
    return copied
