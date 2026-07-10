"""ReAct Agent evaluation loop — orchestrates multi-turn interaction."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Set

from ..core.types import InferenceRequest, InferenceResult, MediaRef, Sample
from ..utils.io import append_jsonl, load_jsonl, rewrite_jsonl
from ..utils.logging import get_logger
from .action_parser import ParsedAction, parse_action_json
from .prompt import build_system_prompt, build_user_prompt
from .tools import ToolRegistry, ToolResult, VideoEnv

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

        all_samples: List[Sample] = list(self.dataset)
        log.info("Dataset '%s' loaded: %d samples", self.dataset.name, len(all_samples))

        done_uids = self._done_uids()
        if done_uids:
            log.info("Resume: skipping %d already-finished samples", len(done_uids))

        for s in all_samples:
            if s.uid in done_uids:
                continue
            result = self._evaluate_one(s)
            self._persist(result)

        self.model.close()
        total = len(self._all_items())
        correct = sum(1 for r in self._all_items() if r.get("is_correct"))
        acc = correct / total if total else 0.0
        log.info("Done. accuracy=%.2f%% total=%d", acc * 100, total)
        return {"total": total, "correct": correct, "accuracy": acc}

    # ── single-sample agent loop ──────────────────────────────────

    def _evaluate_one(self, sample: Sample) -> Dict[str, Any]:
        video = self._find_video(sample.media)
        if video is None:
            return self._error_result(sample.uid, "no video in sample media")

        env = VideoEnv(video.path)
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
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ]

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
                return {
                    "uid": sample.uid, "dataset": sample.dataset,
                    "question": sample.question, "choices": list(sample.choices),
                    "raw_output": answer, "parsed_answer": answer,
                    "correct_answer": sample.answer,
                    "is_correct": correct,
                    "latency_s": time.time() - t_start,
                    "error": None,
                    "meta": {"steps": step, "history": history,
                             "agent_confidence": parsed.confidence},
                }

            # 4) execute tool
            tool = ToolRegistry.get(parsed.action_type)
            if tool is None:
                history.append({"step": step, "error": f"unknown tool: {parsed.action_type}"})
                messages.append({"role": "user", "content": (
                    f"Unknown tool '{parsed.action_type}'. Available: "
                    + ", ".join(t.name for t in ToolRegistry.all())
                )})
                continue

            try:
                result = tool.execute(parsed.action, env)
            except Exception as e:
                log.warning("tool %s failed: %s", parsed.action_type, e)
                messages.append({"role": "user", "content": f"Tool error: {e}"})
                history.append({"step": step, "error": str(e)})
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

        # max steps reached without answer
        return self._error_result(sample.uid, f"max_steps={self.max_steps} reached without answer",
                                  {"steps": self.max_steps, "history": history})

    # ── helpers ────────────────────────────────────────────────────

    def _call_model(self, messages: List[Dict[str, Any]]) -> str:
        """Call the model backend with pre-built conversation messages."""
        req = InferenceRequest(
            sample=self._messages_to_sample(messages),
            modality_mode="text",  # media already inlined in messages
            generation_kwargs=self.gen_kwargs,
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
        """Replace old media content in assistant msgs with [MEDIA OMITTED]."""
        placeholder = [{"type": "text", "text": "[MEDIA OMITTED]"}]
        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue
            if isinstance(msg["content"], list) and len(msg["content"]) > 1:
                # keep only the first element if it's text, or just the placeholder
                messages[i]["content"] = placeholder

    # ── persistence ────────────────────────────────────────────────

    def _done_uids(self) -> Set[str]:
        done: Set[str] = set()
        for rec in self._all_items():
            if not rec.get("error") and rec.get("parsed_answer") is not None:
                done.add(rec.get("uid", ""))
        return done

    def _all_items(self) -> List[Dict[str, Any]]:
        return load_jsonl(self.items_path)

    def _persist(self, rec: Dict[str, Any]) -> None:
        append_jsonl(self.items_path, rec)
        if rec.get("error"):
            append_jsonl(self.failed_path, rec)

    def _error_result(self, uid: str, error: str,
                       extra_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "uid": uid, "dataset": "", "question": "", "choices": [],
            "raw_output": "", "parsed_answer": None, "correct_answer": None,
            "is_correct": False, "latency_s": 0.0, "error": error,
            "meta": extra_meta or {},
        }
