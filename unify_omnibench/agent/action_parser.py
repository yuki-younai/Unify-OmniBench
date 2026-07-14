"""Parse model output into structured action + observation/think/confidence."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ParsedAction:
    """Structured output from the Agent's JSON response."""
    action_type: str                     # "get_frames" | "get_audio" | "get_clip" | "answer"
    action: Dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    think: str = ""
    confidence: float = 0.0
    raw_json: str = ""

    @property
    def answer_content(self) -> Optional[str]:
        if self.action_type == "answer":
            return self.action.get("content")
        return None


def parse_action_json(raw: str) -> ParsedAction:
    """Extract JSON from model output and parse into ParsedAction.

    Handles common cases:
    - Pure JSON string
    - JSON wrapped in ```json...``` fences
    - JSON with leading/trailing text
    """
    text = raw.strip()

    # 1) ```json ... ``` fences
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()

    # 2) find outermost { ... }
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return ParsedAction(action_type="unknown", raw_json=text)

    if not isinstance(obj, dict):
        return ParsedAction(action_type="unknown", raw_json=text)

    action = obj.get("action", {})
    # Model sometimes returns "action" as a bare string (e.g. "get_frames")
    # instead of {"type": "get_frames", ...}. Normalize defensively.
    if isinstance(action, str):
        action = {"type": action}
    elif not isinstance(action, dict):
        action = {}

    confidence_raw = obj.get("confidence", 0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0

    return ParsedAction(
        action_type=action.get("type", "unknown"),
        action=action,
        observation=obj.get("observation", "") if isinstance(obj.get("observation"), str) else "",
        think=obj.get("think", "") if isinstance(obj.get("think"), str) else "",
        confidence=confidence,
        raw_json=text,
    )


# ── action field validation ─────────────────────────────────────────────
# Mirrors OmniAgent's video_env.py::_parse_action field checks: reject a
# tool call up front (before execution) if required fields are missing or
# have the wrong type, rather than silently defaulting to 0 and letting a
# malformed call through as if it were valid.

_REQUIRED_NUMERIC_FIELDS = {
    "get_frames": ("start", "end"),
    "get_audio": ("start", "end"),
    "get_clip": ("start", "end"),
}


def validate_action(action_type: str, action: Dict[str, Any]) -> Optional[str]:
    """Return an error message if the action is malformed, else ``None``.

    Checked (matching OmniAgent's ``_parse_action``):
    - get_frames / get_audio / get_clip: 'start'/'end' must be present and
      numeric (int or float).
    - get_frames: 'num' must additionally be present and an int.
    - answer: 'content' must be present and a string.
    """
    if action_type in _REQUIRED_NUMERIC_FIELDS:
        for fld in _REQUIRED_NUMERIC_FIELDS[action_type]:
            if fld not in action or not isinstance(action[fld], (int, float)):
                return f"{action_type} requires numeric '{fld}'"
        if action_type == "get_frames":
            if "num" not in action or not isinstance(action["num"], int):
                return "get_frames requires integer 'num'"
    elif action_type == "answer":
        if "content" not in action or not isinstance(action["content"], str):
            return "answer requires string 'content'"
    return None

