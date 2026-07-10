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

    action = obj.get("action", {})
    return ParsedAction(
        action_type=action.get("type", "unknown"),
        action=action,
        observation=obj.get("observation", ""),
        think=obj.get("think", ""),
        confidence=float(obj.get("confidence", 0)),
        raw_json=text,
    )
