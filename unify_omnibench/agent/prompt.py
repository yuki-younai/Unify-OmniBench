"""ReAct system / user prompt templates — built from tool registry at import time."""
from __future__ import annotations

from .tools import ToolRegistry

_SYSTEM_BASE = """You are an AI agent with visual and audio perception capabilities.
Your task is to answer a question about a video by actively exploring its content.

Available tools (respond with exactly one JSON per turn):
{tool_list}

Response format (MUST be valid single-line JSON, no extra text):
{{
  "observation": "<summary of what you just perceived, with evidence tags like [Frames 0.0s-10.0s] or [Audio 5.0s-8.0s]>",
  "think": "<evidence review → gap analysis → next action reasoning>",
  "confidence": <0.0-1.0 how certain you are right now>,
  "action": <one of the tool objects above>
}}

Rules:
- You CANNOT answer on the first step (must explore at least once).
- Use [Frames Xs-Ys] / [Audio Xs-Ys] / [Clip Xs-Ys] tags to cite evidence in observation.
- Confidence < 0.9 → you likely need more evidence before answering.
- Previously viewed media is marked as [MEDIA OMITTED] to save context;
  rely on your own observation summaries to remember what you saw/heard.
- Each response must contain EXACTLY ONE JSON object on a single line."""


def build_system_prompt() -> str:
    """Generate system prompt with current tool list from registry."""
    return _SYSTEM_BASE.format(tool_list=ToolRegistry.tool_descriptions())


_USER_TEMPLATE = """Video: duration={duration:.1f}s, fps={fps:.1f}, has_audio={has_audio}

Question: {question}
Options: {choices}
(answer format: single capital letter A/B/C/D)

Begin by exploring the video. What would you like to do first?"""


def build_user_prompt(question: str, choices: list, meta: dict) -> str:
    return _USER_TEMPLATE.format(
        duration=float(meta.get("duration", 0)),
        fps=float(meta.get("fps", 25)),
        has_audio=bool(meta.get("has_audio", False)),
        question=question,
        choices="\n".join(str(c) for c in choices),
    )
