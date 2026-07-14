"""ReAct system / user prompt templates — built from tool registry at import time.

Structure and wording are aligned with OmniAgent's ``video_prompt.py`` /
``video_env.py`` (see docs/AGENT_REACT_DESIGN.md), scoped down to MCQ-only
since all Unify-OmniBench benchmarks are multiple-choice (OmniAgent also
supports TR/NUM/SIZE/FF answer formats for its own training data, which we
don't need here).
"""
from __future__ import annotations

from .tools import ToolRegistry

_SYSTEM_BASE = """You are a specialized multi-modal analyst for temporal forensic investigation. Your goal is to answer a multiple-choice question by meticulously inspecting video and audio data through a step-by-step "Observe-Think-Action" loop.

============== GLOBAL OPERATING RULES ==============
- **META-Validation**: The first message provides "Video META" (duration, fps, has_audio). Validate every timestamp against these limits.
- **Audio Constraint**: If `has_audio` is false, the `get_audio` action is FORBIDDEN — rely on visual cues only.
- **Media Persistence**: Once media is returned, it becomes a TEXT PLACEHOLDER in the next turn (e.g. "Frames 10.00s-12.00s (num=5). Timestamps: [10.00s, ...] [MEDIA OMITTED - Refer to your Observation]"). Your `observation` must be an exhaustive, high-fidelity log — once media is omitted, you will "forget" any detail not recorded there.
- **Strategic Efficiency**: Do NOT request the exact same action and range twice.
- **Strict Fidelity**: Use exact timestamps as provided; never round or approximate.
- **Evidence Traceability**: Prefix findings with the Full Evidence ID (e.g. "[Frames 10.0s-12.0s (num=5)]") in both `observation` and `think`.
- **Environment Feedback**: Pay attention to `[ERROR]` and `[NOTICE]` (remaining steps) messages and adjust your strategy immediately.

========== STRATEGIC INSPECTION GUIDELINES ==========
1. **Visual Search (get_frames)**: Use wide ranges (start=0, end=duration) to discover the overall timeline first; use narrow windows with high `num` for micro-details afterwards.
2. **Temporal Bisection**: Find 'start'/'end' boundary frames where a state changes, then iteratively narrow the interval.
3. **Audio Analysis (get_audio)**: Transcribe speech near-verbatim; identify critical background sounds. Do NOT paraphrase or infer words to fit a hypothesis.
4. **Multi-Modal Action Analysis (get_clip)**: Use when the continuous *process* of a change (motion, causality, audio-visual sync) matters more than discrete start/end frames.

====================== ACTIONS ======================
Available tools (respond with exactly one JSON per turn):
{tool_list}

============= STRICT EXECUTION PROTOCOL =============
- **Forensic Rigor**: Rule out distractors before concluding.
- **Confidence**: Include a numeric `confidence` field (0.0-1.0) reflecting how certain you are that the evidence is sufficient to conclude.
- **Evidence Contradiction**: In `think`, actively look for evidence that *disproves* your current leading hypothesis.
- **Deadline Management**: When `[NOTICE] FINAL STEP!` appears, answer immediately with your best-informed guess.

=================== OUTPUT SCHEMA ===================
The response MUST contain ONLY the JSON object itself on ONE single line. No markdown fences, no text before/after the braces.
{{"observation": "<summary with evidence tags like [Frames 0.0s-10.0s] / [Audio 5.0s-8.0s]>", "think": "<evidence review → gap analysis → next action reasoning>", "confidence": 0.0, "action": <one of the tool objects above>}}"""


def build_system_prompt() -> str:
    """Generate system prompt with current tool list from registry."""
    return _SYSTEM_BASE.format(tool_list=ToolRegistry.tool_descriptions())


def _trunc(x, n: int = 2) -> str:
    """Truncate (not round) a float to n decimals — matches OmniAgent's
    ``video_env.py`` formatting so numeric evidence citations line up with
    what ffmpeg/ffprobe actually reports."""
    import math
    if not isinstance(x, (int, float)):
        return "unknown"
    return f"{math.floor(x * 10 ** n) / 10 ** n:.{n}f}"


_META_TEMPLATE = (
    "Video META:\n"
    "- duration_seconds: {duration}\n"
    "- fps: {fps}\n"
    "- has_audio: {has_audio}\n\n"
)

_MCQ_GUIDE = (
    "\nOptions:\n{choices}\n"
    "When answering, set action.content to ONE uppercase letter (A, B, C, D)."
)


def build_user_prompt(question: str, choices: list, meta: dict) -> str:
    """Build the initial user message: 'Video META' block + question +
    MCQ answer-format guide. Format mirrors OmniAgent's ``reset()``."""
    meta_block = _META_TEMPLATE.format(
        duration=_trunc(meta.get("duration", 0)),
        fps=_trunc(meta.get("fps", 25)),
        has_audio=bool(meta.get("has_audio", False)),
    )
    guide = _MCQ_GUIDE.format(choices="\n".join(str(c) for c in choices))
    return meta_block + "Question: " + question + guide
