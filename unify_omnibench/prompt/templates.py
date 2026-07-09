"""Prompt templates.

Prompt 现在全部由 dataset_config.yaml 统一定义（system_prompt + prompt_template），
不再有 model-backend 级默认值。这里只保留 --mode cot 的特殊模板。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PromptTemplate:
    """A prompt template with placeholders.

    Supported placeholders:
      * ``{media_desc}`` — e.g. "given video and audio together"
      * ``{question}`` — the question text
      * ``{choices}`` — newline-joined choices
    """

    user: str
    system: Optional[str] = None

    def render(
        self,
        media_desc: str,
        question: str,
        choices: List[str],
    ) -> str:
        """Render the user prompt with actual values."""
        choices_text = "\n".join(str(c) for c in choices)
        return self.user.format(
            media_desc=media_desc,
            question=question,
            choices=choices_text,
        )

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"user": self.user}
        if self.system is not None:
            d["system"] = self.system
        return d


# ── CoT user prompt (only special template kept; --mode cot in run.py) ──

_USER_PROMPT_COT = (
    "Your task is to accurately answer multiple-choice questions "
    "based on the {media_desc}.\n\n"
    "First, analyze the visual and audio information step by step, "
    "explaining your reasoning.\n"
    "Then provide your final answer on a new line as: ANSWER: X\n\n"
    "Question: {question}\n"
    "Choices: {choices}"
)
