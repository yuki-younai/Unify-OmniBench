"""Prompt templates — model-specific with optional benchmark overrides.

Usage
-----
Each model backend provides a default template.  Benchmarks can override it
in their YAML config via ``prompt_template`` / ``system_prompt`` fields:

.. code-block:: yaml

    # config/datasets/my_bench.yaml
    name: my_bench
    ...
    prompt_template: |
      基于以下{media_desc}回答问题：
      {question}
      选项：
      {choices}
    system_prompt: "你是一个多模态评测助手。"

New benchmarks that don't override these fields get the model-default template.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], default: "PromptTemplate") -> "PromptTemplate":
        """Merge a YAML config's prompt fields with a default template.

        If the config provides ``prompt_template`` and/or ``system_prompt``,
        they take precedence over *default*.
        """
        return cls(
            user=cfg.get("prompt_template") or default.user,
            system=cfg.get("system_prompt") or default.system,
        )


# ── Shared user prompt (all backends use the same text) ─────────────────

_USER_PROMPT_NORM = (
    "Your task is to accurately answer multiple-choice questions "
    "based on the {media_desc}.\n"
    "Select the single most accurate answer from the given choices.\n"
    "Question: {question}\n"
    "Choices: {choices}\n"
    "Your answer should be a capital letter representing your choice: "
    "A, B, C, or D. Don't generate any other text."
)

_USER_PROMPT_COT = (
    "Your task is to accurately answer multiple-choice questions "
    "based on the {media_desc}.\n\n"
    "First, analyze the visual and audio information step by step, "
    "explaining your reasoning.\n"
    "Then provide your final answer on a new line as: ANSWER: X\n\n"
    "Question: {question}\n"
    "Choices: {choices}"
)

# ── Per-backend defaults (system prompt only differs) ──────────────────

QWEN_OMNI_DEFAULT = PromptTemplate(
    system=(
        "You are Qwen, a virtual human developed by the Qwen Team, "
        "Alibaba Group, capable of perceiving auditory and visual "
        "inputs, as well as generating text and speech."
    ),
    user=_USER_PROMPT_NORM,
)

OPENAI_DEFAULT = PromptTemplate(
    system="You are a multimodal evaluator. Answer with one letter only (A/B/C/D).",
    user=_USER_PROMPT_NORM,
)
