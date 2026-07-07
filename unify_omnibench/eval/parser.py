"""Cascade answer-letter extractor compatible with Daily-Omni / OmniBench / OmniVideoBench output styles."""
from __future__ import annotations

import re
from typing import Dict, Optional

LETTERS = ("A", "B", "C", "D", "E", "F")  # tolerate up to 6 options; most are A-D

# Pre-compiled patterns
_JSON_ANS_RE = re.compile(r'"answer"\s*:\s*"([A-F])"', re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{(?:\\text\{)?([A-F])\}?\}", re.IGNORECASE)
# OmniVideoBench original extract_model_answer also handles /box{X} (no backslash).
# Unify's cascade previously missed this because the backslash-prefixed \boxed{}
# is LaTeX-standard, but the official OmniVideoBench eval code explicitly looks
# for both variants.  Add it as a separate step so /box{B} is treated the same as
# \boxed{B} (both are common when the model is prompted to output /box{} format).
_SLASH_BOX_RE = re.compile(r"/box\{(?:\\text\{)?([A-F])\}?\}", re.IGNORECASE)
_STANDALONE_RE = re.compile(r"\b([A-F])\b")
_PAREN_RE = re.compile(r"[\(\[\<]([A-F])[\)\]\>]", re.IGNORECASE)


def extract_choice_letter(
    text: str,
    index2ans: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Extract a single answer letter from a model's free-form output.

    Strategy (in order):
      1) JSON: {"answer":"X"}                          (OmniVideoBench style)
      2) \\boxed{X} / \\boxed{\\text{X}}               (OmniVideoBench CoT style)
      3) /box{X} / /box{\\text{X}}                     (OmniVideoBench slash variant)
      4) Parenthesized letter:  (X) / [X] / <X>        (priority over reverse-lookup)
      5) First non-space char is A-F                   (Daily-Omni style)
      6) Option content reverse lookup via index2ans   (OmniBench style — before bare-letter
                                                        scan, otherwise "a banana" hits "A")
      7) First standalone \\b[A-F]\\b                  (Daily-Omni fallback)

    Returns ``None`` if nothing matches.
    """
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None

    # 1) JSON answer field
    m = _JSON_ANS_RE.search(s)
    if m:
        return m.group(1).upper()

    # 2) \boxed{X}
    m = _BOXED_RE.search(s)
    if m:
        return m.group(1).upper()

    # 3) /box{X}  (OmniVideoBench slash variant — original extract_model_answer handles this)
    m = _SLASH_BOX_RE.search(s)
    if m:
        return m.group(1).upper()

    # 4) (A) [A] <A>
    m = _PAREN_RE.search(s)
    if m:
        return m.group(1).upper()

    # 5) First char is a letter (Daily-Omni: model often outputs just "A")
    first = s[0]
    if first.upper() in LETTERS and (len(s) == 1 or not s[1].isalpha()):
        # require boundary: "A" / "A." / "A)" — but NOT "Apple"
        return first.upper()

    # 6) Option content reverse lookup (before bare-letter scan, so we don't
    #    mistake the article "a" or "I" for letters)
    if index2ans:
        s_low = s.lower()
        hits = [
            (letter.upper(), ans)
            for letter, ans in index2ans.items()
            if ans and ans.lower() in s_low
        ]
        if len(hits) == 1:
            return hits[0][0]

    # 7) Standalone letter \bA\b (last-resort)
    m = _STANDALONE_RE.search(s)
    if m:
        return m.group(1).upper()

    return None


def choices_to_index2ans(choices) -> Dict[str, str]:
    """Convert ``["A. foo", "B. bar"]`` or ``["foo", "bar"]`` to ``{"A":"foo", ...}``."""
    if not choices:
        return {}
    out: Dict[str, str] = {}
    letter_prefix_re = re.compile(r"^\s*([A-F])\s*[\.\):]\s*(.*)$", re.IGNORECASE)
    for i, c in enumerate(choices):
        if i >= len(LETTERS):
            break
        s = str(c).strip()
        m = letter_prefix_re.match(s)
        letter = LETTERS[i]
        if m:
            out[letter] = m.group(2).strip()
        else:
            out[letter] = s
    return out
