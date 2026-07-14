"""Generate HTML trajectory visualizations for Agent ReAct runs.

Creates a self-contained HTML file per sample showing the agent's
step-by-step reasoning, tool calls, observations, and final answer.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agent Trajectory — {uid}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f5f7fa;color:#1a1a2e;line-height:1.6;padding:24px}}
.container{{max-width:900px;margin:0 auto}}
.header{{background:#fff;border-radius:12px;padding:24px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.header .uid{{font-size:12px;color:#888;text-transform:uppercase;letter-spacing:.5px}}
.header .question{{font-size:18px;font-weight:600;margin:12px 0;color:#1a1a2e}}
.choices{{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}}
.choice{{background:#f0f2f5;border-radius:8px;padding:6px 14px;font-size:14px}}
.choice.correct{{background:#d4edda;color:#155724;border:1px solid #c3e6cb}}
.choice.chosen{{border:2px solid #667eea}}
.outcome{{display:flex;align-items:center;gap:10px;margin-top:14px}}
.badge{{display:inline-block;padding:4px 12px;border-radius:20px;font-size:13px;font-weight:600}}
.badge.ok{{background:#d4edda;color:#155724}}
.badge.fail{{background:#f8d7da;color:#721c24}}
.badge.error{{background:#fff3cd;color:#856404}}
.meta{{font-size:13px;color:#666;margin-top:8px}}
.trajectory{{margin-top:16px}}
.step{{background:#fff;border-radius:10px;padding:18px;margin-bottom:12px;box-shadow:0 1px 3px rgba(0,0,0,.06);border-left:4px solid #e0e0e0;transition:border-color .2s}}
.step:hover{{border-left-color:#667eea}}
.step-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}}
.step-num{{font-weight:700;font-size:14px;color:#667eea}}
.step-type{{font-size:12px;text-transform:uppercase;letter-spacing:.5px;padding:2px 8px;border-radius:4px}}
.step-type.tool{{background:#e8f0fe;color:#1967d2}}
.step-type.answer{{background:#d4edda;color:#155724}}
.step-type.error{{background:#f8d7da;color:#721c24}}
.step-type.unknown{{background:#f0f2f5;color:#666}}
.think{{background:#fafbfc;border-radius:8px;padding:12px;margin-bottom:10px;font-size:13px;color:#555;border:1px dashed #ddd;white-space:pre-wrap;max-height:200px;overflow-y:auto}}
.think::before{{content:'💭 Thinking';display:block;font-size:11px;color:#888;margin-bottom:6px;text-transform:uppercase;letter-spacing:.5px}}
.tool-call{{background:#f0f5ff;border-radius:8px;padding:12px;margin-bottom:8px;font-size:13px}}
.tool-call .tool-name{{font-weight:600;color:#1967d2}}
.tool-call .tool-args{{font-size:12px;color:#666;margin-top:4px;font-family:monospace}}
.observation{{background:#f0fff4;border-radius:8px;padding:12px;font-size:13px;color:#22543d;white-space:pre-wrap;max-height:250px;overflow-y:auto}}
.observation::before{{content:'📋 Observation';display:block;font-size:11px;color:#888;margin-bottom:6px;text-transform:uppercase;letter-spacing:.5px}}
.raw-json{{background:#1e1e2e;color:#cdd6f4;border-radius:8px;padding:12px;font-size:12px;font-family:monospace;white-space:pre-wrap;overflow-x:auto;max-height:300px;overflow-y:auto;margin-top:6px;display:none}}
.raw-toggle{{font-size:11px;color:#667eea;cursor:pointer;margin-top:4px;display:inline-block;user-select:none}}
.raw-toggle:hover{{text-decoration:underline}}
.final-answer{{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;border-radius:10px;padding:20px;margin-top:16px;text-align:center}}
.final-answer .label{{font-size:12px;opacity:.8;text-transform:uppercase;letter-spacing:1px}}
.final-answer .answer-text{{font-size:20px;font-weight:700;margin:8px 0}}
.footer{{text-align:center;font-size:12px;color:#aaa;margin-top:24px;padding:16px}}
.media-tag{{display:inline-block;background:#e8f0fe;color:#1967d2;border-radius:4px;padding:2px 6px;font-size:11px;margin:2px 4px}}
</style>
<script>
function toggleRaw(id){{var e=document.getElementById(id);if(e)e.style.display=e.style.display==='block'?'none':'block'}}
</script>
</head>
<body>
<div class="container">
<div class="header">
<div class="uid">Sample #{uid}</div>
<div class="question">{question}</div>
<div class="choices">{choices_html}</div>
<div class="outcome">{outcome_html}</div>
<div class="meta">Steps: {steps} &ensp;|&ensp; Duration: {duration:.1f}s &ensp;|&ensp; Latency: {latency:.1f}s</div>
</div>

<div class="trajectory">
<h3 style="font-size:15px;color:#666;margin-bottom:12px">Agent Trajectory</h3>
{steps_html}
</div>

<div class="final-answer">
<div class="label">{outcome_label}</div>
<div class="answer-text">{answer_text}</div>
</div>

<div class="footer">Unify-OmniBench Agent ReAct &mdash; {uid}</div>
</div>
</body>
</html>"""


def _choice_html(letter: str, text: str, is_correct: bool,
                 is_chosen: bool) -> str:
    cls = []
    if is_correct:
        cls.append("correct")
    if is_chosen:
        cls.append("chosen")
    cls_str = " ".join(cls)
    return f'<span class="choice {cls_str}"><b>{letter}.</b> {_escape(text)}</span>'


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _step_html(step: Dict[str, Any], step_idx: int) -> str:
    parts = []
    step_num = step.get("step", step_idx + 1)
    parsed_type = step.get("parsed_type", "unknown")

    type_cls = parsed_type if parsed_type in ("tool", "answer") else (
        "error" if step.get("error") else "unknown")
    tool_display = (parsed_type if parsed_type not in ("answer", "unknown")
                    else "action")

    parts.append(
        f'<div class="step">'
        f'<div class="step-header">'
        f'<span class="step-num">Step {step_num}</span>'
        f'<span class="step-type {type_cls}">{tool_display}</span>'
        f'</div>'
    )

    # raw JSON toggle
    raw_id = f"raw-{step_num}"
    raw_json = _escape(step.get("raw", ""))
    if raw_json.strip():
        parts.append(
            f'<div class="raw-json" id="{raw_id}">{raw_json}</div>'
            f'<span class="raw-toggle" onclick="toggleRaw(\'{raw_id}\')">'
            f'&#9654; View raw model output</span>'
        )

    if step.get("error"):
        parts.append(
            f'<div style="color:#721c24;font-size:13px;margin-top:4px">'
            f'&#9888; {_escape(step["error"])}</div>'
        )

    parts.append('</div>')
    return "\n".join(parts)


def generate_trajectory_html(result: Dict[str, Any],
                              media_paths: List[str] | None = None) -> str:
    """Generate a self-contained HTML file for one agent trajectory.

    Args:
        result: The result dict from ReActEvaluator._evaluate_one()
        media_paths: Optional list of media file paths mentioned in the trajectory.
    """
    uid = result.get("uid", "unknown")
    question = _escape(result.get("question", ""))
    choices = result.get("choices", [])
    correct_answer = result.get("correct_answer", "")
    parsed_answer = result.get("parsed_answer", "")
    is_correct = result.get("is_correct", False)
    error = result.get("error")
    meta = result.get("meta", {})
    steps_count = meta.get("steps", 0)
    history = meta.get("history", [])
    latency = result.get("latency_s", 0)

    # media_tags = " ".join(  # noqa: E800
    #     f'<span class="media-tag">{os.path.basename(p)}</span>'
    #     for p in (media_paths or [])
    # )

    # choices HTML
    choice_letters = [chr(ord('A') + i) for i in range(len(choices))]
    choices_parts = []
    for i, ch in enumerate(choices):
        letter = choice_letters[i]
        choices_parts.append(_choice_html(
            letter, ch,
            is_correct=(letter == correct_answer),
            is_chosen=(letter == parsed_answer.upper()),
        ))
    choices_html = "\n".join(choices_parts)

    # outcome
    if error:
        outcome_html = f'<span class="badge error">ERROR: {_escape(error)}</span>'
    elif is_correct:
        outcome_html = '<span class="badge ok">&#10003; Correct</span>'
    else:
        outcome_html = '<span class="badge fail">&#10007; Incorrect</span>'

    # steps
    steps_parts = [_step_html(s, i) for i, s in enumerate(history)]
    steps_html = "\n".join(steps_parts) if steps_parts else (
        '<div style="color:#888;text-align:center;padding:24px">'
        'No trajectory steps recorded</div>')

    # duration from first step
    duration = meta.get("duration", 0)

    # final answer block
    if error:
        outcome_label = "Failed"
        answer_text = _escape(error)
    else:
        outcome_label = "Correct" if is_correct else "Incorrect"
        answer_text = _escape(parsed_answer or "(no answer)")

    return _HTML_TEMPLATE.format(
        uid=_escape(uid),
        question=question,
        choices_html=choices_html,
        outcome_html=outcome_html,
        steps=steps_count,
        latency=latency,
        duration=duration,
        steps_html=steps_html,
        outcome_label=outcome_label,
        answer_text=answer_text,
    )


def save_trajectory(result: Dict[str, Any], messages: List[Dict[str, Any]],
                    trajectory_dir: str) -> str:
    """Save trajectory as OpenAI-format JSON + HTML visualization.

    Args:
        result: Evaluation result dict.
        messages: Full conversation messages list.
        trajectory_dir: Directory to save trajectory files.

    Returns:
        Path to the generated HTML file.
    """
    os.makedirs(trajectory_dir, exist_ok=True)
    uid = result.get("uid", "unknown")

    # 1) OpenAI messages format (strip system prompt's "media" paths to
    #    avoid leaking absolute filesystem paths)
    openai_msgs = _sanitize_messages(messages)
    json_path = os.path.join(trajectory_dir, f"{uid}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(openai_msgs, f, ensure_ascii=False, indent=2)

    # 2) HTML visualization
    html = generate_trajectory_html(result)
    html_path = os.path.join(trajectory_dir, f"{uid}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return html_path


def _sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Replace absolute file paths in media entries with basenames.

    Keeps the conversation structure intact for training.
    """
    cleaned = []
    for msg in messages:
        content = msg.get("content")
        role = msg.get("role", "user")
        if isinstance(content, list):
            new_content = []
            for block in content:
                if isinstance(block, dict):
                    b = dict(block)
                    for key in ("image", "audio", "video"):
                        if key in b and isinstance(b[key], str):
                            b[key] = os.path.basename(b[key])
                    new_content.append(b)
                else:
                    new_content.append(block)
            cleaned.append({"role": role, "content": new_content})
        else:
            cleaned.append({"role": role, "content": str(content)})
    return cleaned
