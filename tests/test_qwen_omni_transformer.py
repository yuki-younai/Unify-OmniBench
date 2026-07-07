"""Minimal example: run Qwen2.5-Omni with 🤗 Transformers on a multi-modal input.

Example media files from the official Qwen2.5-Omni repo are stored in
``Unify-OmniBench/example/``.

Usage:

    # with media (default: example/draw.mp4 + example/cough.wav)
    MODEL=/path/to/Qwen2.5-Omni-7B  python tests/test_qwen_omni_transformer.py

    # text-only
    MODEL=/path/to/Qwen2.5-Omni-7B  python tests/test_qwen_omni_transformer.py --text

    # custom media
    VIDEO=/path/to/v.mp4 AUDIO=/path/to/a.wav  python tests/test_qwen_omni_transformer.py

    # only run one of the two use_audio_in_video variants (default: both, when
    # video+audio are both present)
    ... --use-audio-in-video=false
    ... --use-audio-in-video=true

Coverage:
    use_audio_in_video wiring (client-side, this process IS the "client" for
    the transformer backend — no separate server) — when both video AND audio
    are provided, runs generation TWICE and compares:
      * use_audio_in_video=False (matches the production
        ``unify_omnibench/models/local/qwen25omni.py::Qwen25OmniModel``,
        which hardcodes False everywhere — see NOTE below): conversation
        keeps BOTH the video content block and a SEPARATE audio content
        block; ``process_mm_info`` must return a non-empty ``audios`` list.
      * use_audio_in_video=True: conversation drops the separate audio
        content block (mirrors ``openai_chat.py``'s wiring — audio is
        expected to come from the video's own track instead), so
        ``process_mm_info`` should return an EMPTY/None ``audios`` list
        (the video's own audio, if any, is interleaved into ``videos``
        instead and consumed internally by the processor/model, not
        exposed as a separate ``audios`` tensor).

    NOTE: ``Qwen25OmniModel`` (the actual production ``transformers_qwen25omni``
    backend used by run.py/eval.sh) hardcodes ``use_audio_in_video=False``
    everywhere and does NOT read this value from cfg — the True path tested
    here only exercises the raw ``qwen_omni_utils.process_mm_info`` /
    ``model.generate()`` API directly, NOT the production model class. If the
    production class needs a configurable ``use_audio_in_video``, it must be
    wired up separately (not done here, by explicit request).

Requirements:
    pip install transformers qwen-omni-utils torch
    (for video decoding: pip install av)
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Step 1 — model path & media
# ---------------------------------------------------------------------------
MODEL_PATH = "/apdcephfs_hldy/share_304318596/weiyangguo/models/Qwen2.5-Omni-3B"

_VID = os.path.join(ROOT, "example", "draw.mp4")
_AUD = os.path.join(ROOT, "example", "cough.wav")

_argv = sys.argv[1:]


def _flag_value(name: str) -> "str | None":
    """Parse ``--name=value`` from argv (case-insensitive value)."""
    prefix = f"--{name}="
    for a in _argv:
        if a.startswith(prefix):
            return a[len(prefix):].strip().lower()
    return None


if any(a in ("--text", "--text-only") for a in _argv):
    VIDEO_PATH, AUDIO_PATH = "", ""
else:
    VIDEO_PATH = os.environ.get("VIDEO", _VID)
    AUDIO_PATH = os.environ.get("AUDIO", _AUD)

if not MODEL_PATH:
    sys.exit("Set MODEL=/path/to/Qwen2.5-Omni-7B")

has_video = os.path.isfile(VIDEO_PATH) if VIDEO_PATH else False
has_audio = os.path.isfile(AUDIO_PATH) if AUDIO_PATH else False

# Which use_audio_in_video variant(s) to run. Default: BOTH when video+audio
# are both present (that's the only case where the two variants can differ
# at all); otherwise just run once with the natural default (False).
_uav_flag = _flag_value("use-audio-in-video")
if _uav_flag in ("true", "false"):
    RUN_VARIANTS = [_uav_flag == "true"]
elif has_video and has_audio:
    RUN_VARIANTS = [False, True]
else:
    RUN_VARIANTS = [False]

if not (has_video or has_audio):
    print("[warn] no video/audio provided — running text-only\n")


# ---------------------------------------------------------------------------
# Step 2 — build a conversation (per use_audio_in_video variant)
# ---------------------------------------------------------------------------
QUESTION = (
    "First, describe what you see and hear in this video and audio. "
)


def build_conversation(use_audio_in_video: bool):
    """Mirror unify_omnibench's wiring semantics: when
    use_audio_in_video=True, the separate audio content block is DROPPED
    (audio is expected to come from the video's own track instead) —
    same rule openai_chat.py applies client-side. When False, both video
    and a separate audio block are kept (matches the production
    Qwen25OmniModel path, which always uses this shape)."""
    conv = [
        {
            "role": "system",
            "content": [
                {"type": "text",
                 "text": "You are Qwen2.5-Omni. Think step by step, then give your final answer."}
            ]
        },
        {"role": "user", "content": []},
    ]
    if has_video:
        conv[1]["content"].append({"type": "video", "video": VIDEO_PATH})
    if has_audio and not (use_audio_in_video and has_video):
        conv[1]["content"].append({"type": "audio", "audio": AUDIO_PATH})
    conv[1]["content"].append({"type": "text", "text": QUESTION})
    return conv


# ---------------------------------------------------------------------------
# Step 3 — load model & processor (once, shared across variants)
# ---------------------------------------------------------------------------
print(f"[*] Loading model from: {MODEL_PATH}")
import torch
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
print("[*] Model loaded\n")

from qwen_omni_utils import process_mm_info


# ---------------------------------------------------------------------------
# Step 4 — run one variant end-to-end (process -> generate -> decode)
# ---------------------------------------------------------------------------
def run_variant(use_audio_in_video: bool) -> dict:
    """Run the full pipeline for one use_audio_in_video value.

    Returns a dict with wiring diagnostics + the decoded answer, so the
    caller can compare both variants side by side.
    """
    tag = f"use_audio_in_video={use_audio_in_video}"
    print(f"\n{'=' * 60}\n[*] Variant: {tag}\n{'=' * 60}")

    conversation = build_conversation(use_audio_in_video)
    block_types = [c["type"] for c in conversation[1]["content"]]
    print(f"    conversation content blocks: {block_types}")

    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=use_audio_in_video
    )
    n_audios = 0 if not audios else len(audios)
    n_videos = 0 if not videos else len(videos)
    print(f"    process_mm_info -> audios={n_audios} images={0 if not images else len(images)} videos={n_videos}")

    # Wiring check (best-effort, informational — this script is a manual
    # smoke test, not a strict pass/fail CI gate like test_qwen_omni_openai.py):
    ok = True
    if has_audio and has_video:
        if use_audio_in_video:
            if n_audios != 0:
                print("    \u274c UNEXPECTED: separate 'audio' block was kept in the "
                      "conversation but use_audio_in_video=True should drop it "
                      "(audio expected to come from the video track instead)")
                ok = False
            else:
                print("    \u2705 separate audio block correctly dropped "
                      "(video's own track — if any — is used instead)")
        else:
            if n_audios == 0:
                print("    \u274c UNEXPECTED: audios is empty — the separate .wav "
                      "file should have been processed independently")
                ok = False
            else:
                print("    \u2705 separate audio block correctly present/processed")

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    print("    --- prompt (first 200 chars) ---")
    print("    " + text[:200].replace("\n", " ") + " ...")

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    print("    [*] Generating...")
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                use_audio_in_video=use_audio_in_video,
                max_new_tokens=32,
                do_sample=False,
                temperature=None,  # greedy
            )
    except Exception as e:
        print(f"    \u274c FAIL: generate() raised {type(e).__name__}: {str(e)[:300]}")
        return {"tag": tag, "ok": False, "answer": None, "error": str(e)}

    # Qwen2.5-Omni returns (ids, audio_tokens) — we only need text ids
    out_ids = out[0] if isinstance(out, tuple) else out
    in_len = inputs["input_ids"].shape[1]
    gen_ids = out_ids[:, in_len:] if out_ids.shape[1] > in_len else out_ids
    answer = processor.batch_decode(
        gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    print("    --- answer ---")
    print("    " + answer)
    return {"tag": tag, "ok": ok, "answer": answer, "error": None}


# ---------------------------------------------------------------------------
# Step 5 — run all requested variants, then summarize
# ---------------------------------------------------------------------------
results = [run_variant(v) for v in RUN_VARIANTS]

if len(results) > 1:
    print(f"\n{'=' * 60}\n[*] Comparison summary\n{'=' * 60}")
    for r in results:
        status = "✅ 正常" if r["ok"] and r["error"] is None else "❌ 有问题"
        print(f"  {r['tag']:28s} {status}  answer={r['answer']!r}")
    if results[0]["answer"] is not None and results[1]["answer"] is not None:
        same = results[0]["answer"].strip().lower() == results[1]["answer"].strip().lower()
        print(f"  identical answers across variants: {same} "
              f"(不要求一定相同——只是供人工核对两条路径是否产生合理且不冲突的结果)")

if any(not r["ok"] or r["error"] for r in results):
    sys.exit(1)
