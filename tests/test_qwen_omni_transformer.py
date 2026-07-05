"""Minimal example: run Qwen2.5-Omni with 🤗 Transformers on a multi-modal input.

Example media files from the official Qwen2.5-Omni repo are stored in
``Unify-OmniBench/example/``.

Usage:

    # with media (default: example/draw.mp4 + example/cough.wav)
    MODEL=/path/to/Qwen2.5-Omni-7B  python tests/test_qwen_omni_basic.py

    # text-only
    MODEL=/path/to/Qwen2.5-Omni-7B  python tests/test_qwen_omni_basic.py --text

    # think-before-answer (two-turn: describe → answer)
    MODEL=/path/to/Qwen2.5-Omni-7B  python tests/test_qwen_omni_basic.py --cot

    # custom media
    VIDEO=/path/to/v.mp4 AUDIO=/path/to/a.wav  python tests/test_qwen_omni_basic.py

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

if any(a in ("--text", "--text-only") for a in sys.argv):
    VIDEO_PATH, AUDIO_PATH = "", ""
else:
    VIDEO_PATH = os.environ.get("VIDEO", _VID)
    AUDIO_PATH = os.environ.get("AUDIO", _AUD)

if not MODEL_PATH:
    sys.exit("Set MODEL=/path/to/Qwen2.5-Omni-7B")


# ---------------------------------------------------------------------------
# Step 2 — build a conversation
# ---------------------------------------------------------------------------
QUESTION = (
    "First, describe what you see and hear in this video and audio. "
)

conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text",
             "text": "You are Qwen2.5-Omni. Think step by step, then give your final answer."}
        ]
    },
    {
        "role": "user",
        "content": []   # filled below
    }
]

has_video = os.path.isfile(VIDEO_PATH) if VIDEO_PATH else False
has_audio = os.path.isfile(AUDIO_PATH) if AUDIO_PATH else False

if has_video:
    conversation[1]["content"].append({"type": "video", "video": VIDEO_PATH})
if has_audio:
    conversation[1]["content"].append({"type": "audio", "audio": AUDIO_PATH})
conversation[1]["content"].append({"type": "text", "text": QUESTION})

if not (has_video or has_audio):
    print("[warn] no video/audio provided — running text-only\n")


# ---------------------------------------------------------------------------
# Step 3 — load model & processor
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


# ---------------------------------------------------------------------------
# Step 4 — process multi-modal inputs
# ---------------------------------------------------------------------------
from qwen_omni_utils import process_mm_info

use_audio_in_video = has_audio and has_video
audios, images, videos = process_mm_info(
    conversation, use_audio_in_video=use_audio_in_video
)

text = processor.apply_chat_template(
    conversation, add_generation_prompt=True, tokenize=False
)
print("--- prompt (first 300 chars) ---")
print(text[:300], "...\n")


# ---------------------------------------------------------------------------
# Step 5 — tokenize & generate
# ---------------------------------------------------------------------------
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

print("[*] Generating...")
with torch.no_grad():
    out = model.generate(
        **inputs,
        use_audio_in_video=use_audio_in_video,
        max_new_tokens=32,
        do_sample=False,
        temperature=None,          # greedy
    )

# Qwen2.5-Omni returns (ids, audio_tokens) — we only need text ids
if isinstance(out, tuple):
    out_ids = out[0]
else:
    out_ids = out

in_len = inputs["input_ids"].shape[1]
gen_ids = out_ids[:, in_len:] if out_ids.shape[1] > in_len else out_ids
answer = processor.batch_decode(
    gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0].strip()

# ---------------------------------------------------------------------------
# Step 6 — result
# ---------------------------------------------------------------------------
print("--- answer ---")
print(answer)
