"""Regression test for the transformers backend: runs ALL scenarios every
time, no mode/variant switches to remember.

Every run automatically covers, in one go:
  1. 独立音频 (use_audio_in_video=False) + 单条调用 (model.generate() 逐条，
     镜像 qwen25omni.py::Qwen25OmniModel.generate())
  2. 交织音频 (use_audio_in_video=True)  + 单条调用
  3. 独立音频 (use_audio_in_video=False) + batch调用 (N 个样本一次性 padding
     进同一个 model.generate() 调用，镜像
     qwen25omni.py::Qwen25OmniModel.generate_batch())
  4. 交织音频 (use_audio_in_video=True)  + batch调用

use_audio_in_video wiring:
  * False（独立音频）: conversation 同时保留 video 内容块和一个独立的 audio
    内容块；``process_mm_info`` 应返回非空 ``audios`` 列表（来自独立 .wav 文件）。
  * True（交织音频）: conversation 里独立的 audio 内容块被丢弃（音频改从视频
    自己的音轨里提取，镜像 openai_chat.py 的客户端逻辑）。``process_mm_info``
    （具体是 ``audio_process.py::process_audio_info``）此时仍会返回非空
    ``audios``——它把视频自带音轨也 append 进了这个列表（不是被隐藏/内部消费），
    真正的交织发生在后面 ``processor.__call__()``/
    ``replace_multimodal_special_tokens()`` 里，按 ``<video>``/``<audio>``
    占位符在模板文本里出现的顺序，从这同一个 ``audios`` 列表里按位置消费。
    所以两种模式下 ``audios`` 都应该非空——区别是音频来自哪里（独立 .wav vs
    视频自己的音轨），不是 audios 是否被填充。

Requirements:
    pip install transformers qwen-omni-utils torch
    (for video decoding: pip install av)

Usage:
    python tests/test_qwen_omni_transformer.py

    # 换模型/视频/音频/batch样本数（这些是基本输入，不是开关）
    MODEL=/path/to/Qwen2.5-Omni-7B  N=3  VIDEO=/path/to/v.mp4  AUDIO=/path/to/a.wav \\
        python tests/test_qwen_omni_transformer.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.environ.get("MODEL", "/apdcephfs_hldy/share_304318596/weiyangguo/models/Qwen2.5-Omni-3B")
N = int(os.environ.get("N", "3"))  # samples per batch-mode scenario

_VID = os.path.join(ROOT, "example", "draw.mp4")
_AUD = os.path.join(ROOT, "example", "cough.wav")
VIDEO_PATH = os.environ.get("VIDEO", _VID)
AUDIO_PATH = os.environ.get("AUDIO", _AUD)

if not MODEL_PATH:
    sys.exit("Set MODEL=/path/to/Qwen2.5-Omni-7B")

has_video = os.path.isfile(VIDEO_PATH) if VIDEO_PATH else False
has_audio = os.path.isfile(AUDIO_PATH) if AUDIO_PATH else False

if not (has_video or has_audio):
    print("[warn] no video/audio found — scenarios will run but the "
          "use_audio_in_video wiring checks below are meaningless without media\n")


# ---------------------------------------------------------------------------
# conversation construction
# ---------------------------------------------------------------------------
QUESTION = "First, describe what you see and hear in this video and audio. "


def build_conversation(use_audio_in_video: bool):
    """Mirror unify_omnibench's wiring semantics: when
    use_audio_in_video=True, the separate audio content block is DROPPED
    (audio is expected to come from the video's own track instead) —
    same rule openai_chat.py applies client-side. When False, both video
    and a separate audio block are kept (matches the production
    Qwen25OmniModel path when use_audio_in_video=False)."""
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
# load model & processor (once, shared across all scenarios)
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
    # Without this, the full Talker+Token2Wav TTS pipeline gets loaded and
    # generate() below will ALSO run autoregressive speech-token generation +
    # DiT vocoder synthesis after the text tokens — far slower than pure text
    # generation. We only need text output for eval, so mirror qwen25omni.py's
    # production wiring and skip TTS entirely.
    enable_audio_output=False,
)
processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
print("[*] Model loaded\n")

from qwen_omni_utils import process_mm_info


# ---------------------------------------------------------------------------
# wiring sanity-check (shared by both single-call and batch-call runners)
# ---------------------------------------------------------------------------
def _check_wiring(use_audio_in_video: bool, block_types: list, n_audios: int) -> bool:
    if not (has_audio and has_video):
        return True
    if use_audio_in_video:
        if "audio" in block_types:
            print("    \u274c UNEXPECTED: separate 'audio' block was NOT dropped "
                  "from the conversation despite use_audio_in_video=True")
            return False
        if n_audios == 0:
            print("    \u26a0\ufe0f  n_audios=0 — video has no audio track to extract")
            return True
        print("    \u2705 separate audio block correctly dropped from the "
              f"conversation; process_mm_info still returned audios={n_audios} "
              "(the video's own track, to be interleaved inside processor.__call__())")
        return True
    else:
        if n_audios == 0:
            print("    \u274c UNEXPECTED: audios is empty — the separate .wav "
                  "file should have been processed independently")
            return False
        print("    \u2705 separate audio block correctly present/processed")
        return True


# ---------------------------------------------------------------------------
# scenario 1/2 — single-request generate(), mirrors
# qwen25omni.py::Qwen25OmniModel.generate()
# ---------------------------------------------------------------------------
def run_single(use_audio_in_video: bool) -> dict:
    conversation = build_conversation(use_audio_in_video)
    block_types = [c["type"] for c in conversation[1]["content"]]
    print(f"    conversation content blocks: {block_types}")

    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    n_audios = 0 if not audios else len(audios)
    n_videos = 0 if not videos else len(videos)
    print(f"    process_mm_info -> audios={n_audios} images={0 if not images else len(images)} videos={n_videos}")
    ok = _check_wiring(use_audio_in_video, block_types, n_audios)

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=use_audio_in_video,
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
                return_audio=False,
            )
    except Exception as e:
        print(f"    \u274c FAIL: generate() raised {type(e).__name__}: {str(e)[:300]}")
        return {"ok": False, "answers": [None], "error": str(e)}

    out_ids = out[0] if isinstance(out, tuple) else out
    in_len = inputs["input_ids"].shape[1]
    gen_ids = out_ids[:, in_len:] if out_ids.shape[1] > in_len else out_ids
    answer = processor.batch_decode(
        gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    print(f"    --- answer ---\n    {answer}")
    return {"ok": ok, "answers": [answer], "error": None}


# ---------------------------------------------------------------------------
# scenario 3/4 — real batched generate() (N samples padded into ONE call),
# mirrors qwen25omni.py::Qwen25OmniModel.generate_batch()
# ---------------------------------------------------------------------------
def run_batch(use_audio_in_video: bool, n: int) -> dict:
    conversation = build_conversation(use_audio_in_video)
    block_types = [c["type"] for c in conversation[1]["content"]]
    print(f"    conversation content blocks: {block_types}  (x{n}, batched into 1 call)")

    texts, all_audios, all_images, all_videos, n_audios0 = [], [], [], [], 0
    for i in range(n):
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
        if i == 0:
            n_audios0 = 0 if not audios else len(audios)
        texts.append(processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False))
        all_audios.append(audios[0] if audios else None)
        all_images.append(images[0] if images else None)
        all_videos.append(videos[0] if videos else None)

    ok = _check_wiring(use_audio_in_video, block_types, n_audios0)

    audios = [x for x in all_audios if x is not None] or None
    images = [x for x in all_images if x is not None] or None
    videos = [x for x in all_videos if x is not None] or None

    inputs = processor(
        text=texts, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=use_audio_in_video,
    )
    for key, value in list(inputs.items()):
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                inputs[key] = value.to(device=model.device, dtype=model.dtype)
            else:
                inputs[key] = value.to(device=model.device)

    print(f"    [*] Generating ({n} samples in one batched call)...")
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                use_audio_in_video=use_audio_in_video,
                return_audio=False,
                max_new_tokens=32,
                num_beams=1,
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
    except Exception as e:
        print(f"    \u274c FAIL: generate() raised {type(e).__name__}: {str(e)[:300]}")
        return {"ok": False, "answers": [None] * n, "error": str(e)}

    gen_ids = out[0] if isinstance(out, tuple) else out
    in_len = inputs["input_ids"].shape[1]
    gen_ids = gen_ids[:, in_len:]
    answers = [a.strip() for a in processor.batch_decode(
        gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )]
    for i, a in enumerate(answers):
        print(f"    [{i+1}/{n}] {a}")
    return {"ok": ok, "answers": answers, "error": None}


# ---------------------------------------------------------------------------
# run all 4 scenarios, then summarize
# ---------------------------------------------------------------------------
scenarios = [
    ("独立音频 + 单条调用", False, "single"),
    ("交织音频 + 单条调用", True, "single"),
    ("独立音频 + batch调用", False, "batch"),
    ("交织音频 + batch调用", True, "batch"),
]

results = []
for name, use_audio_in_video, mode in scenarios:
    print(f"\n{'='*60}\n[*] Scenario: {name}\n{'='*60}")
    if mode == "single":
        r = run_single(use_audio_in_video)
    else:
        r = run_batch(use_audio_in_video, N)
    passed = r["ok"] and r["error"] is None
    results.append((name, passed, r["answers"]))

print(f"\n{'='*60}\n[*] Summary\n{'='*60}")
all_ok = True
for name, passed, answers in results:
    all_ok = all_ok and passed
    status = "\u2705 PASS" if passed else "\u274c FAIL"
    preview = answers[0] if answers else None
    if preview and len(preview) > 60:
        preview = preview[:60] + "..."
    print(f"  {status}  {name:<18}  answer={preview!r}")

if not all_ok:
    sys.exit(1)
