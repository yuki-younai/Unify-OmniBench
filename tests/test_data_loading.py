"""Test whether failed video/audio files can be loaded by process_mm_info.

No GPU needed — only tests data loading (process_mm_info is CPU-side).

Usage:
    python tests/test_data_loading.py

    # test specific video ids
    VIDS=G_VTkkb34gw,Me4W36_lUcI  python tests/test_data_loading.py
"""
from __future__ import annotations

import json
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = os.environ.get(
    "VIDEO_DIR",
    "/apdcephfs/private_weiyangguo/Agent-Tool/Datasets/Daily-Omni/Videos"
)
MODEL_PATH = os.environ.get(
    "MODEL",
    "/apdcephfs_hldy/share_304318596/weiyangguo/models/Qwen2.5-Omni-7B"
)

# ── pick video ids to test ────────────────────────────────────────────────
env_vids = os.environ.get("VIDS", "")
if env_vids:
    test_vids = env_vids.split(",")
else:
    # read from failed.jsonl automatically
    failed_path = os.path.join(
        ROOT, "results", "daily_omni", "Qwen2.5-Omni-7B", "failed.jsonl"
    )
    if os.path.exists(failed_path):
        with open(failed_path) as f:
            failed = [json.loads(l) for l in f if l.strip()]
        # uid format: daily_omni:<idx>:<video_id>
        seen = set()
        test_vids = []
        for r in failed[:20]:
            vid = r.get("uid", "").split(":")[-1]
            if vid and vid not in seen:
                seen.add(vid)
                test_vids.append(vid)
        print(f"[*] Found {len(failed)} failures, testing first {len(test_vids)} unique video ids\n")
    else:
        # fallback: use example file
        test_vids = []
        print("[warn] No failed.jsonl, testing example/draw.mp4 only\n")

print(f"[*] Loading processor from {MODEL_PATH} ...")
from transformers import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
print("[*] Processor loaded\n")


def test_one(vid: str) -> bool:
    vpath = os.path.join(VIDEO_DIR, vid, f"{vid}_video.mp4")
    apath = os.path.join(VIDEO_DIR, vid, f"{vid}_audio.wav")

    print(f"[{vid}]")
    print(f"  video: {'EXISTS' if os.path.isfile(vpath) else 'MISSING':7s} {vpath}")
    print(f"  audio: {'EXISTS' if os.path.isfile(apath) else 'MISSING':7s} {apath}")

    if not os.path.isfile(vpath):
        print(f"  => SKIP (video missing)")
        return False

    conv = [
        {"role": "system", "content": [{"type": "text", "text": "You are Qwen."}]},
        {"role": "user", "content": [
            {"type": "video", "video": vpath},
            *([{"type": "audio", "audio": apath}] if os.path.isfile(apath) else []),
            {"type": "text", "text": "What is happening?"},
        ]},
    ]

    # Step 1: process_mm_info
    try:
        audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
        print(f"  process_mm_info => "
              f"audios={type(audios).__name__}({len(audios) if audios is not None else 'None'}) "
              f"images={type(images).__name__}({len(images) if images is not None else 'None'}) "
              f"videos={type(videos).__name__}({len(videos) if videos is not None else 'None'})")
        if videos is None or len(videos) == 0:
            print(f"  => WARN: videos is {videos!r}  ← this would cause NoneType error in vLLM")
    except Exception as e:
        print(f"  process_mm_info FAIL: {type(e).__name__}: {e}")
        traceback.print_exc(limit=4)
        return False

    # Step 2: apply_chat_template
    try:
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        print(f"  apply_chat_template => text_len={len(text)}")
    except Exception as e:
        print(f"  apply_chat_template FAIL: {type(e).__name__}: {e}")
        traceback.print_exc(limit=4)
        return False

    # Step 3: processor() tokenize
    try:
        inputs = processor(
            text=text,
            audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=False,
        )
        shapes = {k: tuple(v.shape) for k, v in inputs.items() if hasattr(v, "shape")}
        print(f"  processor() => {shapes}")
    except Exception as e:
        print(f"  processor() FAIL: {type(e).__name__}: {e}")
        traceback.print_exc(limit=4)
        return False

    print(f"  => ALL OK")
    return True


# ── run ──────────────────────────────────────────────────────────────────
if test_vids:
    ok = sum(test_one(vid) for vid in test_vids)
    print(f"\n{'='*50}")
    print(f"ok={ok} / {len(test_vids)}")
else:
    # example fallback
    ex_vid = os.path.join(ROOT, "example", "draw.mp4")
    ex_aud = os.path.join(ROOT, "example", "cough.wav")
    conv = [
        {"role": "system", "content": [{"type": "text", "text": "You are Qwen."}]},
        {"role": "user", "content": [
            {"type": "video", "video": ex_vid},
            {"type": "audio", "audio": ex_aud},
            {"type": "text", "text": "What is happening?"},
        ]},
    ]
    audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
    print(f"example => audios={type(audios).__name__} videos={type(videos).__name__}")
    print("OK")
