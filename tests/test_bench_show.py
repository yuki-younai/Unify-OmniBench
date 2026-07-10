"""单题推理 + 媒体检查：原生 transformers 跑一条 benchmark 题目，复制音视频到 example/。

Usage:
    python tests/test_bench_show.py                           # daily_omni 第1题
    python tests/test_bench_show.py --bench omnibench --index 5
    python tests/test_bench_show.py --bench worldsense --index 100
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

EXAMPLE_DIR = os.path.join(REPO_ROOT, "example")
os.makedirs(EXAMPLE_DIR, exist_ok=True)

from unify_omnibench.config import get_dataset_cfg
from unify_omnibench.core.registry import build_dataset
from unify_omnibench.prompt.media import media_description, filter_media
import unify_omnibench.datasets  # noqa

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info


def main():
    p = argparse.ArgumentParser(description="单题推理 + 媒体检查")
    p.add_argument("--bench", default="daily_omni",
                   choices=("daily_omni", "omnibench", "omnivideobench", "worldsense"))
    p.add_argument("--index", type=int, default=1)
    p.add_argument("--model-path", default="/apdcephfs_hldy/share_304318596/weiyangguo/models/Qwen2.5-Omni-7B")
    p.add_argument("--gpus", default="0")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # ── 加载数据 ──
    dataset_cfg = get_dataset_cfg(args.bench)
    dataset = build_dataset(dataset_cfg)
    samples = list(dataset)
    if args.index < 1 or args.index > len(samples):
        sys.exit(f"index out of range: 1..{len(samples)}")
    s = samples[args.index - 1]

    use_audio_in_video = bool(dataset_cfg.get("use_audio_in_video", False))
    video_cfg = dict(dataset_cfg.get("video") or {})
    video_kwargs = {k: v for k, v in video_cfg.items() if v is not None}
    system_prompt = dataset_cfg.get("system_prompt") or ""
    user_template = dataset_cfg.get("prompt_template") or ""

    # ── 显示题目 ──
    print(f"{'=' * 60}")
    print(f"[{args.index}/{len(samples)}] bench={args.bench}  uid={s.uid}")
    print(f"  task_type: {s.meta.get('task_type', '?')}")
    print(f"  use_audio_in_video: {use_audio_in_video}")
    print(f"  题目: {s.question}")
    if s.choices:
        print(f"  选项: {', '.join(s.choices)}")
    if s.answer:
        print(f"  答案: {s.answer}")

    # ── 复制媒体到 example/ ──
    for m in s.media:
        src = m.path
        if not os.path.exists(src):
            print(f"  [{m.kind}] MISSING: {src}")
            continue
        ext = os.path.splitext(src)[1]
        dst = os.path.join(EXAMPLE_DIR, f"{args.bench}_{args.index:04d}_{m.kind}{ext}")
        shutil.copy2(src, dst)
        size_mb = os.path.getsize(src) / 1024 / 1024
        print(f"  [{m.kind}] {os.path.basename(src)} ({size_mb:.1f} MB) → example/{os.path.basename(dst)}")

    # ── 构建 conversation（原生 transformers 方式） ──
    desc = media_description(s, dataset_cfg.get("modality", "av"))
    user_content = filter_media(s, dataset_cfg.get("modality", "av"), video_kwargs=video_kwargs)
    choices_text = "\n".join(str(c) for c in s.choices)
    prompt = user_template.format(media_desc=desc, question=s.question, choices=choices_text)
    user_content.append({"type": "text", "text": prompt})

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]

    # ── 加载模型（原生 transformers） ──
    print(f"\nLoading model from {args.model_path} ...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
        enable_audio_output=False,
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    print("Model loaded.\n")

    # ── 推理 ──
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=use_audio_in_video,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    print("Running inference ...")
    text_ids = model.generate(
        **inputs,
        use_audio_in_video=use_audio_in_video,
        return_audio=False,
        max_new_tokens=512,
        num_beams=1,
        do_sample=False,
    )
    result = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    result = (result[0] if result else "").strip()

    print(f"\n{'=' * 60}")
    print(f"  模型输出: {result}")
    print(f"  正确答案: {s.answer}")
    if s.answer and result:
        correct = result.strip().upper() == s.answer.strip().upper()
        print(f"  是否正确: {'✅' if correct else '❌'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
