"""Simulate the actual eval loop to reproduce the vLLM bug.

Key difference from the previous version:
  - Each iteration calls process_mm_info() fresh (like the real eval does)
  - Uses multiple different video files if available (to vary tensor shapes)
  - Reports first failure index to pinpoint when the engine starts breaking

Usage:
    # repeat same video N times (baseline — should always pass)
    N=20  MODEL=/path/to/model  python tests/test_qwen_omni_vllm.py

    # use real Daily-Omni videos to reproduce the bug
    N=20  VIDEO_DIR=/path/to/Daily-Omni/videos  MODEL=/path/to/model \\
        python tests/test_qwen_omni_vllm.py
"""
from __future__ import annotations

import os
import sys
import time
import traceback

# Must be set before any vLLM import
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_DISABLE_PROGRESS_BAR", "1")


def build_vllm_input(processor, process_mm_info, video_path, audio_path, prompt):
    """Build vLLM input — identical to vllm_runner._build_one."""
    conv = [
        {"role": "system", "content": [
            {"type": "text",
             "text": "You are Qwen, a virtual human developed by the Qwen Team, "
                      "Alibaba Group, capable of perceiving auditory and visual "
                      "inputs, as well as generating text and speech."},
        ]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path},
            *([{"type": "audio", "audio": audio_path}] if os.path.isfile(audio_path) else []),
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
    video_shape = tuple(videos[0].shape) if videos else None
    audio_shape = tuple(audios[0].shape) if audios else None
    mm_data = {}
    if videos:
        mm_data["video"] = videos
    if audios:
        mm_data["audio"] = audios
    if images:
        mm_data["image"] = images
    return {"prompt": text, "multi_modal_data": mm_data,
            "mm_processor_kwargs": {"use_audio_in_video": False},
            "_debug": {"video_shape": video_shape, "audio_shape": audio_shape}}


def main() -> int:
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.environ.get("MODEL", "/apdcephfs_hldy/share_304318596/weiyangguo/models/Qwen2.5-Omni-7B")
    N = int(os.environ.get("N", "100"))
    VIDEO_DIR = os.environ.get(
        "VIDEO_DIR",
        "/apdcephfs/private_weiyangguo/Agent-Tool/Datasets/Daily-Omni/Videos"
    )

    if not MODEL_PATH:
        sys.exit("Set MODEL=/path/to/Qwen2.5-Omni-7B")

    # collect video files to cycle through
    if VIDEO_DIR and os.path.isdir(VIDEO_DIR):
        # Use real Daily-Omni QA ordering (same video appears multiple times)
        qa_file = os.path.join(os.path.dirname(VIDEO_DIR), "qa.json")
        if os.path.isfile(qa_file):
            import json
            with open(qa_file) as f:
                qa_data = json.load(f)
            samples = []
            for item in qa_data[:N]:
                vid = str(item["video_id"])
                vp = os.path.join(VIDEO_DIR, vid, f"{vid}_video.mp4")
                ap = os.path.join(VIDEO_DIR, vid, f"{vid}_audio.wav")
                if os.path.isfile(vp):
                    samples.append((vid, vp, ap))
            print(f"[*] Using QA order: {len(samples)} samples "
                  f"({len({s[0] for s in samples})} unique videos) — "
                  f"duplicates will test tensor reuse bug")
        else:
            # fallback: unique videos alphabetically
            vid_ids = sorted(os.listdir(VIDEO_DIR))[:N]
            samples = []
            for vid in vid_ids:
                vp = os.path.join(VIDEO_DIR, vid, f"{vid}_video.mp4")
                ap = os.path.join(VIDEO_DIR, vid, f"{vid}_audio.wav")
                if os.path.isfile(vp):
                    samples.append((vid, vp, ap))
            print(f"[*] Using {len(samples)} unique videos from {VIDEO_DIR}")
    else:
        # fallback: repeat example/draw.mp4 N times
        vp = os.path.join(ROOT, "example", "draw.mp4")
        ap = os.path.join(ROOT, "example", "cough.wav")
        if not os.path.isfile(vp):
            sys.exit(f"Video not found: {vp}")
        samples = [(f"draw_{i}", vp, ap) for i in range(N)]
        print(f"[*] Using example/draw.mp4 repeated {N} times (same tensor shape each time)")

    print(f"[*] Will run {N} iterations\n")

    # load engine
    print(f"[*] Loading vLLM from: {MODEL_PATH}")
    import torch
    from transformers import Qwen2_5OmniProcessor
    from vllm import LLM, SamplingParams
    from qwen_omni_utils import process_mm_info

    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.95,
        max_num_seqs=1,
        max_model_len=32768,
        dtype="bfloat16",
        seed=1234,
        limit_mm_per_prompt={"image": 1, "video": 1, "audio": 1},
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    sp = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=10)
    print("[*] vLLM engine ready\n")

    prompt = "What is happening? Reply with one letter A/B/C/D."
    ok = fail = 0
    first_fail = None
    t0 = time.time()

    for i, (vid, vp, ap) in enumerate(samples[:N]):
        # ── simulate real eval: build input fresh each time ──────────────
        try:
            vllm_in = build_vllm_input(processor, process_mm_info, vp, ap, prompt)
            debug = vllm_in.pop("_debug", {})
        except Exception as e:
            fail += 1
            first_fail = first_fail or i
            print(f"[{i+1}/{N}] FAIL (build_input) {vid}: {type(e).__name__}: {e}")
            continue

        try:
            outs = llm.generate([vllm_in], sampling_params=sp)
            raw = outs[0].outputs[0].text if outs and outs[0].outputs else ""
            ok += 1
            print(f"[{i+1}/{N}] OK   {vid}  shape={debug}  out={repr(raw)}")
        except Exception as e:
            fail += 1
            first_fail = first_fail or i
            print(f"[{i+1}/{N}] FAIL (llm.generate) {vid}  shape={debug}")
            print(f"       {type(e).__name__}: {e}")
            traceback.print_exc(limit=6)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"ok={ok}  fail={fail}  total={N}  elapsed={elapsed:.0f}s")
    if first_fail is not None:
        print(f"First failure at index {first_fail}")
    if ok:
        print(f"avg {elapsed/ok:.1f}s per successful call")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
