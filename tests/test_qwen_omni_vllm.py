"""Regression test for the vLLM backend: runs ALL scenarios every time,
no mode switches to remember.

Every run automatically covers, in one go:
  1. 独立音频 (use_audio_in_video=False) + 单条调用 (sequential generate())
  2. 交织音频 (use_audio_in_video=True)  + 单条调用 (sequential generate())
  3. 独立音频 (use_audio_in_video=False) + batch调用 (single multi-prompt generate())
  4. 交织音频 (use_audio_in_video=True)  + batch调用 (single multi-prompt generate())

Background on why both axes matter:

- **独立音频 vs 交织音频** (``use_audio_in_video``): 独立音频是 video/audio 作为
  两个独立的 multi_modal_data 条目发送（Daily-Omni/OmniBench 的模式）；交织音频
  是音频从视频容器自身音轨按时间戳提取、跟视频帧交替排列（OmniVideoBench/
  WorldSense 的模式）。交织模式曾经在 vLLM V1 引擎上必现崩溃：
      RuntimeError: Worker failed with error 'index 1 is out of bounds for
      dimension 0 with size 1'
  (tracked upstream: https://github.com/vllm-project/vllm/issues/25473,
  fixed for Qwen2.5-Omni by PR #33605, merged 2026-02-04). 已确认随 vllm
  升级修复，这里保留作回归测试：每次升级 vllm 后重跑一次确认没有回归。

- **单条调用 vs batch调用**：batch调用是 vllm_runner.py::generate_batch() 的
  真实实现——一次性把整批 prompt 提交给 ``llm.generate([...])``，让 vLLM 内部
  做 continuous batching 调度（之前的实现是 Python 层 for 循环单条调用，跟
  sequential 完全等效、零吞吐提升，已废弃）。这里验证两件事：
    1. ``llm.generate()`` 返回顺序跟输入顺序一致（每条结果打印时标注所属视频
       id，方便肉眼核对没有错位）
    2. len(outputs) == len(inputs) 始终成立（防御性检查）

Usage:
    python tests/test_qwen_omni_vllm.py

    # 换模型/视频/样本数（这些是基本输入，不是开关）
    MODEL=/path/to/model  N=5  VIDEO_DIR=/path/to/Daily-Omni/videos \\
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
# Ensure spawn-mode vLLM worker subprocesses (a fresh interpreter each,
# re-importing transformers from scratch) also pick up sitecustomize.py's
# patch — they inherit env vars (incl. PYTHONPATH) but NOT in-memory
# monkeypatches from this process.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONPATH"] = _repo_root + os.pathsep + os.environ.get("PYTHONPATH", "")

# transformers/vllm version-compat shim (see Unify-OmniBench/sitecustomize.py
# for the full explanation) — this script is often run standalone (bypassing
# eval.sh's PYTHONPATH export that would auto-load sitecustomize.py), so the
# same patch is duplicated here to avoid re-hitting:
#   AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended
try:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(
            lambda self: self.all_special_tokens
        )
except Exception:
    pass

# Fixed batch size for the batch-mode scenarios below. Not exposed as an env
# var on purpose — the user just wants "run everything", not another knob.
BATCH_SIZE = 4


def build_vllm_input(processor, process_mm_info, video_path, audio_path, prompt,
                      use_audio_in_video: bool):
    """Build vLLM input — identical to vllm_runner._build_one.

    use_audio_in_video=False: video + independent audio file, both attached
    as separate content blocks (Daily-Omni/OmniBench's "mixed_modalities"
    pattern).
    use_audio_in_video=True: only a video content block is attached (NO
    separate audio block) — the video's own audio track is extracted and
    interleaved with video tokens (OmniVideoBench/WorldSense's pattern).
    """
    content = [{"type": "video", "video": video_path}]
    if not use_audio_in_video and os.path.isfile(audio_path):
        content.append({"type": "audio", "audio": audio_path})
    content.append({"type": "text", "text": prompt})
    conv = [
        {"role": "system", "content": [
            {"type": "text",
             "text": "You are Qwen, a virtual human developed by the Qwen Team, "
                      "Alibaba Group, capable of perceiving auditory and visual "
                      "inputs, as well as generating text and speech."},
        ]},
        {"role": "user", "content": content},
    ]
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conv, use_audio_in_video=use_audio_in_video)
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
            "mm_processor_kwargs": {"use_audio_in_video": use_audio_in_video},
            "_debug": {"video_shape": video_shape, "audio_shape": audio_shape}}


def _run_sequential(llm, processor, process_mm_info, samples, prompt, sp, use_audio_in_video):
    """One llm.generate([single_prompt]) call per sample."""
    N = len(samples)
    ok = fail = 0
    first_fail = None

    for i, (vid, vp, ap) in enumerate(samples):
        try:
            vllm_in = build_vllm_input(processor, process_mm_info, vp, ap, prompt,
                                        use_audio_in_video)
            debug = vllm_in.pop("_debug", {})
        except Exception as e:
            fail += 1
            first_fail = first_fail if first_fail is not None else i
            print(f"[{i+1}/{N}] FAIL (build_input) {vid}: {type(e).__name__}: {e}")
            continue

        try:
            outs = llm.generate([vllm_in], sampling_params=sp)
            raw = outs[0].outputs[0].text if outs and outs[0].outputs else ""
            ok += 1
            print(f"[{i+1}/{N}] OK   {vid}  shape={debug}  out={repr(raw)}")
        except Exception as e:
            fail += 1
            first_fail = first_fail if first_fail is not None else i
            print(f"[{i+1}/{N}] FAIL (llm.generate) {vid}  shape={debug}")
            print(f"       {type(e).__name__}: {e}")
            traceback.print_exc(limit=6)

    return ok, fail, first_fail


def _run_batch(llm, processor, process_mm_info, samples, prompt, sp, use_audio_in_video,
               batch_size):
    """Real multi-prompt vLLM batching — mirrors vllm_runner.py::generate_batch():
    build ALL prompts in a chunk up front, submit them to ``llm.generate([...])``
    in ONE call, then verify output order/count match input order/count.
    """
    N = len(samples)
    ok = fail = 0
    first_fail = None

    for chunk_start in range(0, N, batch_size):
        chunk = samples[chunk_start:chunk_start + batch_size]
        vllm_ins, debugs, valid_idx = [], [], []
        for j, (vid, vp, ap) in enumerate(chunk):
            i = chunk_start + j
            try:
                vllm_in = build_vllm_input(processor, process_mm_info, vp, ap, prompt,
                                            use_audio_in_video)
                debugs.append(vllm_in.pop("_debug", {}))
                vllm_ins.append(vllm_in)
                valid_idx.append(i)
            except Exception as e:
                fail += 1
                first_fail = first_fail if first_fail is not None else i
                print(f"[{i+1}/{N}] FAIL (build_input) {vid}: {type(e).__name__}: {e}")

        if not vllm_ins:
            continue

        try:
            # ── the actual real-batch call: ALL prompts in this chunk go
            # through vLLM's engine in a SINGLE .generate() invocation ──
            outs = llm.generate(vllm_ins, sampling_params=sp)
            if len(outs) != len(vllm_ins):
                raise RuntimeError(
                    f"vLLM returned {len(outs)} outputs for {len(vllm_ins)} inputs "
                    f"in chunk starting at {chunk_start} — ORDER/ALIGNMENT BROKEN"
                )
            for k, out in enumerate(outs):
                i = valid_idx[k]
                vid = chunk[i - chunk_start][0]
                raw = out.outputs[0].text if out and out.outputs else ""
                ok += 1
                print(f"[{i+1}/{N}] OK   {vid}  shape={debugs[k]}  out={repr(raw)}  "
                      f"(batch pos {k+1}/{len(outs)})")
        except Exception as e:
            fail += len(vllm_ins)
            first_fail = first_fail if first_fail is not None else valid_idx[0]
            print(f"[chunk {chunk_start}:{chunk_start+len(chunk)}] "
                  f"FAIL (llm.generate, batch of {len(vllm_ins)}): {type(e).__name__}: {e}")
            traceback.print_exc(limit=6)

    return ok, fail, first_fail


def main() -> int:
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.environ.get("MODEL", "/apdcephfs_hldy/share_304318596/weiyangguo/models/Qwen2.5-Omni-7B")
    N = int(os.environ.get("N", "5"))
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

    print(f"[*] Will run {N} samples per scenario, 4 scenarios total "
          f"(独立音频/交织音频 × 单条/batch调用, batch_size={BATCH_SIZE})\n")

    # load engine (max_num_seqs sized for the batch scenarios; sequential
    # scenarios only ever submit 1 prompt at a time, so this is harmless there)
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
        max_num_seqs=max(BATCH_SIZE, 1),
        max_model_len=32768,
        dtype="bfloat16",
        seed=1234,
        limit_mm_per_prompt={"image": 1, "video": 1, "audio": 1},
        # [2026-07-02 CONFIRMED ROOT CAUSE, see vllm_runner.py for the full
        # writeup] Without this, repeating the SAME video content (as this
        # script deliberately does, to simulate real eval reuse) hits vLLM's
        # multi-modal processor cache on the 2nd+ occurrence, which returns a
        # `None` placeholder that pinned vllm==0.11.0's use_audio_in_video
        # auto-detection doesn't guard against -> "TypeError: 'NoneType'
        # object is not subscriptable". Disabling it here mirrors the
        # already-applied fix in config/models/vllm.yaml.
        mm_processor_cache_gb=0,
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    sp = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=10)
    print("[*] vLLM engine ready\n")

    prompt = "What is happening? Reply with one letter A/B/C/D."
    scenarios = [
        ("独立音频 + 单条调用", False, "sequential"),
        ("交织音频 + 单条调用", True, "sequential"),
        ("独立音频 + batch调用", False, "batch"),
        ("交织音频 + batch调用", True, "batch"),
    ]

    summary = []
    for name, use_audio_in_video, mode in scenarios:
        print(f"{'='*60}\n[*] Scenario: {name}\n{'='*60}")
        t0 = time.time()
        if mode == "batch":
            ok, fail, first_fail = _run_batch(
                llm, processor, process_mm_info, samples[:N], prompt, sp,
                use_audio_in_video, BATCH_SIZE
            )
        else:
            ok, fail, first_fail = _run_sequential(
                llm, processor, process_mm_info, samples[:N], prompt, sp,
                use_audio_in_video
            )
        elapsed = time.time() - t0
        print(f"  -> ok={ok}  fail={fail}  elapsed={elapsed:.0f}s"
              + (f"  first_fail={first_fail}" if first_fail is not None else "") + "\n")
        summary.append((name, ok, fail, elapsed))

    print(f"{'='*60}\n[*] Summary\n{'='*60}")
    all_ok = True
    for name, ok, fail, elapsed in summary:
        status = "\u2705 PASS" if fail == 0 else "\u274c FAIL"
        all_ok = all_ok and (fail == 0)
        print(f"  {status}  {name:<18}  ok={ok} fail={fail}  elapsed={elapsed:.0f}s")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
