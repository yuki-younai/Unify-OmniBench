"""Test OpenAIChatModel against a local vLLM-omni server.

Directly exercises the openai backend (config/models/openai.yaml) via
InferenceRequest / MediaRef — the same path used in production eval.
Uses ``modalities=["text"]`` to skip Talker/Token2Wav audio generation.

Prereq:
    # 纯 vLLM (推荐):
    bash ../../vllm_deploy.sh
    # 或者 vllm-omni (--omni 多阶段 pipeline):
    bash ../../vllm_omni_deploy.sh

Usage:
    python tests/test_qwen_omni_openai.py
    URL=http://localhost:8001/v1  python tests/test_qwen_omni_openai.py

Coverage:
    use_audio_in_video wiring  — client-side only, no server needed (runs first,
                                 fails fast if av requests are built incorrectly).
                                 Asserts whichever mode is CURRENTLY configured
                                 (see config/models/openai.yaml):
                                   * use_audio_in_video=false (current default —
                                     matches the transformer reference backend's
                                     hardcoded False): audio_url/input_audio block
                                     MUST be present (independent audio input,
                                     not interleaved into video's position encoding)
                                   * use_audio_in_video=true (video_mode=video_mp4/
                                     file_url only): audio block must be DROPPED
                                     client-side, extra_body.mm_processor_kwargs.
                                     use_audio_in_video must be set instead
                                   * qwen_native mode (predecoded JPEG sequence, not
                                     a real container) must ALWAYS keep the audio
                                     block regardless of use_audio_in_video (server
                                     can't extract audio from a JPEG sequence)
    temporal ordering  — REQUIRES server; ~4-5 short requests EACH on a
                         synthetically generated 30s video with objectively-
                         known event order (see check_temporal_ordering()),
                         run TWICE: once video-only (mode="visual") and once
                         with an audio track attached (mode="av") — isolates
                         whether a video-ordering regression comes from the raw
                         second_per_grid_ts computation or from merely having
                         audio media present in the request (independent or
                         interleaved, depending on use_audio_in_video config —
                         position-interleaving logic itself (only the latter
                         gets exercised on every real Daily-Omni sample,
                         since those always ship both video+audio). Cheap,
                         fast substitute for a full eval.sh run — no need to
                         wait for a 1200-sample Daily-Omni pass just to know
                         whether video temporal encoding works at all.
    text-only          — warmup + single
    video (mp4)        — serial: visual-only  (server-side decode, 128 frames/2fps)
    audio (data:)      — serial: audio-only
    video + audio      — serial: av (取决于 use_audio_in_video 配置：false=独立发送
                         音频, true=从视频自带音轨提取)
    text concurrency   — N threads, text-only
    video concurrency  — M threads, visual-only
    av concurrency     — M threads, video+audio
"""
from __future__ import annotations

import os, sys, time, shutil, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import unify_omnibench.models.api.openai_chat as openai_chat_mod
from unify_omnibench.models.api.openai_chat import OpenAIChatModel
from unify_omnibench.core.types import InferenceRequest, MediaRef, Sample

try:
    import cv2  # noqa: F401 -- only needed for check_temporal_ordering()
    import numpy as np  # noqa: F401
except ImportError:
    cv2 = None
    np = None

VIDEO = os.path.join(ROOT, "example", "draw.mp4")
AUDIO = os.path.join(ROOT, "example", "cough.wav")
URL   = os.environ.get("URL", "http://localhost:8001/v1")
N     = int(os.environ.get("N", "8"))

# ── build model (mirrors openai.yaml) ────────────────────────────────
model = OpenAIChatModel({
    "name": "openai_chat",
    "model": os.environ.get("MODEL_NAME", "Qwen2.5-Omni-7B"),
    "base_url": URL,
    "api_key": "EMPTY",
    "system_prompt": (
        "You are Qwen, a virtual human developed by the Qwen Team, "
        "Alibaba Group, capable of perceiving auditory and visual inputs, "
        "as well as generating text and speech."
    ),
    # video_mp4 + use_audio_in_video: 服务端从同一个mp4容器解出画面+音轨，音画时序
    # 天然同步（与 config/models/openai.yaml 保持一致）
    "video": {"mode": "video_mp4", "num_frames": 128, "fps": 2},
    "audio": {"mode": "audio_url"},
    "use_audio_in_video": False,  # 对齐 openai.yaml 当前默认值
})
model.load()

GEN_SHORT = {"temperature": 0.0, "max_new_tokens": 20}
GEN_DESC  = {"temperature": 0.0, "max_new_tokens": 256}


def make_req(question, media=None, mode="av", gen=None):
    sample = Sample(
        uid="test", dataset="test",
        question=question, choices=[],
        media=media or [],
    )
    return InferenceRequest(
        sample=sample,
        modality_mode=mode,
        prompt_template=question,   # 直接用问题原文，绕过多选题包装
        generation_kwargs=gen or GEN_SHORT,
    )


def _short(content):
    """Printable copy of content blocks with long base64 payloads truncated."""
    out = []
    for c in content:
        if not isinstance(c, dict):
            out.append(c)
            continue
        c2 = dict(c)
        for key in ("video_url", "audio_url", "image_url", "input_audio"):
            if key in c2 and isinstance(c2[key], dict):
                v = dict(c2[key])
                for f in ("url", "data"):
                    if f in v and isinstance(v[f], str) and len(v[f]) > 80:
                        v[f] = v[f][:60] + f"...<{len(v[f])} chars>"
                c2[key] = v
        out.append(c2)
    return out


def check_use_audio_in_video_wiring() -> bool:
    """CPU-only sanity check (no HTTP call) for the use_audio_in_video +
    mm_processor_kwargs.fps wiring.

    Asserts the CURRENTLY CONFIGURED behavior is actually what gets built,
    for whichever value ``use_audio_in_video`` currently has:

      * use_audio_in_video=True  (video_mp4/file_url only): the av request
        must NOT emit a separate audio_url/input_audio block (audio is
        expected to come from the video's own track server-side), and
        extra_body.mm_processor_kwargs.use_audio_in_video must be set.
      * use_audio_in_video=False (current default — matches the
        transformer reference backend's hardcoded False, see
        models/local/qwen25omni.py::build_messages): the av request MUST
        still emit a separate audio_url/input_audio block — silently
        dropping it here would mean the openai backend loses audio
        entirely instead of falling back to independent audio input.

    Either way, a video_url block must be present, and (whenever video is
    present) extra_body.mm_processor_kwargs.fps must be set — this is the
    only channel Qwen2_5OmniProcessor reads to compute second_per_grid_ts
    (video temporal position encoding); if missing, it silently falls back
    to a hardcoded 2.0 default, which only happens to be correct by
    coincidence with our config.
    """
    print("\n[*] use_audio_in_video + mm_processor_kwargs.fps wiring check (client-side only, no server call)")
    req = make_req(
        "Describe what you see and hear in this video.",
        media=[MediaRef(kind="video", path=VIDEO), MediaRef(kind="audio", path=AUDIO)],
        mode="av",
    )
    messages = model.build_messages(req)
    content = messages[-1]["content"]
    types = [c["type"] for c in content if isinstance(c, dict)]
    print(f"    config: use_audio_in_video={model.use_audio_in_video} video_mode={model.video_mode!r}")
    print("    content blocks:", _short(content))

    ok = True
    has_audio_block = "audio_url" in types or "input_audio" in types
    dedup_expected = model.use_audio_in_video and model.video_mode in ("video_mp4", "file_url")
    if dedup_expected:
        if has_audio_block:
            print("    \u274c FAIL: separate audio block present — should be skipped "
                  "(server should extract audio from the video track instead)")
            ok = False
    else:
        if not has_audio_block:
            print("    \u274c FAIL: audio block missing — with use_audio_in_video=False, "
                  "audio MUST be sent independently or it's silently lost")
            ok = False
    if "video_url" not in types:
        print("    \u274c FAIL: video_url block missing")
        ok = False

    # Replicate generate()'s extra_body computation to check mm_processor_kwargs wiring
    # (generate() builds this inline and doesn't expose it, so we mirror the same
    # condition here instead of duplicating a real HTTP call).
    refs = model._media_refs_for_mode(req.sample, req.modality_mode)
    has_video = any(m.kind == "video" for m in refs)
    has_audio = any(m.kind == "audio" for m in refs)
    would_set_audio_flag = (
        has_video and has_audio
        and model.use_audio_in_video and model.video_mode in ("video_mp4", "file_url")
    )
    would_set_fps = has_video and model.video_fps is not None
    print(f"    extra_body.mm_processor_kwargs.use_audio_in_video would be set: {would_set_audio_flag}")
    print(f"    extra_body.mm_processor_kwargs.fps would be set: {would_set_fps} (value={model.video_fps})")
    if dedup_expected and not would_set_audio_flag:
        print("    \u274c FAIL: expected mm_processor_kwargs.use_audio_in_video to be set for this av request")
        ok = False
    if not dedup_expected and would_set_audio_flag:
        print("    \u274c FAIL: mm_processor_kwargs.use_audio_in_video should NOT be set "
              "when use_audio_in_video=False")
        ok = False
    if not would_set_fps:
        print("    \u274c FAIL: expected mm_processor_kwargs.fps to be set whenever video is present — "
              "without it, second_per_grid_ts relies on an implicit server-side default that may not "
              "match the actual sampled frame rate")
        ok = False

    print("    " + ("\u2705 正常" if ok else "\u274c 有问题"))
    return ok


def check_qwen_native_mode_keeps_audio() -> bool:
    """Mutual-exclusivity check: when video_mode='qwen_native' (a predecoded
    JPEG-sequence, not a real container the server can pull an audio track
    from), the audio block must NOT be skipped even if use_audio_in_video is
    True — otherwise audio would silently disappear from the request.

    Monkeypatches the (slow, real-decode) frame extractor so this only
    exercises the audio-block branch logic, not actual video decoding.
    """
    print("\n[*] qwen_native mode must NOT drop audio (mutual-exclusivity check)")
    orig_mode = model.video_mode
    orig_extract = openai_chat_mod.extract_qwen_native_frames_base64
    openai_chat_mod.extract_qwen_native_frames_base64 = lambda *a, **kw: (["deadbeef"], 2.0)
    ok = True
    try:
        model.video_mode = "qwen_native"
        req = make_req(
            "Describe what you see and hear in this video.",
            media=[MediaRef(kind="video", path=VIDEO), MediaRef(kind="audio", path=AUDIO)],
            mode="av",
        )
        messages = model.build_messages(req)
        types = [c["type"] for c in messages[-1]["content"] if isinstance(c, dict)]
        print("    content block types:", types)
        if "audio_url" not in types and "input_audio" not in types:
            print("    \u274c FAIL: audio block missing in qwen_native mode "
                  "(server can't extract audio from a JPEG-sequence — this would silently lose audio)")
            ok = False
    finally:
        model.video_mode = orig_mode
        openai_chat_mod.extract_qwen_native_frames_base64 = orig_extract

    print("    " + ("\u2705 正常" if ok else "\u274c 有问题"))
    return ok


def _generate_sequence_video(path, labels, seconds_each=6.0, fps=10, size=(320, 240)):
    """Synthesize an mp4 where each ``labels[i]`` word is shown alone, in
    order, for ``seconds_each`` seconds, on a distinct solid-color background.

    This gives us an OBJECTIVELY KNOWN ground-truth event order that is
    completely decoupled from real-world video content recognition — a
    wrong answer here can only mean the video TEMPORAL POSITION ENCODING
    pipeline (frame sampling rate vs. second_per_grid_ts) is broken, not
    that the model failed to recognize some real-world action/object.

    ``fps``/``size`` only affect the *source* file's native encoding — the
    server resamples to whatever ``--media-io-kwargs``/``mm_processor_kwargs.fps``
    dictate, same as any real video, so this is a faithful smoke test of
    the exact pipeline used in production.
    """
    colors = [
        (60, 60, 220), (60, 200, 60), (220, 160, 40),
        (200, 60, 200), (60, 200, 200), (200, 200, 60),
    ]
    w, h = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {path}")
    try:
        for i, label in enumerate(labels):
            color = colors[i % len(colors)]
            frame = np.full((h, w, 3), color, dtype=np.uint8)
            font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4
            (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
            x, y = max(0, (w - tw) // 2), (h + th) // 2
            cv2.putText(frame, label, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
            for _ in range(int(seconds_each * fps)):
                writer.write(frame)
    finally:
        writer.release()


FFMPEG_BIN = shutil.which("ffmpeg")


def _mux_audio_track(video_path: str, audio_path: str, out_path: str, duration: float) -> "str | None":
    """Mux a real audio stream into ``video_path`` (which has NONE — cv2's
    VideoWriter cannot write audio) so the video's own MP4 container
    actually contains a 'soun' track.

    This is required to exercise ``use_audio_in_video``'s server-side audio
    extraction path safely: with a video that has zero audio streams, the
    server errors out with a bare 500 when it tries to extract one — that's
    a container-format edge case, NOT the position-interleaving bug we're
    trying to isolate. Content is irrelevant (we loop ``audio_path`` to
    cover the full ``duration`` and hard-cut with ``-t``), we only need
    SOME real audio track physically present.
    """
    if not FFMPEG_BIN:
        return None
    cmd = [
        FFMPEG_BIN, "-y", "-loglevel", "error",
        "-i", video_path,
        "-stream_loop", "-1", "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac",
        "-t", str(duration),
        out_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0 or not os.path.exists(out_path):
        print(f"    [ffmpeg mux failed, rc={result.returncode}] "
              f"{result.stderr.decode(errors='replace')[-500:]}")
        return None
    return out_path


def check_temporal_ordering(with_audio: bool = False) -> bool:
    """Cheap, fast substitute for a full eval.sh run: sanity-check VIDEO
    TEMPORAL POSITION ENCODING using a synthetic video with objectively-known
    event order, instead of waiting ~20min for a 1200-sample Daily-Omni pass
    to see if 'Event Sequence' accuracy improved.

    Args:
        with_audio: if True, attach a real (content-irrelevant) audio track
            and send as ``mode="av"`` so the request also carries audio
            media, matching the actual shape of every real Daily-Omni
            request (which always ships both video+audio). The concrete
            wiring this exercises depends on the model's CURRENT
            ``use_audio_in_video`` config (see openai.yaml):
              * False (current default, matches the transformer reference
                backend's hardcoded False) -> audio is sent as an
                INDEPENDENT block alongside video_url, NOT interleaved into
                the video's own temporal position encoding.
              * True -> audio is dropped client-side and
                mm_processor_kwargs.use_audio_in_video=True is set instead,
                triggering server-side audio/video position-interleaving
                (``get_chunked_index`` in Qwen2_5OmniProcessor).
            Running both the visual-only and this av variant side-by-side
            checks whether merely having audio media present in the request
            (regardless of interleaving) disturbs the video's own event
            ordering — e.g. via prompt-template/token-budget side effects.

    Requires a running server — unlike the wiring checks above, this
    actually calls model.generate() (~4-5 short requests total per variant).
    """
    if cv2 is None:
        print("\n[*] Temporal ordering check SKIPPED (cv2/numpy not installed in this env)")
        return True

    labels = ["APPLE", "BANANA", "CHERRY", "DURIAN", "EGGPLANT"]
    seconds_each = 6.0
    total_s = len(labels) * seconds_each
    silent_path = os.path.join(ROOT, "example", "_sequence_test_silent.mp4")
    av_path = os.path.join(ROOT, "example", "_sequence_test_av.mp4")
    if with_audio:
        tag = ("video+audio (av, interleaved via use_audio_in_video=True)"
               if model.use_audio_in_video else
               "video+audio (av, audio sent independently, use_audio_in_video=False)")
    else:
        tag = "video-only (visual)"
    print(f"\n[*] Temporal ordering check [{tag}]: {total_s:.0f}s synthetic video "
          f"({' -> '.join(labels)})")
    _generate_sequence_video(silent_path, labels, seconds_each=seconds_each)

    video_path = silent_path
    if with_audio:
        # cv2.VideoWriter cannot write audio, so `silent_path` has ZERO audio
        # streams. If we sent it as-is with use_audio_in_video=True, the
        # server would try to extract a non-existent audio track and error
        # out with a bare 500 — a container-format edge case, NOT the
        # position-interleaving bug this variant is meant to isolate. Mux in
        # a real (content-irrelevant) audio track first so the video's own
        # container actually has a 'soun' stream, same as every real
        # Daily-Omni sample.
        muxed = _mux_audio_track(silent_path, AUDIO, av_path, duration=total_s)
        if muxed is None:
            print("    (ffmpeg unavailable or mux failed — cannot embed an audio "
                  "track into the synthetic video, skipping av variant)")
            try:
                os.remove(silent_path)
            except OSError:
                pass
            return True
        video_path = muxed

    ok = True
    try:
        def ask(question, gen=None):
            media = [MediaRef(kind="video", path=video_path)]
            mode = "visual"
            if with_audio:
                # Content is irrelevant here — we only need SOME audio media
                # present so use_audio_in_video's interleaving path actually
                # triggers, matching what happens on every real Daily-Omni
                # sample (which always ships both video+audio). The actual
                # bytes sent for this block get dropped client-side anyway
                # (see check_use_audio_in_video_wiring) — the audio that
                # matters is the one now muxed into `video_path` itself.
                media.append(MediaRef(kind="audio", path=AUDIO))
                mode = "av"
            req = make_req(question, media=media, mode=mode, gen=gen or GEN_SHORT)
            return model.generate(req).strip()

        # 1) first / last word — cheapest, most direct probes of "does the
        #    model know WHEN in the video it is looking", i.e. is the start
        #    of the video anchored near position 0 and the end near the max
        #    position, or is the whole timeline squeezed/stretched/shifted.
        for desc, question, expected in [
            ("word at the very START of the video",
             "What single word appears on screen at the very beginning "
             "(first few seconds) of this video? Answer with only that one word.",
             labels[0]),
            ("word at the very END of the video",
             "What single word appears on screen at the very end "
             "(last few seconds) of this video? Answer with only that one word.",
             labels[-1]),
        ]:
            answer = ask(question)
            other_present = [l for l in labels if l != expected and l.lower() in answer.lower()]
            passed = expected.lower() in answer.lower() and not other_present
            print(f"── {desc} ──\n    Q: {question}\n    A: {answer!r}")
            if passed:
                print(f"    \u2705 correct (expected {expected!r})")
            else:
                print(f"    \u274c FAIL: expected {expected!r} not (unambiguously) found")
                ok = False

        # 2) full chronological order — requires correctly ordering ALL 5
        #    segments relative to each other, a stricter test than just the
        #    endpoints; failure here with endpoints correct would point at
        #    distortion in the MIDDLE of the timeline specifically.
        order_question = (
            "This video shows five words appearing one after another, each on "
            "a different colored background, for several seconds each. List "
            "the five words in the exact order they appear, separated by commas."
        )
        answer = ask(order_question, gen={"temperature": 0.0, "max_new_tokens": 60})
        print(f"── full chronological order ──\n    Q: {order_question}\n    A: {answer!r}")
        lower, search_from, positions, found_all = answer.lower(), 0, [], True
        for l in labels:
            idx = lower.find(l.lower(), search_from)
            if idx == -1:
                found_all = False
                break
            positions.append(idx)
            search_from = idx + len(l)
        if found_all and positions == sorted(positions):
            print(f"    \u2705 correct order (expected {labels})")
        else:
            print(f"    \u274c FAIL: expected order {labels}, answer doesn't preserve it")
            ok = False
    finally:
        for p in (silent_path, av_path):
            try:
                os.remove(p)
            except OSError:
                pass

    print("    " + ("\u2705 正常 (temporal ordering looks correct)" if ok else
                     "\u274c 有问题 (temporal encoding still broken — see FAILs above)"))
    return ok


def run_concurrency(label, reqs, workers):
    print(f"\n[*] {label} x{workers} ({len(reqs)} reqs) ...")
    t0 = time.time()
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(model.generate, r): i for i, r in enumerate(reqs)}
        for fut in as_completed(futures):
            try:
                out = fut.result()
                ok += 1
            except Exception as e:
                fail += 1
                print(f"  FAIL [{futures[fut]}]: {str(e)[:120]}")
    elapsed = time.time() - t0
    status = "✅ 正常" if fail == 0 else f"❌ 失败 {fail}/{len(reqs)}"
    print(f"Done. ok={ok} fail={fail}  {elapsed:.1f}s  {status}")
    return fail == 0


# ── client-side wiring checks (no server needed, run first) ─────────────
wiring_ok = check_use_audio_in_video_wiring()
wiring_ok = check_qwen_native_mode_keeps_audio() and wiring_ok

# ── warmup ─────────────────────────────────────────────────────────────
out = model.generate(make_req("Introduce yourself in one sentence.", mode="text", gen=GEN_DESC))
print(f"[*] Warmup:\n{out}\n")

# ── temporal ordering check (requires server, ~4-5 short requests per variant) ──
try:
    temporal_ok_visual = check_temporal_ordering(with_audio=False)
except Exception as e:
    print(f"  FAIL temporal ordering check (visual): {str(e)[:200]}")
    temporal_ok_visual = False
try:
    temporal_ok_av = check_temporal_ordering(with_audio=True)
except Exception as e:
    print(f"  FAIL temporal ordering check (av): {str(e)[:200]}")
    temporal_ok_av = False
temporal_ok = temporal_ok_visual and temporal_ok_av
if temporal_ok_visual and not temporal_ok_av:
    mode_desc = "interleaving (use_audio_in_video=True)" if model.use_audio_in_video else \
                "independent audio input (use_audio_in_video=False)"
    print(f"\n[!] Visual-only ordering is fine but AV (with audio attached, {mode_desc}) "
          "ordering broke — this points at the audio-attached code path specifically, "
          "NOT the raw per-video second_per_grid_ts computation.")

# ── serial tests ───────────────────────────────────────────────────────
serial_cases = [
    ("text-only",
     make_req("Explain multimodal AI in one sentence.", mode="text", gen=GEN_DESC)),

    ("video only (mp4)",
     make_req("Describe what is happening in this video in detail.",
              media=[MediaRef(kind="video", path=VIDEO)],
              mode="visual", gen=GEN_DESC)),

    ("audio only (data:)",
     make_req("Describe the audio content in detail. What do you hear?",
              media=[MediaRef(kind="audio", path=AUDIO)],
              mode="audio", gen=GEN_DESC)),

    ("video + audio (av)",
     make_req("Describe what you see and hear in this video.",
              media=[
                  MediaRef(kind="video", path=VIDEO),
                  MediaRef(kind="audio", path=AUDIO),
              ],
              mode="av", gen=GEN_DESC)),
]

for label, req in serial_cases:
    try:
        out = model.generate(req)
        print(f"── {label} ──")
        print(out[:200])
        print()
    except Exception as e:
        print(f"  FAIL {label:26s}  {str(e)[:120]}")

# ── concurrency tests ──────────────────────────────────────────────────
M = min(N, 4)

run_concurrency(
    "Text concurrency",
    [make_req(f"Say the number {i}.", mode="text") for i in range(N)],
    workers=N,
)

run_concurrency(
    "Video-only concurrency",
    [make_req("What is shown in this video? One sentence.",
              media=[MediaRef(kind="video", path=VIDEO)],
              mode="visual", gen=GEN_SHORT)
     for _ in range(M * 2)],
    workers=M,
)

run_concurrency(
    "Video+Audio (av) concurrency",
    [make_req("Describe what you see and hear in one sentence.",
              media=[
                  MediaRef(kind="video", path=VIDEO),
                  MediaRef(kind="audio", path=AUDIO),
              ],
              mode="av", gen=GEN_SHORT)
     for _ in range(M * 2)],
    workers=M,
)

# ── final summary ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("use_audio_in_video / mm_processor_kwargs wiring:", "✅ 正常" if wiring_ok else "❌ 有问题（见上方 FAIL）")
print("temporal ordering, video-only          :", "✅ 正常" if temporal_ok_visual else "❌ 有问题（见上方 FAIL）")
print("temporal ordering, video+audio (av)    :", "✅ 正常" if temporal_ok_av else "❌ 有问题（见上方 FAIL）")
if not (wiring_ok and temporal_ok):
    sys.exit(1)
