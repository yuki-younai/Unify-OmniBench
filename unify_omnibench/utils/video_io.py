"""Video / audio IO helpers (frame extraction, base64 encoding, ffmpeg).

Data-URL encoding utilities
----------------------------
Use ``encode_*_data_url`` functions to produce self-contained ``data:`` URIs
suitable for OpenAI-compatible API payloads:

    * :func:`encode_image_data_url`  — ``data:image/<ext>;base64,<b64>``
    * :func:`encode_audio_data_url`  — ``data:audio/<ext>;base64,<b64>``
    * :func:`encode_video_data_url`  — ``data:video/jpeg;base64,<f1>,<f2>,...``
      (multi-frame format required by vLLM for Qwen2.5-Omni)
    * :func:`encode_video_audio_data_urls` — returns (video_url, audio_url)
      extracting both from a single video file
"""
from __future__ import annotations

import base64
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple


# -------------------------------------------------------------- base64 helpers
def encode_file_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_video_base64(path: str) -> str:
    return encode_file_base64(path)


def encode_audio_base64(path: str) -> str:
    return encode_file_base64(path)


# -------------------------------------------------------------- ffmpeg helpers
def _ffmpeg_path() -> str:
    return os.environ.get("FFMPEG_PATH") or shutil.which("ffmpeg") or "ffmpeg"


def extract_audio_from_video(
    video_path: str,
    out_path: Optional[str] = None,
    ffmpeg_path: Optional[str] = None,
    ar: int = 16000,
    ac: int = 1,
) -> str:
    """Extract mono 16k WAV. Returns the output path. Caller is responsible for cleanup."""
    ffm = ffmpeg_path or _ffmpeg_path()
    if out_path is None:
        out_path = (
            f"/tmp/temp_audio_{os.path.basename(video_path)}_"
            f"{random.randint(1000, 99999)}.wav"
        )
    cmd = [
        ffm, "-i", video_path, "-vn",
        "-ac", str(ac), "-ar", str(ar),
        "-y", out_path, "-hide_banner", "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_path


def strip_audio_from_video(
    video_path: str,
    out_path: Optional[str] = None,
    ffmpeg_path: Optional[str] = None,
) -> str:
    """Return a copy of the video without the audio track. Caller cleans up."""
    ffm = ffmpeg_path or _ffmpeg_path()
    if out_path is None:
        out_path = (
            f"/tmp/temp_no_audio_{os.path.basename(video_path)}_"
            f"{random.randint(1000, 99999)}.mp4"
        )
    cmd = [
        ffm, "-i", video_path, "-an", "-vcodec", "copy", "-y", out_path,
        "-hide_banner", "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_path


# -------------------------------------------------------------- frame extract
# Smart frame-count constants (mirror qwen_omni_utils.vision_process)
_SMART_FPS = 2.0          # default target fps
_SMART_MIN_FRAMES = 4     # minimum frames
_SMART_MAX_FRAMES = 768   # maximum frames
_SMART_FRAME_FACTOR = 2   # align nframes to this factor


def smart_nframes(
    total_frames: int,
    video_fps: float,
    target_fps: float = _SMART_FPS,
    min_frames: int = _SMART_MIN_FRAMES,
    max_frames: int = _SMART_MAX_FRAMES,
) -> int:
    """Replicate qwen_omni_utils.vision_process.smart_nframes logic.

    Calculate the number of frames to sample based on video duration and target fps,
    clamped to [min_frames, max_frames] and aligned to FRAME_FACTOR.
    """
    import math
    factor = _SMART_FRAME_FACTOR
    nframes = total_frames / video_fps * target_fps
    if nframes > total_frames:
        nframes = total_frames
    min_f = math.ceil(min_frames / factor) * factor
    max_f = math.floor(min(max_frames, total_frames) / factor) * factor
    nframes = min(max(nframes, min_f), max_f)
    nframes = math.floor(nframes / factor) * factor
    return int(max(nframes, factor))


def probe_video_frame_count(
    video_path: str,
    target_fps: float = _SMART_FPS,
    min_frames: int = _SMART_MIN_FRAMES,
    max_frames: int = _SMART_MAX_FRAMES,
) -> int:
    """Cheaply probe a video's (total_frames, native_fps) via OpenCV metadata
    only (no frame decoding) and return the frame count the vLLM server
    should sample for it — mirrors ``qwen_omni_utils.smart_nframes`` exactly,
    i.e. the SAME per-video dynamic frame count the transformer backend uses
    (``duration_s * target_fps``, clamped to ``[min_frames, max_frames]``).

    Used by ``openai_chat.py`` to override the server's ``video_mp4``/
    ``file_url`` frame sampling **per request** via the top-level
    ``media_io_kwargs`` field, instead of relying on the server's fixed
    ``--media-io-kwargs`` startup default (see ``vllm_deploy.sh``) which
    samples the SAME frame count for every video regardless of its actual
    duration — a real mismatch vs. the transformer reference for any video
    whose ``duration_s * target_fps`` isn't coincidentally close to that
    fixed default.
    """
    import cv2  # local import; opencv may not be installed for API-only users
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS) or 1.0
    finally:
        video.release()
    if total <= 0:
        raise RuntimeError(f"No frames found in {video_path}")
    return smart_nframes(
        total_frames=total, video_fps=fps,
        target_fps=target_fps, min_frames=min_frames, max_frames=max_frames,
    )


def extract_frames_base64(
    video_path: str,
    seconds_per_frame: float = 2.0,
    max_frames: Optional[int] = None,
) -> List[str]:
    """Extract frames uniformly across the full video duration, JPEG-base64 encoded.

    Uses :func:`smart_nframes` to compute the number of frames — matching
    ``qwen_omni_utils.vision_process.smart_nframes`` exactly.
    """
    import cv2  # local import; opencv may not be installed for API-only users
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS) or 1.0
        if total <= 0:
            raise RuntimeError(f"No frames found in {video_path}")

        # 完全对齐 smart_nframes 逻辑
        target_fps = 1.0 / seconds_per_frame if seconds_per_frame > 0 else _SMART_FPS
        target = smart_nframes(
            total_frames=total,
            video_fps=fps,
            target_fps=target_fps,
            max_frames=max_frames or _SMART_MAX_FRAMES,
        )

        # uniform sampling via linspace (matches qwen_omni_utils)
        import numpy as np
        indices = np.linspace(0, total - 1, target, dtype=int).tolist()

        out: List[str] = []
        for idx in indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = video.read()
            if not ok:
                continue
            # JPEG 最高质量编码，尽量减少有损压缩损失
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            if ok:
                out.append(base64.b64encode(buf).decode("utf-8"))
    finally:
        video.release()
    if not out:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return out


# -------------------------------------------------------------- data-URL helpers
_AUDIO_MIME = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg", ".flac": "audio/flac", ".m4a": "audio/mp4",
}
_IMAGE_MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
}


def encode_image_data_url(path: str) -> str:
    """Return ``data:image/<ext>;base64,<b64>`` for an image file."""
    ext = Path(path).suffix.lower()
    mime = _IMAGE_MIME.get(ext, "image/jpeg")
    return f"data:{mime};base64,{encode_file_base64(path)}"


def encode_audio_data_url(path: str) -> str:
    """Return ``data:audio/<ext>;base64,<b64>`` for an audio file.

    Reads the file directly without resampling — no librosa dependency.
    """
    ext = Path(path).suffix.lower()
    mime = _AUDIO_MIME.get(ext, "audio/wav")
    return f"data:{mime};base64,{encode_file_base64(path)}"


def encode_video_data_url(
    path: str,
    seconds_per_frame: float = 0.5,   # 2fps，与 qwen_omni_utils smart_nframes 默认 FPS=2 对齐
    max_frames: int = 768,             # 与 qwen_omni_utils FPS_MAX_FRAMES=768 对齐
) -> str:
    """Return ``data:video/jpeg;base64,<f1>,<f2>,...`` (vLLM multi-frame format).

    Default parameters mirror qwen_omni_utils.vision_process.smart_nframes:
      - seconds_per_frame=0.5 → 2fps (FPS=2.0)
      - max_frames=768 (FPS_MAX_FRAMES=768)
    """
    frames = extract_frames_base64(path, seconds_per_frame=seconds_per_frame,
                                   max_frames=max_frames)
    return f"data:video/jpeg;base64,{','.join(frames)}"


def extract_qwen_native_frames_base64(
    video_path: str,
    fps: float = 2.0,
    min_frames: int = 4,
    max_frames: int = 768,
) -> Tuple[List[str], float]:
    """Extract + resize video frames using the **exact same** code path as
    ``qwen_omni_utils.vision_process.fetch_video`` (used by the transformers /
    vllm-offline backends), to guarantee pixel-level parity with those paths:

      * same frame-index selection: ``torch.linspace(...).round()`` (our old
        ``extract_frames_base64`` used ``np.linspace(..., dtype=int)`` which
        **truncates** instead of rounding — a real, if small, discrepancy);
      * same video-reader backend priority (torchcodec > decord > torchvision);
      * same "video pixel budget" resize (``VIDEO_MIN_PIXELS``/``VIDEO_MAX_PIXELS``,
        28-aligned) — our old function sent frames at **native video
        resolution**, which is NOT what Qwen was tuned to see for video input
        and is the most likely cause of degraded accuracy vs. transformer.

    Returns ``(jpeg_b64_frames, achieved_fps)`` — ``achieved_fps`` is what
    ``fetch_video`` actually used (may differ slightly from the requested
    ``fps`` due to clamping), so the caller can report it truthfully to the
    server instead of assuming the nominal target fps.
    """
    from qwen_omni_utils.v2_5.vision_process import fetch_video  # type: ignore
    import io as _io
    from PIL import Image

    video_tensor, sample_fps = fetch_video(
        {"video": video_path, "fps": fps, "min_frames": min_frames, "max_frames": max_frames},
        return_video_sample_fps=True,
    )
    # video_tensor: (T, C, H, W) float tensor, pixel range ~[0, 255] after
    # fetch_video's resize (torchvision resize does not rescale to [0,1]).
    frames_b64: List[str] = []
    for t in range(video_tensor.shape[0]):
        arr = video_tensor[t].permute(1, 2, 0).clamp(0, 255).round().byte().cpu().numpy()  # HWC uint8
        img = Image.fromarray(arr)
        buf = _io.BytesIO()
        # subsampling=0 -> 4:4:4 (no chroma subsampling). PIL's JPEG encoder
        # defaults to 4:2:0 even at quality=100, which silently discards half
        # the color resolution — the transformer path never touches JPEG at
        # all (raw float tensor all the way), so this is the only remaining
        # lossy step in the qwen_native pipeline; eliminate it explicitly.
        img.save(buf, format="JPEG", quality=100, subsampling=0)
        frames_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    if not frames_b64:
        raise RuntimeError(f"No frames extracted from {video_path} via qwen_omni_utils.fetch_video")
    return frames_b64, float(sample_fps)


def encode_audio_data_url_16k(path: str, sr: int = 16000) -> str:
    """Resample audio to ``sr`` Hz mono WAV before base64-encoding.

    Matches ``qwen_omni_utils.audio_process.process_audio_info``'s
    ``SAMPLE_RATE = 16000`` + ``librosa.load(path, sr=SAMPLE_RATE)`` exactly —
    the plain :func:`encode_audio_data_url` above sends the file's *native*
    sample rate untouched, which may not match what the transformer backend
    actually feeds the model if the source .wav isn't already 16kHz.
    """
    import io as _io
    import librosa
    import soundfile

    y, _ = librosa.load(path, sr=sr, mono=True)
    buf = _io.BytesIO()
    soundfile.write(buf, y, sr, format="WAV")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:audio/wav;base64,{b64}"


def encode_video_audio_data_urls(
    video_path: str,
    seconds_per_frame: float = 1.0,
    max_frames: int = 16,
    audio_resample_hz: int = 16000,
) -> Tuple[str, str]:
    """Extract video frames and audio from a single video file.

    Returns ``(video_data_url, audio_data_url)`` ready for API payloads.
    Uses ffmpeg to extract audio (no librosa dependency).
    """
    video_url = encode_video_data_url(video_path, seconds_per_frame, max_frames)

    tmp = extract_audio_from_video(video_path, ar=audio_resample_hz)
    try:
        audio_url = f"data:audio/wav;base64,{encode_file_base64(tmp)}"
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass

    return video_url, audio_url
