"""Agent tool system: plugin registry + VideoEnv + built-in tools.

New tools are added by subclassing ``BaseTool`` and calling
``ToolRegistry.register()`` — zero changes to the evaluator.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.types import MediaRef


# ── tool result ────────────────────────────────────────────────────────


@dataclass
class ToolResult:
    """Returned by tool execution: text observation + new media refs."""
    observation: str                     # e.g. "[Frames 0.0s-10.0s (num=8)]"
    media: List[MediaRef] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


# ── video environment ──────────────────────────────────────────────────


class VideoEnv:
    """Thin wrapper around a video file for ffprobe / ffmpeg tool execution."""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.tmp_dir = tempfile.mkdtemp(prefix="agent_")
        self.step_counter = 0
        self._meta = self._probe()
        # duration-tolerance check on extracted clips/audio — matches
        # OmniAgent's video_env.py: BYPASS_DURATION_CHECK env var relaxes
        # the tolerance (some containers report slightly off durations
        # after ffmpeg re-encode/copy, causing false-positive failures).
        self.bypass_dur_check = os.getenv("BYPASS_DURATION_CHECK", "false").lower() in (
            "1", "true", "t", "yes", "y",
        )
        self.dur_tol = 99999.0 if self.bypass_dur_check else 1.0
        # Minimum audio duration — matches OmniAgent's video_env.py
        # (hardcoded ``self.min_audio_len = 0.1``, NOT env-configurable
        # there either). This is NOT relaxed by BYPASS_DURATION_CHECK: a
        # near-zero-length audio request/extraction produces 0 decoded
        # PCM samples, which crashes vLLM's Qwen2.5-Omni audio processor
        # with a fatal (uncaught, worker-killing) ValueError — independent
        # of the duration-*tolerance* check above.
        self.min_audio_len = 0.1

    @property
    def has_audio(self) -> bool:
        return bool(self._meta.get("has_audio", False))

    def meta(self) -> Dict[str, Any]:
        return dict(self._meta)

    def _probe(self) -> Dict[str, Any]:
        """ffprobe duration / fps / has_audio."""
        try:
            dur = float(subprocess.check_output([
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "default=nw=1", self.video_path,
            ], text=True, timeout=30).strip())
        except Exception:
            dur = 0.0

        try:
            fps_str = subprocess.check_output([
                "ffprobe", "-v", "quiet", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=nw=1", self.video_path,
            ], text=True, timeout=30).strip()
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) else 25.0
        except Exception:
            fps = 25.0

        try:
            subprocess.check_output([
                "ffprobe", "-v", "quiet", "-select_streams", "a:0",
                "-show_entries", "stream=codec_type",
                "-of", "default=nw=1", self.video_path,
            ], text=True, timeout=30)
            has_audio = True
        except Exception:
            has_audio = False

        return {"duration": dur, "fps": fps, "has_audio": has_audio}

    def _ffmpeg(self, *args) -> str:
        """Run ffmpeg in tmp_dir, return tmp_dir path (callers derive file paths)."""
        self.step_counter += 1
        out = os.path.join(self.tmp_dir, f"step_{self.step_counter:02d}")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + list(args)
        subprocess.run(cmd, check=True, timeout=120, cwd=self.tmp_dir)
        return out

    def get_media_durations(self, file_path: str) -> Optional[Dict[str, float]]:
        """ffprobe a produced media file for its video/audio stream durations.

        Mirrors OmniAgent's ``video_env.py::get_media_durations`` — used to
        sanity-check that ``get_audio``/``get_clip`` actually extracted the
        requested time range (ffmpeg copy/seek can silently produce a
        shorter/longer file near stream boundaries).
        """
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_streams", file_path],
                capture_output=True, text=True, timeout=30, check=True,
            )
            info = json.loads(out.stdout)
        except Exception:
            return None

        video_dur = None
        audio_dur = None
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video" and video_dur is None:
                if "duration" in stream:
                    video_dur = float(stream["duration"])
            elif stream.get("codec_type") == "audio" and audio_dur is None:
                if "duration" in stream:
                    audio_dur = float(stream["duration"])
        return {"video": video_dur, "audio": audio_dur}

    # ── built-in tool implementations ──────────────────────────────

    def get_frames(self, start: float, end: float, num: int = 10) -> List[MediaRef]:
        paths = []
        for k in range(num):
            ts = start + (end - start) * k / max(num - 1, 1) if num > 1 else start
            out = self._ffmpeg(
                "-ss", f"{ts:.3f}", "-i", self.video_path,
                "-frames:v", "1", "-q:v", "2", f"{ts:.3f}.jpg",
            )
            _, _, files = next(os.walk(os.path.dirname(out)))
            jpg = os.path.join(os.path.dirname(out),
                               next(f for f in files if f.endswith(".jpg")))
            paths.append(MediaRef(kind="image", path=jpg))
        return paths

    def get_audio(self, start: float, end: float) -> MediaRef:
        dur_expect = end - start
        out = self._ffmpeg(
            "-ss", f"{start:.3f}", "-i", self.video_path,
            "-t", f"{dur_expect:.3f}", "-map", "0:a:0?",
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            "audio.wav",
        )
        wav = os.path.join(os.path.dirname(out), "audio.wav")
        # validate actual extracted duration — matches OmniAgent's
        # get_audio duration check (dur_tol*5 tolerance + min_audio_len
        # floor, the latter NOT relaxed by BYPASS_DURATION_CHECK — see
        # __init__ comment on why 0.1s is a hard floor, not a tolerance).
        info = self.get_media_durations(wav) or {}
        audio_dur = info.get("audio")
        if (audio_dur is None or audio_dur <= 0
                or audio_dur < self.min_audio_len
                or abs(audio_dur - dur_expect) > self.dur_tol * 5):
            raise RuntimeError(
                f"FFMPEG_AUDIO_FAIL: bad audio duration {audio_dur} (expect {dur_expect:.3f})"
            )
        return MediaRef(kind="audio", path=wav)

    def get_clip(self, start: float, end: float) -> MediaRef:
        dur_expect = end - start
        out = self._ffmpeg(
            "-ss", f"{start:.3f}", "-i", self.video_path,
            "-t", f"{dur_expect:.3f}", "-map", "0:v:0?", "-map", "0:a:0?",
            "-c", "copy", "-avoid_negative_ts", "make_zero", "clip.mp4",
        )
        mp4 = os.path.join(os.path.dirname(out), "clip.mp4")
        # validate actual extracted duration — matches OmniAgent's
        # get_clip duration check (video vs expected, audio vs video sync)
        info = self.get_media_durations(mp4) or {}
        video_dur = info.get("video")
        audio_dur = info.get("audio")
        bad = False
        if video_dur is not None and abs(video_dur - dur_expect) > self.dur_tol * 5:
            bad = True
        elif video_dur is None or video_dur <= 0:
            bad = True
        elif self.has_audio:
            if audio_dur is None or audio_dur <= 0 or audio_dur < self.min_audio_len:
                bad = True
            elif abs(audio_dur - video_dur) > self.dur_tol:
                bad = True
        if bad:
            raise RuntimeError(
                f"FFMPEG_CLIP_FAIL: bad clip duration v={video_dur}, a={audio_dur}, "
                f"expect {dur_expect:.3f}"
            )
        return MediaRef(kind="video", path=mp4)


# ── tool registry ─────────────────────────────────────────────────────


class BaseTool(ABC):
    """Base class for all agent tools."""
    name: str = ""
    description: str = ""
    schema_str: str = ""   # single-line JSON example shown in system prompt

    @abstractmethod
    def execute(self, action: Dict[str, Any], env: VideoEnv) -> ToolResult: ...


class ToolRegistry:
    """Plugin-style tool registry — add tools via ``register()``."""
    _tools: Dict[str, BaseTool] = {}
    _cfg: Dict[str, Any] = {}

    @classmethod
    def register(cls, tool: BaseTool) -> None:
        cls._tools[tool.name] = tool

    @classmethod
    def configure(cls, react_cfg: Dict[str, Any]) -> None:
        """Set react config so tools can read limits (max_frames_len, etc.)."""
        cls._cfg = react_cfg

    @classmethod
    def get_cfg(cls) -> Dict[str, Any]:
        return cls._cfg

    @classmethod
    def get(cls, name: str) -> Optional[BaseTool]:
        return cls._tools.get(name)

    @classmethod
    def all(cls) -> List[BaseTool]:
        return list(cls._tools.values())

    @classmethod
    def tool_descriptions(cls) -> str:
        """Build the "Available tools" block for the system prompt."""
        lines = []
        for t in cls._tools.values():
            if t.name == "answer":
                lines.append(f'- {t.schema_str}  ← submit final answer')
            else:
                lines.append(f'- {t.schema_str}')
        return "\n".join(lines)


# ── built-in tools ────────────────────────────────────────────────────


class GetFramesTool(BaseTool):
    name = "get_frames"
    description = "Extract evenly-spaced video frames"
    schema_str = '{"type": "get_frames", "start": 0.0, "end": 10.0, "num": 8}'

    def execute(self, action: Dict[str, Any], env: VideoEnv) -> ToolResult:
        limit = ToolRegistry.get_cfg().get("max_frames_len", 60)
        start = float(action.get("start", 0))
        end = float(action.get("end", 0))
        num = min(int(action.get("num", 10)), limit)
        frames = env.get_frames(start, end, num)
        rng = f"[Frames {start:.1f}s-{end:.1f}s (num={len(frames)})]"
        return ToolResult(rng, frames)


class GetAudioTool(BaseTool):
    name = "get_audio"
    description = "Extract audio from a time range"
    schema_str = '{"type": "get_audio", "start": 0.0, "end": 15.0}'

    def execute(self, action: Dict[str, Any], env: VideoEnv) -> ToolResult:
        # matches OmniAgent's video_env.py: get_audio is FORBIDDEN on a
        # video with no audio track — reject up front instead of silently
        # returning a blank/empty wav.
        if not env.has_audio:
            raise ValueError("NO_AUDIO: video has no audio track")
        limit = ToolRegistry.get_cfg().get("max_audio_len", 300)
        start = float(action.get("start", 0))
        end = min(float(action.get("end", 0)), start + limit)
        # reject a degenerate (near-zero-length) request BEFORE extraction
        # — matches OmniAgent's pre-extraction AUDIO_TOO_SHORT check.
        # Without this, a near-zero range produces a wav with 0 decoded
        # PCM samples, which crashes vLLM's Qwen2.5-Omni audio processor
        # with an uncaught, worker-killing ValueError (not just a failed
        # sample — see tools.py::VideoEnv.min_audio_len for detail).
        dur = end - start
        if dur < env.min_audio_len:
            raise ValueError(
                f"AUDIO_TOO_SHORT: {dur:.3f}s < {env.min_audio_len}s"
            )
        audio = env.get_audio(start, end)
        rng = f"[Audio {start:.1f}s-{end:.1f}s]"
        return ToolResult(rng, [audio])


class GetClipTool(BaseTool):
    name = "get_clip"
    description = "Extract a video clip (video+audio tracks)"
    schema_str = '{"type": "get_clip", "start": 5.0, "end": 15.0}'

    def execute(self, action: Dict[str, Any], env: VideoEnv) -> ToolResult:
        limit = ToolRegistry.get_cfg().get("max_clip_len", 60)
        start = float(action.get("start", 0))
        end = min(float(action.get("end", 0)), start + limit)
        # reject a clip too short to contain >=4 frames — matches
        # OmniAgent's video_env.py CLIP_TOO_SHORT pre-extraction check
        # (same rationale as GetAudioTool.min_audio_len: an ultra-short
        # clip can decode to 0 usable frames downstream).
        fps = env.meta().get("fps") or 25.0
        dur = end - start
        if fps >= 2 and dur * fps < 3.999:
            min_dur = 4.0 / float(fps)
            raise ValueError(
                f"CLIP_TOO_SHORT: {dur:.3f}s < {min_dur:.3f}s (need >= 4 frames)"
            )
        clip = env.get_clip(start, end)
        rng = f"[Clip {start:.1f}s-{end:.1f}s]"
        return ToolResult(rng, [clip])


class AnswerTool(BaseTool):
    name = "answer"
    description = "Submit final answer"
    schema_str = '{"type": "answer", "content": "A"}'

    def execute(self, action: Dict[str, Any], env: VideoEnv) -> ToolResult:
        return ToolResult(observation="", meta={"answer": action.get("content", "")})


# ── register built-in tools ───────────────────────────────────────────

ToolRegistry.register(GetFramesTool())
ToolRegistry.register(GetAudioTool())
ToolRegistry.register(GetClipTool())
ToolRegistry.register(AnswerTool())
