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
        """Run ffmpeg, return output path."""
        self.step_counter += 1
        out = os.path.join(self.tmp_dir, f"step_{self.step_counter:02d}")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + list(args)
        subprocess.run(cmd, check=True, timeout=120)
        return out

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
        dur = end - start
        out = self._ffmpeg(
            "-ss", f"{start:.3f}", "-i", self.video_path,
            "-t", f"{dur:.3f}", "-map", "0:a:0?",
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            "audio.wav",
        )
        wav = os.path.join(os.path.dirname(out), "audio.wav")
        return MediaRef(kind="audio", path=wav)

    def get_clip(self, start: float, end: float) -> MediaRef:
        dur = end - start
        out = self._ffmpeg(
            "-ss", f"{start:.3f}", "-i", self.video_path,
            "-t", f"{dur:.3f}", "-map", "0:v:0?", "-map", "0:a:0?",
            "-c", "copy", "-avoid_negative_ts", "make_zero", "clip.mp4",
        )
        mp4 = os.path.join(os.path.dirname(out), "clip.mp4")
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
        num = min(int(action.get("num", 10)), limit)
        frames = env.get_frames(
            float(action.get("start", 0)), float(action.get("end", 0)), num,
        )
        rng = f"[Frames {action['start']:.1f}s-{action['end']:.1f}s (num={len(frames)})]"
        return ToolResult(rng, frames)


class GetAudioTool(BaseTool):
    name = "get_audio"
    description = "Extract audio from a time range"
    schema_str = '{"type": "get_audio", "start": 0.0, "end": 15.0}'

    def execute(self, action: Dict[str, Any], env: VideoEnv) -> ToolResult:
        limit = ToolRegistry.get_cfg().get("max_audio_len", 300)
        start = float(action.get("start", 0))
        end = min(float(action.get("end", 0)), start + limit)
        audio = env.get_audio(start, end)
        rng = f"[Audio {action['start']:.1f}s-{action['end']:.1f}s]"
        return ToolResult(rng, [audio])


class GetClipTool(BaseTool):
    name = "get_clip"
    description = "Extract a video clip (video+audio tracks)"
    schema_str = '{"type": "get_clip", "start": 5.0, "end": 15.0}'

    def execute(self, action: Dict[str, Any], env: VideoEnv) -> ToolResult:
        limit = ToolRegistry.get_cfg().get("max_clip_len", 60)
        start = float(action.get("start", 0))
        end = min(float(action.get("end", 0)), start + limit)
        clip = env.get_clip(start, end)
        rng = f"[Clip {action['start']:.1f}s-{action['end']:.1f}s]"
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
