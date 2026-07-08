"""Adapter sanity tests for UnifiedAdapter — the only dataset adapter
actually registered (see unify_omnibench/datasets/unified.py; it backs
omnibench/daily_omni/omnivideobench/worldsense, all now sharing the same
converted-JSON schema produced by script/convert_*.py)."""
import json
import os
import tempfile

from unify_omnibench.core.registry import build_dataset
import unify_omnibench.datasets  # noqa: F401


def _write_media(path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\x00")
    return path


def test_unified_adapter_video_audio():
    with tempfile.TemporaryDirectory() as tmp:
        vp = _write_media(os.path.join(tmp, "media", "video", "v1.mp4"))
        ap = _write_media(os.path.join(tmp, "media", "audio", "v1.wav"))
        records = [{
            "id": "daily_omni:0",
            "question": "What sound is heard?",
            "choices": ["A. cat", "B. dog", "C. bird", "D. car"],
            "answer": "A",
            "video_path": "media/video/v1.mp4",
            "audio_path": "media/audio/v1.wav",
            "task_type": "audio",
            "category": "indoor",
            "duration": "<30s",
        }]
        data_file = os.path.join(tmp, "data.json")
        with open(data_file, "w") as f:
            json.dump(records, f)

        ds = build_dataset({"name": "daily_omni", "data_file": data_file, "media_root": tmp})
        assert len(ds) == 1
        s = next(iter(ds))
        assert s.dataset == "daily_omni"
        assert s.answer == "A"
        assert s.meta["task_type"] == "audio"
        kinds = sorted(m.kind for m in s.media)
        assert kinds == ["audio", "video"]


def test_unified_adapter_image_audio():
    with tempfile.TemporaryDirectory() as tmp:
        ip = _write_media(os.path.join(tmp, "media", "image", "1.jpg"))
        ap = _write_media(os.path.join(tmp, "media", "audio", "1.wav"))
        records = [{
            "id": "omnibench:7",
            "question": "What is heard?",
            "choices": ["A. cat", "B. dog", "C. bird", "D. car"],
            "answer": "B",
            "image_path": "media/image/1.jpg",
            "audio_path": "media/audio/1.wav",
            "task_type": "av_match",
        }]
        data_file = os.path.join(tmp, "data.json")
        with open(data_file, "w") as f:
            json.dump(records, f)

        ds = build_dataset({"name": "omnibench", "data_file": data_file, "media_root": tmp})
        s = next(iter(ds))
        assert s.answer == "B"
        kinds = sorted(m.kind for m in s.media)
        assert kinds == ["audio", "image"]
        assert s.meta["task_type"] == "av_match"


def test_unified_adapter_video_only_and_missing_media():
    with tempfile.TemporaryDirectory() as tmp:
        vp = _write_media(os.path.join(tmp, "media", "video", "v2.mp4"))
        records = [
            {
                "id": "omnivideobench:0",
                "question": "Q1?", "choices": ["A", "B", "C", "D"], "answer": "A",
                "video_path": "media/video/v2.mp4", "task_type": "t1",
            },
            {
                "id": "omnivideobench:1",
                "question": "Q2?", "choices": ["A", "B", "C", "D"], "answer": "C",
                "video_path": "media/video/missing.mp4",  # 不存在 -> media 应为空
                "task_type": "t2",
            },
        ]
        data_file = os.path.join(tmp, "data.json")
        with open(data_file, "w") as f:
            json.dump(records, f)

        ds = build_dataset({"name": "omnivideobench", "data_file": data_file, "media_root": tmp})
        assert len(ds) == 2
        samples = list(ds)
        assert samples[0].media[0].kind == "video"
        assert samples[1].media == []  # 文件不存在时 UnifiedAdapter 会跳过
