"""Quick adapter sanity tests using on-the-fly fixtures."""
import json
import os
import tempfile

from unify_omnibench.core.registry import build_dataset
import unify_omnibench.datasets  # noqa: F401


def test_daily_omni_adapter():
    with tempfile.TemporaryDirectory() as tmp:
        qa = [{
            "video_id": "vidX",
            "Question": "Q?",
            "Choice": ["A. a", "B. b", "C. c", "D. d"],
            "Answer": "C",
            "Type": "audio",
            "video_category": "music",
            "video_duration": "<30s",
        }]
        p = os.path.join(tmp, "qa.json")
        with open(p, "w") as f:
            json.dump(qa, f)
        ds = build_dataset({"name": "daily_omni", "qa_file": p, "video_base_dir": tmp, "require_audio": False})
        assert len(ds) == 1
        s = next(iter(ds))
        assert s.dataset == "daily_omni"
        assert s.answer == "C"
        assert s.meta["video_id"] == "vidX"
        assert any(m.kind == "video" for m in s.media)


def test_omnibench_adapter():
    with tempfile.TemporaryDirectory() as tmp:
        rec = {
            "index": 7,
            "question": "What is heard?",
            "options": ["A. cat", "B. dog", "C. bird", "D. car"],
            "answer": "B",
            "image_path": "img/1.jpg",
            "audio_path": "aud/1.wav",
            "task type": "av_match",
            "audio type": "speech",
        }
        p = os.path.join(tmp, "data.jsonl")
        with open(p, "w") as f:
            f.write(json.dumps(rec) + "\n")
        ds = build_dataset({"name": "omnibench", "data_file": p, "mm_root": tmp})
        s = next(iter(ds))
        assert s.answer == "B"
        kinds = sorted(m.kind for m in s.media)
        assert kinds == ["audio", "image"]
        assert s.meta["task_type"] == "av_match"


def test_omnivideobench_adapter():
    with tempfile.TemporaryDirectory() as tmp:
        data = [{
            "video": "v1",
            "video_type": "lecture",
            "duration": "02:30",
            "questions": [
                {"question": "Q1?", "options": ["A","B","C","D"], "correct_option": "A", "question_type": "t1"},
                {"question": "Q2?", "options": ["A","B","C","D"], "correct_option": "C", "question_type": "t2"},
            ],
        }]
        p = os.path.join(tmp, "d.json")
        with open(p, "w") as f:
            json.dump(data, f)
        ds = build_dataset({"name": "omnivideobench", "data_file": p, "video_dir": tmp})
        assert len(ds) == 2
        s = next(iter(ds))
        assert s.meta["duration_s"] == 150
        assert s.media[0].kind == "video"
