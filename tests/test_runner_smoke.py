"""End-to-end smoke test using the EchoModel and a temp Daily-Omni-like dataset."""
import json
import os
import tempfile

from unify_omnibench.core.registry import build_dataset, build_model
from unify_omnibench.runner import Runner

# trigger registration
import unify_omnibench.datasets  # noqa: F401
import unify_omnibench.models  # noqa: F401


def _make_fake_daily_omni(tmp):
    qa = [
        {
            "video_id": "vid001",
            "Question": "What sound is heard?",
            "Choice": ["A. cat", "B. dog", "C. bird", "D. car"],
            "Answer": "A",
            "Type": "audio",
            "video_category": "indoor",
            "video_duration": "<30s",
        },
        {
            "video_id": "vid002",
            "Question": "Color of the shirt?",
            "Choice": ["A. red", "B. blue", "C. green", "D. yellow"],
            "Answer": "B",
            "Type": "visual",
            "video_category": "outdoor",
            "video_duration": "<60s",
        },
    ]
    qa_path = os.path.join(tmp, "qa.json")
    with open(qa_path, "w") as f:
        json.dump(qa, f)
    return qa_path


def test_runner_end_to_end_echo():
    with tempfile.TemporaryDirectory() as tmp:
        qa_path = _make_fake_daily_omni(tmp)
        run_dir = os.path.join(tmp, "run")
        cfg = {
            "run_dir": run_dir,
            "modality_mode": "text",
            "dataset": {
                "name": "daily_omni",
                "qa_file": qa_path,
                "video_base_dir": tmp,
                "require_audio": False,
            },
            "model": {"name": "echo", "fixed_answer": "A"},
            "generation": {},
            "concurrency": {"mode": "thread", "max_workers": 2},
        }
        ds = build_dataset(cfg["dataset"])
        md = build_model(cfg["model"])
        summary = Runner(ds, md, cfg).run()

        # Echo always answers A -> sample1 (gold A) correct, sample2 (gold B) wrong
        assert summary["total"] == 2
        assert summary["valid"] == 2
        assert summary["correct"] == 1
        assert abs(summary["accuracy"] - 0.5) < 1e-6
        assert os.path.exists(os.path.join(run_dir, "items.jsonl"))
        assert os.path.exists(os.path.join(run_dir, "summary.json"))
        assert os.path.exists(os.path.join(run_dir, "summary.md"))


def test_runner_resume_skips_done():
    with tempfile.TemporaryDirectory() as tmp:
        qa_path = _make_fake_daily_omni(tmp)
        run_dir = os.path.join(tmp, "run")
        cfg = {
            "run_dir": run_dir,
            "modality_mode": "text",
            "dataset": {
                "name": "daily_omni", "qa_file": qa_path,
                "video_base_dir": tmp, "require_audio": False,
            },
            "model": {"name": "echo", "fixed_answer": "A"},
            "generation": {},
            "concurrency": {"mode": "sequential"},
        }
        ds = build_dataset(cfg["dataset"])
        md = build_model(cfg["model"])
        Runner(ds, md, cfg).run()

        # second run should not append duplicates
        ds2 = build_dataset(cfg["dataset"])
        md2 = build_model(cfg["model"])
        s2 = Runner(ds2, md2, cfg).run()
        with open(os.path.join(run_dir, "items.jsonl")) as f:
            n_lines = sum(1 for _ in f)
        assert n_lines == 2, f"expected 2 lines after resume, got {n_lines}"
        assert s2["total"] == 2
