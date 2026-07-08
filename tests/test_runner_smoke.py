"""End-to-end smoke test using the EchoModel and a temp UnifiedAdapter dataset."""
import json
import os
import tempfile

from unify_omnibench.core.registry import build_dataset, build_model
from unify_omnibench.runner import Runner

# trigger registration
import unify_omnibench.datasets  # noqa: F401
import unify_omnibench.models  # noqa: F401


def _make_fake_dataset(tmp) -> str:
    """Unified-JSON fixture (the schema produced by script/convert_*.py) —
    text-only records so no actual media files are needed."""
    records = [
        {
            "id": "daily_omni:0",
            "question": "What sound is heard?",
            "choices": ["A. cat", "B. dog", "C. bird", "D. car"],
            "answer": "A",
            "task_type": "audio",
        },
        {
            "id": "daily_omni:1",
            "question": "Color of the shirt?",
            "choices": ["A. red", "B. blue", "C. green", "D. yellow"],
            "answer": "B",
            "task_type": "visual",
        },
    ]
    data_file = os.path.join(tmp, "data.json")
    with open(data_file, "w") as f:
        json.dump(records, f)
    return data_file


def test_runner_end_to_end_echo():
    with tempfile.TemporaryDirectory() as tmp:
        data_file = _make_fake_dataset(tmp)
        run_dir = os.path.join(tmp, "run")
        cfg = {
            "run_dir": run_dir,
            "modality_mode": "text",
            "dataset": {"name": "daily_omni", "data_file": data_file, "media_root": tmp},
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
        data_file = _make_fake_dataset(tmp)
        run_dir = os.path.join(tmp, "run")
        cfg = {
            "run_dir": run_dir,
            "modality_mode": "text",
            "dataset": {"name": "daily_omni", "data_file": data_file, "media_root": tmp},
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
