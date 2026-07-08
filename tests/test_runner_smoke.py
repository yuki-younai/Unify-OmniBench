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


def test_runner_auto_retries_failed_and_compacts_items():
    """A uid that errored on a previous (possibly interrupted) run must be
    automatically re-attempted on the next plain run() call — no separate
    --rerun-failed step needed — and its stale failed record must be
    replaced (not left alongside the new one), so summary stats aren't
    double-counted."""
    with tempfile.TemporaryDirectory() as tmp:
        data_file = _make_fake_dataset(tmp)
        run_dir = os.path.join(tmp, "run")
        os.makedirs(run_dir, exist_ok=True)

        # Simulate a previous run where sample 0 errored out.
        stale_failed = {
            "uid": "daily_omni:0", "dataset": "daily_omni",
            "question": "What sound is heard?", "choices": [],
            "raw_output": "", "parsed_answer": None, "correct_answer": "A",
            "is_correct": False, "latency_s": 0.1,
            "error": "RuntimeError: simulated failure", "meta": {},
        }
        items_path = os.path.join(run_dir, "items.jsonl")
        with open(items_path, "w") as f:
            f.write(json.dumps(stale_failed) + "\n")
        failed_path = os.path.join(run_dir, "failed.jsonl")
        with open(failed_path, "w") as f:
            f.write(json.dumps(stale_failed) + "\n")

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
        summary = Runner(ds, md, cfg).run()

        # uid "daily_omni:0" must have been retried (echo always succeeds),
        # and its stale error record must be gone — not duplicated.
        with open(items_path) as f:
            recs = [json.loads(l) for l in f if l.strip()]
        assert len(recs) == 2, f"expected 2 deduped records, got {len(recs)}"
        rec0 = next(r for r in recs if r["uid"] == "daily_omni:0")
        assert rec0["error"] is None
        assert rec0["is_correct"] is True

        # failed.jsonl must be removed entirely — no uid is failing anymore,
        # so an empty leftover file should not linger.
        assert not os.path.exists(failed_path), "failed.jsonl should be removed once nothing fails"

        assert summary["total"] == 2, "stale duplicate must not inflate total"
        assert summary["failed"] == 0
