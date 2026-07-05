"""Thread-safe progress bar with running accuracy / failure counter."""
from __future__ import annotations

import threading

from tqdm import tqdm


class ProgressManager:
    def __init__(self, total: int, desc: str = "Eval"):
        self.total = total
        self.lock = threading.Lock()
        self.completed = 0
        self.correct = 0
        self.failed = 0
        self.bar = tqdm(
            total=total,
            desc=desc,
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| "
                "{n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
            ),
        )

    def update(self, is_failed: bool = False, is_correct: bool = False) -> None:
        with self.lock:
            if is_failed:
                self.failed += 1
            else:
                self.completed += 1
                if is_correct:
                    self.correct += 1
            denom = self.completed
            acc = (self.correct / denom) if denom else 0.0
            self.bar.set_postfix_str(
                f"Acc:{acc:.1%}({self.correct}/{denom}) Failed:{self.failed}"
            )
            self.bar.update(1)

    def close(self) -> None:
        self.bar.close()

    def __enter__(self) -> "ProgressManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
