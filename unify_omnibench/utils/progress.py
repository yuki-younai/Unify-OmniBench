"""Thread-safe progress bar with running accuracy / failure counter."""
from __future__ import annotations

import threading

from tqdm import tqdm


class ProgressManager:
    def __init__(self, total: int, desc: str = "Eval",
                 initial: int = 0, initial_correct: int = 0, initial_failed: int = 0):
        """``initial``/``initial_correct``/``initial_failed`` let a resumed run
        continue the SAME dataset-wide bar/accuracy instead of restarting at
        0 — e.g. 411 samples already done in a previous invocation should
        show as "411/1000" (not "0/589"), and the accuracy postfix should
        already reflect those 411 samples, not just whatever gets processed
        in this session."""
        self.total = total
        self.lock = threading.Lock()
        self.completed = initial
        self.correct = initial_correct
        self.failed = initial_failed
        self.bar = tqdm(
            total=total,
            initial=initial,
            desc=desc,
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| "
                "{n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
            ),
        )
        self._refresh_postfix()

    def _refresh_postfix(self) -> None:
        denom = self.completed
        acc = (self.correct / denom) if denom else 0.0
        self.bar.set_postfix_str(
            f"Acc:{acc:.1%}({self.correct}/{denom}) Failed:{self.failed}"
        )

    def update(self, is_failed: bool = False, is_correct: bool = False) -> None:
        with self.lock:
            if is_failed:
                self.failed += 1
            else:
                self.completed += 1
                if is_correct:
                    self.correct += 1
            self._refresh_postfix()
            self.bar.update(1)

    def close(self) -> None:
        self.bar.close()

    def __enter__(self) -> "ProgressManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

