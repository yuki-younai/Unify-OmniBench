"""Generic exponential-backoff retry decorator."""
from __future__ import annotations

import functools
import random
import time
from typing import Callable, Tuple, Type


def retry(
    max_retries: int = 4,
    base_delay: float = 4.0,
    jitter: float = 1.0,
    retry_on: Tuple[Type[BaseException], ...] = (Exception,),
    fatal_on: Tuple[Type[BaseException], ...] = (),
) -> Callable:
    """Retry on `retry_on`, never retry on `fatal_on` (re-raise immediately)."""

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrap(*args, **kwargs):
            last_exc: BaseException | None = None
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except fatal_on:
                    raise
                except retry_on as e:  # noqa: PERF203
                    last_exc = e
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt) + random.uniform(0, jitter)
                    time.sleep(delay)
            # unreachable, but for type-checkers
            assert last_exc is not None
            raise last_exc
        return wrap
    return deco
