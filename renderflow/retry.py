"""Retry with exponential backoff for external calls."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

log = logging.getLogger("renderflow.retry")

P = ParamSpec("P")
T = TypeVar("T")


def retryable(
    attempts: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last: Exception | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last = exc
                    if attempt == attempts:
                        break
                    delay = min(base_delay * 2 ** (attempt - 1), max_delay)
                    delay += random.uniform(0, 1)
                    log.warning(
                        "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                        fn.__qualname__, attempt, attempts, exc, delay,
                    )
                    time.sleep(delay)
            assert last is not None
            raise last

        return wrapper

    return decorator
