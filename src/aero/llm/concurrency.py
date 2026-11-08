"""Parallel LLM calls. vLLM batches concurrent requests, so N parallel calls cost about one call's latency."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

A = TypeVar("A")
R = TypeVar("R")


def pmap(fn: Callable[[A], R], items: Sequence[A], max_workers: int = 8) -> list[R]:
    """Order-preserving parallel map. The first exception propagates."""
    if len(items) <= 1 or max_workers <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        return list(pool.map(fn, items))
