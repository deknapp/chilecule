"""Parallel map over molecules.

One function, because that is all this project needs. Every workflow here is
designed to finish on a laptop CPU, so there is no execution backend to choose
between and no abstraction worth building over a single implementation.

The only judgement encoded here is thread pool versus process pool. RDKit
releases the GIL for much of its C++ work, so I/O-shaped and RDKit-heavy work
parallelizes fine on threads. Python-level per-molecule loops do not, which is
what ``cpu_bound`` selects for.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")

# Below this many items, pool startup costs more than the work it distributes,
# and most interactive calls are small batches.
PARALLEL_THRESHOLD = 32


def pmap(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    cpu_bound: bool = False,
    max_workers: int | None = None,
    threshold: int = PARALLEL_THRESHOLD,
) -> list[R]:
    """Apply ``fn`` to every item, returning results in input order."""
    items = list(items)
    if not items:
        return []
    if len(items) < threshold:
        return [fn(item) for item in items]

    workers = max_workers or max(1, (os.cpu_count() or 2) - 1)
    executor_cls = ProcessPoolExecutor if cpu_bound else ThreadPoolExecutor
    with executor_cls(max_workers=min(workers, len(items))) as pool:
        return list(pool.map(fn, items))
