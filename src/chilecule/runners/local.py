"""Local execution: threads for I/O-bound work, processes for CPU-bound work."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

from .base import ResourceProfile, Runner


class LocalRunner(Runner):
    """Runs work on this machine.

    Chooses a process pool for CPU-bound profiles and a thread pool otherwise.
    RDKit releases the GIL for much of its C++ work, but Python-level
    per-molecule loops do not, so anything at ``cpu-heavy`` goes to processes.

    Serial execution below ``parallel_threshold`` items is deliberate: pool
    startup costs more than the work for small batches, and most interactive
    calls are small batches.
    """

    name = "local"

    def __init__(self, max_workers: int | None = None, parallel_threshold: int = 32) -> None:
        self.max_workers = max_workers or max(1, (os.cpu_count() or 2) - 1)
        self.parallel_threshold = parallel_threshold

    def available(self) -> bool:
        return True

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "available": True,
            "max_workers": self.max_workers,
            "cpu_count": os.cpu_count(),
            "supports_gpu": False,
        }

    def map(
        self,
        fn: Callable,
        items: Iterable,
        profile: ResourceProfile,
        **kwargs: Any,
    ) -> list:
        items = list(items)
        if not items:
            return []

        if profile.gpu:
            raise RuntimeError(
                "LocalRunner cannot satisfy a GPU profile. This step needs a GPU "
                "backend; see docs/ARCHITECTURE.md for the cloud runner status."
            )

        if len(items) < self.parallel_threshold:
            return [fn(item) for item in items]

        executor_cls = ProcessPoolExecutor if profile.tier == "cpu-heavy" else ThreadPoolExecutor
        workers = min(self.max_workers, len(items), max(1, self.max_workers // profile.cpus))
        with executor_cls(max_workers=workers) as pool:
            return list(pool.map(fn, items))

    def submit(self, fn: Callable, *args: Any, profile: ResourceProfile) -> Future:
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(fn, *args)
        future.add_done_callback(lambda _: pool.shutdown(wait=False))
        return future
