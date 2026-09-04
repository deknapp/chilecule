"""Execution backends.

Every expensive step in a workflow declares what it needs -- cores, memory,
wall time, whether it wants a GPU -- and hands a callable to a Runner. The
workflow code never knows where the work ran.

The point of the abstraction is that scaling out must not require rewriting
the science. A docking sweep that runs on a laptop over 500 ligands is the
same code that would run over 5 million on a cluster; only the Runner changes.

Two implementations are planned. :class:`~chilecule.runners.local.LocalRunner`
is complete and is what every shipped workflow uses.
:class:`~chilecule.runners.aws_batch.AWSBatchRunner` is deliberately a
scaffold -- the interface is fixed and documented, the implementation is not
written. See docs/ARCHITECTURE.md for why, and for what implementing it would
involve.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class ResourceProfile:
    """What one unit of work needs in order to run.

    ``estimated_seconds`` is per item and is used two ways: to decide whether a
    job is worth shipping to a remote backend at all (a 200 ms task never is),
    and to show the user a projected cost before anything is submitted.
    """

    cpus: int = 1
    memory_gb: float = 2.0
    estimated_seconds: float = 1.0
    gpu: bool = False
    tier: str = "cpu-light"

    @property
    def laptop_feasible(self) -> bool:
        """Whether one item is reasonable to run on a developer machine."""
        return not self.gpu and self.cpus <= 8 and self.memory_gb <= 16


# Profiles for the operations this project performs. Named rather than inlined
# so that a workflow reads as "this step is DOCKING_POSE work" instead of
# carrying magic numbers.
CHEAP = ResourceProfile(cpus=1, memory_gb=0.5, estimated_seconds=0.01, tier="cpu-light")
DESCRIPTOR = ResourceProfile(cpus=1, memory_gb=1.0, estimated_seconds=0.05, tier="cpu-light")
CONFORMER = ResourceProfile(cpus=1, memory_gb=1.0, estimated_seconds=2.0, tier="cpu-medium")
DOCKING_POSE = ResourceProfile(cpus=1, memory_gb=2.0, estimated_seconds=30.0, tier="cpu-heavy")
STRUCTURE_PREDICTION = ResourceProfile(
    cpus=8, memory_gb=32.0, estimated_seconds=600.0, gpu=True, tier="gpu"
)


class Runner(Protocol):
    """Executes work items. Implementations decide where."""

    name: str

    def map(
        self,
        fn: Callable[[T], R],
        items: Iterable[T],
        profile: ResourceProfile,
        **kwargs: Any,
    ) -> list[R]:
        """Apply ``fn`` to every item, returning results in input order."""
        ...

    def submit(self, fn: Callable[..., R], *args: Any, profile: ResourceProfile) -> Future[R]:
        """Schedule a single call, returning a Future."""
        ...

    def available(self) -> bool:
        """Whether this backend can actually run right now."""
        ...

    def describe(self) -> dict[str, Any]:
        """Human-readable backend status, surfaced by ``chilecule doctor``."""
        ...


def estimate_cost(
    n_items: int, profile: ResourceProfile, usd_per_cpu_hour: float = 0.017
) -> dict[str, float]:
    """Project wall time and spend for a batch.

    The default rate is roughly a Fargate Spot vCPU-hour. It exists so that a
    workflow can tell a user "this sweep is about 40 minutes and $1.80" before
    submitting anything, rather than after. Any backend that spends money is
    required to surface this first.
    """
    cpu_seconds = n_items * profile.estimated_seconds * profile.cpus
    return {
        "n_items": n_items,
        "cpu_hours": round(cpu_seconds / 3600, 3),
        "estimated_usd": round((cpu_seconds / 3600) * usd_per_cpu_hour, 2),
        "serial_hours": round(n_items * profile.estimated_seconds / 3600, 2),
    }
