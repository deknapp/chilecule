"""AWS Batch runner -- SCAFFOLD, NOT IMPLEMENTED.

This file exists to fix the interface and to document exactly what building
the cloud tier involves. It is not a stub left behind by accident: nothing in
the shipped workflows needs it, because every shipped workflow is designed to
finish on a laptop CPU.

What implementing this would require, in order:

1. **A container image.** Built from the project's own ``environment.yml`` so
   that the cloud environment is the local environment. Two dependency
   specifications drift apart within a month; one does not.

2. **Infrastructure, as code.** An S3 bucket for inputs and results, an ECR
   repository, a Batch compute environment on Spot, a job queue, and a job
   definition per resource tier. One CloudFormation stack, created by
   ``chilecule cloud init`` and destroyed by ``chilecule cloud destroy``. A
   portfolio repository that leaves billable infrastructure running is worse
   than one with no cloud tier at all.

3. **A cost gate.** :func:`~chilecule.runners.base.estimate_cost` runs before
   submission and the projection is shown to the user. Above a threshold, the
   run requires confirmation. Spot instances by default, with a hard budget
   cap on the compute environment.

4. **Result addressing.** Content-addressed S3 keys derived from a hash of
   (input, tool version, parameters), so a re-run of an unchanged job is a
   cache lookup rather than a second charge.

Credentials are deliberately not handled here. Boto3's own resolution chain --
environment, shared config profiles, SSO, instance roles -- already does this
correctly, and a project that invents its own credential file is a project
that will eventually leak one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import Future
from typing import Any

from .base import ResourceProfile, Runner, estimate_cost

NOT_IMPLEMENTED_MESSAGE = (
    "AWSBatchRunner is a scaffold and is not implemented.\n"
    "The Runner interface it satisfies is stable; the backend is not written.\n"
    "Use LocalRunner -- every workflow shipped with chilecule is designed to\n"
    "complete on laptop CPU. See docs/ARCHITECTURE.md for the implementation plan."
)


class AWSBatchRunner(Runner):
    """Planned AWS Batch backend. Every execution method raises."""

    name = "aws-batch"

    def __init__(
        self,
        job_queue: str = "chilecule-queue",
        region: str | None = None,
        bucket: str | None = None,
        max_usd: float = 25.0,
    ) -> None:
        self.job_queue = job_queue
        self.region = region
        self.bucket = bucket
        self.max_usd = max_usd

    def available(self) -> bool:
        """Always False. Credential presence is reported separately.

        ``chilecule doctor`` distinguishes "you have AWS credentials" from
        "the cloud tier works", because conflating them would tell a user
        their setup is fine when the backend does not exist.
        """
        return False

    def describe(self) -> dict[str, Any]:
        from ..config import aws_credentials_present

        return {
            "backend": self.name,
            "available": False,
            "status": "scaffold -- interface defined, implementation not written",
            "aws_credentials_detected": aws_credentials_present(),
            "supports_gpu": True,
            "plan": "docs/ARCHITECTURE.md",
        }

    def map(self, fn: Callable, items: Iterable, profile: ResourceProfile, **kwargs: Any) -> list:
        items = list(items)
        projection = estimate_cost(len(items), profile)
        raise NotImplementedError(
            f"{NOT_IMPLEMENTED_MESSAGE}\n\n"
            f"This call would have submitted {projection['n_items']} items "
            f"({projection['cpu_hours']} CPU-hours, ~${projection['estimated_usd']})."
        )

    def submit(self, fn: Callable, *args: Any, profile: ResourceProfile) -> Future:
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)
