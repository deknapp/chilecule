"""Tests for the execution backends and environment detection."""

import pytest

from chilecule.config import EXTERNAL_BINARIES, detect
from chilecule.runners.aws_batch import AWSBatchRunner
from chilecule.runners.base import (
    CHEAP,
    DESCRIPTOR,
    DOCKING_POSE,
    STRUCTURE_PREDICTION,
    ResourceProfile,
    estimate_cost,
)
from chilecule.runners.local import LocalRunner


def double(x: int) -> int:
    return x * 2


def test_local_runner_preserves_input_order():
    runner = LocalRunner(parallel_threshold=4)
    assert runner.map(double, range(50), DESCRIPTOR) == [i * 2 for i in range(50)]


def test_local_runner_runs_small_batches_serially():
    runner = LocalRunner(parallel_threshold=100)
    assert runner.map(double, [1, 2, 3], CHEAP) == [2, 4, 6]


def test_local_runner_handles_empty_input():
    assert LocalRunner().map(double, [], CHEAP) == []


def test_local_runner_refuses_gpu_work():
    """Silently running a GPU profile on CPU would produce a result that looks
    fine and took a hundred times too long."""
    with pytest.raises(RuntimeError, match="GPU"):
        LocalRunner().map(double, [1], STRUCTURE_PREDICTION)


def test_resource_profiles_classify_feasibility():
    assert DESCRIPTOR.laptop_feasible
    assert DOCKING_POSE.laptop_feasible
    assert not STRUCTURE_PREDICTION.laptop_feasible


def test_cost_estimate_scales_with_work():
    small = estimate_cost(10, DOCKING_POSE)
    large = estimate_cost(1000, DOCKING_POSE)
    # Proportional to within the rounding the projection applies for display.
    assert large["cpu_hours"] == pytest.approx(small["cpu_hours"] * 100, rel=0.01)
    assert large["estimated_usd"] > small["estimated_usd"]
    assert estimate_cost(0, DOCKING_POSE)["estimated_usd"] == 0.0


def test_aws_runner_is_honestly_unavailable():
    """The scaffold must never claim it can run. Reporting available() as True
    because credentials exist would tell a user their cloud tier works when the
    backend does not exist."""
    runner = AWSBatchRunner()
    assert runner.available() is False
    assert "scaffold" in runner.describe()["status"]


def test_aws_runner_raises_with_a_cost_projection():
    """The error should say what the call would have cost, so the message is
    useful for planning rather than only for stopping."""
    with pytest.raises(NotImplementedError) as excinfo:
        AWSBatchRunner().map(double, range(100), DOCKING_POSE)
    message = str(excinfo.value)
    assert "not implemented" in message.lower()
    assert "100 items" in message


def test_detect_reports_without_raising():
    env = detect()
    assert env.python_version
    assert set(env.binaries) == set(EXTERNAL_BINARIES)
    # data-and-sar depends only on RDKit, which is a hard dependency.
    assert env.tiers["data-and-sar"] is True
    # The cloud tier is a scaffold and must never report ready.
    assert env.tiers["cloud-burst"] is False


def test_detect_never_reads_credential_values(monkeypatch):
    """Detection reports presence, not secrets."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")
    env = detect()
    assert env.anthropic.available
    assert "sk-ant-secret-value" not in env.anthropic.detail


def test_profile_is_immutable():
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        ResourceProfile(cpus=1).cpus = 4  # type: ignore[misc]
