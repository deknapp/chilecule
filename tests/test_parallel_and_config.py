"""Tests for the parallel helper and environment detection."""

import pytest

from chilecule.config import EXTERNAL_BINARIES, detect
from chilecule.parallel import pmap


def double(x: int) -> int:
    return x * 2


def test_pmap_preserves_input_order():
    assert pmap(double, range(100), threshold=4) == [i * 2 for i in range(100)]


def test_pmap_runs_small_batches_serially():
    assert pmap(double, [1, 2, 3], threshold=100) == [2, 4, 6]


def test_pmap_handles_empty_input():
    assert pmap(double, []) == []


def test_pmap_process_pool_path():
    assert pmap(double, range(40), cpu_bound=True, threshold=4) == [i * 2 for i in range(40)]


def test_detect_reports_without_raising():
    env = detect()
    assert env.python_version
    assert set(env.binaries) == set(EXTERNAL_BINARIES)
    # Cheminformatics depends only on RDKit, which is a hard dependency.
    assert env.tiers["cheminformatics"] is True


def test_detect_never_reads_credential_values(monkeypatch):
    """Detection reports presence, not secrets."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")
    env = detect()
    assert env.anthropic.available
    assert "sk-ant-secret-value" not in env.anthropic.detail


def test_no_cloud_tier_is_advertised():
    """The project is laptop-scoped. Nothing should imply a cloud backend."""
    assert "cloud-burst" not in detect().tiers


@pytest.mark.parametrize("binary", ["smina", "fpocket"])
def test_external_binaries_are_documented_with_licenses(binary):
    assert binary in EXTERNAL_BINARIES
    assert any(tag in EXTERNAL_BINARIES[binary] for tag in ("MIT", "Apache", "GPL"))
