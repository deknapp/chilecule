"""Tests for early-recognition metrics.

Each metric is checked against a value that is known analytically, not against
a value produced by a previous run of this code.
"""

import numpy as np
import pytest

from chilecule.bench.metrics import (
    bedroc,
    enrichment_factor,
    evaluate,
    max_enrichment_factor,
    roc_auc,
)


@pytest.fixture
def labels():
    y = np.zeros(1000, dtype=int)
    y[:50] = 1
    return y


@pytest.fixture
def perfect_scores():
    return np.concatenate([np.linspace(1.0, 0.9, 50), np.linspace(0.5, 0.0, 950)])


@pytest.fixture
def worst_scores():
    return np.concatenate([np.linspace(0.1, 0.0, 50), np.linspace(1.0, 0.2, 950)])


def test_perfect_ranking_saturates_every_metric(labels, perfect_scores):
    metrics = evaluate(perfect_scores, labels)
    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.bedroc_alpha20 == pytest.approx(1.0, abs=1e-6)


def test_worst_ranking_bottoms_out(labels, worst_scores):
    metrics = evaluate(worst_scores, labels)
    assert metrics.roc_auc == pytest.approx(0.0)
    assert metrics.bedroc_alpha20 == pytest.approx(0.0, abs=1e-6)


def test_constant_scores_give_exactly_half(labels):
    """A scoring function that returns the same value for everything is
    uninformative and must score 0.5 -- not 1.0, which is what naive tie
    handling produces when actives happen to sort first."""
    assert roc_auc(np.ones(len(labels)), labels) == pytest.approx(0.5)


def test_ties_are_broken_pessimistically(labels):
    """An active tied with inactives is ranked after them.

    Optimistic tie-breaking manufactures enrichment out of nothing, and
    scoring functions produce ties constantly.
    """
    scores = np.ones(1000)
    scores[:50] = 1.0  # actives tied with everything else
    assert enrichment_factor(scores, labels, 0.01) == pytest.approx(0.0)


def test_enrichment_respects_its_ceiling(labels, perfect_scores):
    ceiling = max_enrichment_factor(1000, 50, 0.01)
    assert enrichment_factor(perfect_scores, labels, 0.01) == pytest.approx(ceiling)
    assert ceiling == pytest.approx(20.0)


def test_random_ranking_is_near_chance(labels):
    rng = np.random.default_rng(0)
    aucs = [roc_auc(rng.random(len(labels)), labels) for _ in range(50)]
    assert np.mean(aucs) == pytest.approx(0.5, abs=0.02)


def test_bedroc_weights_the_top_of_the_list(labels):
    """Two rankings with identical ROC-AUC must be distinguished by BEDROC if
    one puts its actives earlier. This is the entire reason BEDROC exists."""
    early = np.zeros(1000)
    early[:50] = np.linspace(1.0, 0.99, 50)   # actives at ranks 1-50
    late = np.zeros(1000)
    late[:50] = np.linspace(0.5, 0.49, 50)    # actives after 500 inactives
    late[50:550] = np.linspace(1.0, 0.6, 500)

    assert bedroc(early, labels) > bedroc(late, labels)


def test_metrics_handle_degenerate_sets():
    all_active = np.ones(10, dtype=int)
    scores = np.arange(10, dtype=float)
    assert np.isnan(roc_auc(scores, all_active))
    assert np.isnan(bedroc(scores, all_active))
