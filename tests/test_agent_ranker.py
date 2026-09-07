"""Tests for the agent's side of the retrospective benchmark.

None of these call an API. Everything that decides whether the agent's score is
counted, discarded, or sunk to the bottom is pure and is tested here, because
those rules are what stop the benchmark from flattering the agent.
"""

from __future__ import annotations

import numpy as np
import pytest

from chilecule.bench.agent_ranker import (
    UNSCORED,
    AgentRanking,
    build_ranking_prompt,
    candidate_ids,
    parse_agent_scores,
    scores_to_array,
    shuffled_batches,
)


def test_candidate_ids_are_fixed_width_and_unique():
    ids = candidate_ids(12)
    assert ids[0] == "C0001"
    assert ids[-1] == "C0012"
    assert len(set(ids)) == 12
    assert len({len(i) for i in ids}) == 1


def test_candidate_ids_widen_past_four_digits():
    ids = candidate_ids(12345)
    assert ids[0] == "C00001"
    assert ids[-1] == "C12345"


def test_prompt_contains_every_candidate_and_active():
    candidates = {"C0001": "CCO", "C0002": "c1ccccc1"}
    actives = ["CC(=O)O", "CCN"]

    prompt = build_ranking_prompt(candidates, actives, target="EGFR")

    for smiles in list(candidates.values()) + actives:
        assert smiles in prompt
    for cid in candidates:
        assert cid in prompt
    assert "EGFR" in prompt


def test_prompt_omits_labels_entirely():
    """The agent must not be able to read activity off the prompt."""
    prompt = build_ranking_prompt({"C0001": "CCO"}, ["CCN"])
    lowered = prompt.lower()
    assert "label" not in lowered
    assert "inactive" not in lowered


def test_parse_plain_lines():
    text = "C0001: 90\nC0002: 10\n"
    scores, malformed = parse_agent_scores(text, {"C0001", "C0002"})
    assert scores == {"C0001": 90.0, "C0002": 10.0}
    assert malformed == 0


@pytest.mark.parametrize(
    "line",
    [
        "C0001: 55",
        "  C0001 : 55  ",
        "- C0001: 55",
        "* **C0001**: 55",
        "C0001 = 55",
        "C0001: 55.",
        "C0001: 55.0",
        "> C0001: 55",
    ],
)
def test_parse_tolerates_formatting_the_model_adds(line):
    scores, malformed = parse_agent_scores(line, {"C0001"})
    assert scores == {"C0001": 55.0}
    assert malformed == 0


def test_prose_around_scores_is_ignored():
    text = (
        "Here are my scores based on scaffold similarity.\n\n"
        "C0001: 88\n"
        "C0002: 12\n\n"
        "The second compound lacks the hinge-binding motif."
    )
    scores, malformed = parse_agent_scores(text, {"C0001", "C0002"})
    assert scores == {"C0001": 88.0, "C0002": 12.0}
    assert malformed == 0


def test_unknown_candidate_id_counts_as_malformed():
    """An invented ID is a failure the ranking metrics cannot see."""
    text = "C0001: 50\nC9999: 70\n"
    scores, malformed = parse_agent_scores(text, {"C0001"})
    assert scores == {"C0001": 50.0}
    assert malformed == 1


@pytest.mark.parametrize("bad", ["-5", "101", "1000"])
def test_out_of_range_scores_are_rejected(bad):
    scores, malformed = parse_agent_scores(f"C0001: {bad}", {"C0001"})
    assert scores == {}
    assert malformed == 1


def test_boundary_scores_are_accepted():
    scores, _ = parse_agent_scores("C0001: 0\nC0002: 100", {"C0001", "C0002"})
    assert scores == {"C0001": 0.0, "C0002": 100.0}


def test_repeated_id_keeps_the_first_value():
    scores, malformed = parse_agent_scores("C0001: 30\nC0001: 80", {"C0001"})
    assert scores == {"C0001": 30.0}
    assert malformed == 0


def test_unscored_candidates_sink_below_every_real_score():
    ranking = AgentRanking(scores={"C0002": 0.0}, n_requested=3)
    array = scores_to_array(ranking, ["C0001", "C0002", "C0003"])

    assert array[1] == 0.0
    assert array[0] == UNSCORED
    assert array[2] == UNSCORED
    # A genuine zero must still outrank an omission.
    assert array[1] > array[0]


def test_scores_align_to_requested_order_not_agent_order():
    ranking = AgentRanking(scores={"C0003": 10.0, "C0001": 90.0}, n_requested=3)
    array = scores_to_array(ranking, ["C0001", "C0002", "C0003"])
    np.testing.assert_array_equal(array, [90.0, UNSCORED, 10.0])


def test_coverage_reports_the_fraction_actually_scored():
    assert AgentRanking(scores={"a": 1.0}, n_requested=4).coverage == 0.25
    assert AgentRanking(n_requested=0).coverage == 0.0


def test_batches_partition_the_candidates_exactly_once():
    ids = candidate_ids(95)
    smiles_by_id = {cid: "CCO" for cid in ids}

    batches = shuffled_batches(ids, smiles_by_id, batch_size=40, seed=0)

    assert [len(b) for b in batches] == [40, 40, 15]
    seen = [cid for batch in batches for cid in batch]
    assert sorted(seen) == sorted(ids)


def test_batching_is_shuffled_but_reproducible():
    ids = candidate_ids(50)
    smiles_by_id = {cid: "CCO" for cid in ids}

    first = shuffled_batches(ids, smiles_by_id, batch_size=10, seed=7)
    again = shuffled_batches(ids, smiles_by_id, batch_size=10, seed=7)
    different_seed = shuffled_batches(ids, smiles_by_id, batch_size=10, seed=8)

    assert [list(b) for b in first] == [list(b) for b in again]
    assert [list(b) for b in first] != [list(b) for b in different_seed]


def test_batching_rejects_a_nonsense_size():
    with pytest.raises(ValueError):
        shuffled_batches(["C0001"], {"C0001": "CCO"}, batch_size=0, seed=0)
