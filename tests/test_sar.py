"""Tests for SAR analysis, including regression tests for bugs found in development."""

import numpy as np
import pandas as pd
import pytest

from chilecule.tools.sar import (
    activity_cliffs,
    matched_pairs,
    murcko_scaffold,
    scaffold_summary,
    similarity,
    transformation_summary,
)


@pytest.fixture
def halogen_series():
    """A small congeneric series: one scaffold, varying para substituent."""
    return pd.DataFrame(
        {
            "smiles": [
                "c1ccc(-c2ccc(F)cc2)cc1",
                "c1ccc(-c2ccc(Cl)cc2)cc1",
                "c1ccc(-c2ccc(Br)cc2)cc1",
                "c1ccc(-c2ccc(C)cc2)cc1",
                "c1ccc(-c2ccc(OC)cc2)cc1",
                "c1ccc(-c2ccc(O)cc2)cc1",
            ],
            "molecule_chembl_id": [f"CHEMBL_TEST_{i}" for i in range(6)],
            "pchembl": [7.0, 7.5, 8.0, 6.5, 6.0, 5.5],
        }
    )


def test_murcko_scaffold_strips_substituents():
    assert murcko_scaffold("CC(=O)Oc1ccccc1C(=O)O") == "c1ccccc1"
    assert murcko_scaffold("not_a_molecule") is None


def test_generic_scaffold_collapses_heteroatoms():
    """A pyrimidine and a benzene share a carbon skeleton."""
    assert murcko_scaffold("c1ccncc1", generic=True) == murcko_scaffold(
        "c1ccccc1", generic=True
    )


def test_similarity_bounds():
    assert similarity("CCO", "CCO") == pytest.approx(1.0)
    assert 0.0 <= similarity("CCO", "c1ccccc1C(=O)NC2CCCCC2") <= 1.0
    assert similarity("CCO", "garbage!!") is None


def test_matched_pairs_are_directionally_symmetric(halogen_series):
    """REGRESSION: transformation deltas must not depend on input row order.

    matched_pairs() originally emitted one direction per pair, taken in the
    order rows appeared in the frame. Because curate_activities() returns a
    potency-sorted frame, every delta came out negative and the transformation
    table claimed that every substitution ever made reduced potency.

    Emitting both directions makes the set of medians symmetric about zero.
    """
    pairs = matched_pairs(halogen_series)
    assert pairs, "expected matched pairs from a congeneric series"

    deltas = np.array([p.delta for p in pairs])
    assert deltas.sum() == pytest.approx(0.0, abs=1e-9)
    assert (deltas > 0).any() and (deltas < 0).any()


def test_matched_pairs_are_invariant_to_row_order(halogen_series):
    """The same conclusions must follow from a shuffled input."""
    forward = transformation_summary(matched_pairs(halogen_series), min_occurrences=1)
    reversed_frame = halogen_series.iloc[::-1].reset_index(drop=True)
    backward = transformation_summary(matched_pairs(reversed_frame), min_occurrences=1)

    forward_map = dict(zip(forward["transformation"], forward["median_delta"]))
    backward_map = dict(zip(backward["transformation"], backward["median_delta"]))
    assert set(forward_map) == set(backward_map)
    for key, value in forward_map.items():
        assert backward_map[key] == pytest.approx(value)


def test_transformation_summary_requires_repeat_observations(halogen_series):
    pairs = matched_pairs(halogen_series)
    assert transformation_summary(pairs, min_occurrences=99).empty


def test_activity_cliffs_finds_similar_pairs_with_divergent_potency():
    frame = pd.DataFrame(
        {
            "smiles": [
                "c1ccc(-c2ccc(Cl)cc2)cc1",
                "c1ccc(-c2ccc(Br)cc2)cc1",
                "CCCCCCCCCCO",
            ],
            "molecule_chembl_id": ["A", "B", "C"],
            "pchembl": [9.0, 6.0, 7.0],
        }
    )
    cliffs = activity_cliffs(frame, similarity_threshold=0.5, activity_threshold=1.0)
    assert len(cliffs) == 1
    row = cliffs.iloc[0]
    assert {row["id_a"], row["id_b"]} == {"A", "B"}
    assert row["delta_activity"] == pytest.approx(3.0)
    assert row["sali"] > 0


def test_activity_cliffs_on_smooth_data_returns_empty(halogen_series):
    smooth = halogen_series.copy()
    smooth["pchembl"] = 7.0
    assert activity_cliffs(smooth).empty


def test_scaffold_summary_fractions_sum_to_one(halogen_series):
    summary = scaffold_summary(halogen_series)
    assert summary["fraction_of_set"].sum() == pytest.approx(1.0)
    assert summary["n_compounds"].sum() == len(halogen_series)


def test_empty_input_is_handled_everywhere():
    empty = pd.DataFrame(columns=["smiles", "molecule_chembl_id", "pchembl"])
    assert matched_pairs(empty) == []
    assert scaffold_summary(empty).empty
    assert activity_cliffs(empty).empty
