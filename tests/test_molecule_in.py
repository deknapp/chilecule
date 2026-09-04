"""Tests for the molecule-in tools: scoring, synthesis, design, R-groups."""

import pandas as pd
import pytest

from chilecule.tools.design import Transformation, apply_transformation, implausibility
from chilecule.tools.rgroup import decompose, free_wilson, substituent_table
from chilecule.tools.score import PROFILES, Criterion, score
from chilecule.tools.synth import assess

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ERLOTINIB = "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1"
GREASE = "CCCCCCCCCCCCCCCCc1ccc(-c2ccc(-c3ccccc3)cc2)cc1"


# ------------------------------------------------------------------ scorecards


def test_scorecard_names_the_limiting_property():
    """A bare score is not actionable; the limiting property is."""
    result = score(GREASE)
    assert result is not None
    assert result.limiting["property"] == "clogp"
    assert "clogp" in result.verdict


def test_scorecard_penalises_greasy_compounds():
    assert score(GREASE).score < score(ERLOTINIB).score


def test_no_limiting_property_when_everything_is_in_range():
    """A compound inside every window should report no limiting property."""
    criterion = Criterion("mw", "window", 100, 1000, 0, 2000)
    result = score(ASPIRIN, criteria=(criterion,))
    assert result.limiting is None
    assert "inside its target window" in result.verdict


def test_desirability_is_bounded():
    for criteria in PROFILES.values():
        for criterion in criteria:
            for value in (-1000, 0, 1, 50, 1000):
                assert 0.0 <= criterion.desirability(value) <= 1.0


def test_lead_like_profile_is_stricter_than_oral():
    """Leads need headroom, so the same compound should score lower as a lead."""
    assert score(ERLOTINIB, "lead_like").score <= score(ERLOTINIB, "oral").score


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="unknown profile"):
        score(ASPIRIN, "nonexistent")


# --------------------------------------------------------- synthetic accessibility


def test_sa_score_orders_by_difficulty():
    simple = assess(ASPIRIN)
    macrocycle = assess(
        "CCC1C(C(C(C(=NCCOCCOCCOCCOC)C(CC(C(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)"
        "(C)OC)C)OC3C(C(CC(O3)C)N(C)C)O)(C)O)C)C)O)(C)O"
    )
    assert simple.sa_score < macrocycle.sa_score
    assert simple.tier == "readily accessible"


def test_unassigned_stereocentres_are_flagged():
    """A structure with undefined stereochemistry is a mixture, not a compound."""
    assessment = assess("CC(O)C(N)C(=O)O")
    assert assessment.n_unassigned_stereocentres > 0
    assert any("unassigned" in flag for flag in assessment.flags)


# ------------------------------------------------------------------ analog design


def test_implausible_chemistry_is_rejected():
    assert implausibility("COBr") == "hypohalite ester (O-halogen)"
    assert implausibility("ClBr") == "halogen-halogen bond"
    assert implausibility(ASPIRIN) is None


def test_transformation_respects_attachment_context():
    """REGRESSION, found by the agent: methyl-to-bromo on a methoxy oxygen
    produces a hypobromite ester."""
    aromatic = Transformation(
        lhs="C[*:1]", rhs="Br[*:1]", context="c",
        n_pairs=28, median_delta=0.5, std_delta=0.4, min_delta=0.0, max_delta=1.0,
    )
    assert apply_transformation("COc1ccc(Nc2ncnc3ccccc23)cc1", aromatic) == []
    assert apply_transformation("Cc1ccc(Nc2ncnc3ccccc23)cc1", aromatic)


def test_transformation_reliability_reflects_spread():
    """A large median with a large spread is not well supported."""
    tight = Transformation("C[*:1]", "F[*:1]", "c", 10, 0.5, 0.2, 0.1, 0.8)
    noisy = Transformation("C[*:1]", "F[*:1]", "c", 10, 0.5, 1.4, -1.9, 2.5)
    rare = Transformation("C[*:1]", "F[*:1]", "c", 2, 0.5, 0.1, 0.4, 0.6)
    assert tight.reliability == "well-supported"
    assert noisy.reliability == "context-dependent"
    assert rare.reliability == "anecdotal"


def test_context_label_is_readable():
    assert Transformation("C[*:1]", "F[*:1]", "c", 5, 0, 0, 0, 0).context_label == "aromatic C"
    assert Transformation("C[*:1]", "F[*:1]", "O", 5, 0, 0, 0, 0).context_label == "O"


# ---------------------------------------------------------------- R-group / SAR


@pytest.fixture
def para_series():
    """A clean congeneric series: one core, one varying position, additive."""
    return pd.DataFrame(
        {
            "smiles": [
                "Fc1ccc(Nc2ncnc3ccccc23)cc1",
                "Clc1ccc(Nc2ncnc3ccccc23)cc1",
                "Brc1ccc(Nc2ncnc3ccccc23)cc1",
                "Cc1ccc(Nc2ncnc3ccccc23)cc1",
                "COc1ccc(Nc2ncnc3ccccc23)cc1",
                "Oc1ccc(Nc2ncnc3ccccc23)cc1",
                "N#Cc1ccc(Nc2ncnc3ccccc23)cc1",
                "O=[N+]([O-])c1ccc(Nc2ncnc3ccccc23)cc1",
            ],
            "pchembl": [7.0, 7.5, 8.0, 6.5, 6.0, 5.5, 7.2, 6.8],
        }
    )


def test_decomposition_finds_a_common_core(para_series):
    result = decompose(para_series)
    assert result is not None
    assert result.n_decomposed >= 6
    assert result.positions


def test_substituent_table_requires_repeat_observations(para_series):
    result = decompose(para_series)
    assert substituent_table(result, min_count=99).empty


def test_free_wilson_refuses_tiny_datasets():
    tiny = pd.DataFrame(
        {"smiles": ["Fc1ccccc1", "Clc1ccccc1", "Brc1ccccc1"], "pchembl": [7.0, 7.5, 8.0]}
    )
    result = decompose(tiny)
    if result is not None:
        model = free_wilson(result)
        assert model is None or model.warning is not None


def test_free_wilson_reports_cross_validated_fit(para_series):
    """A one-hot model memorizes a small series, so the training fit is
    meaningless and must not be what gets reported."""
    result = decompose(para_series)
    model = free_wilson(result)
    if model is not None and model.warning is None:
        assert 0.0 <= model.r_squared <= 1.0
        assert "extrapolate" in model.to_dict()["limitation"]


def test_no_analog_diagnosis_distinguishes_enantiomer_swaps():
    """An empty result must say which of three things happened.

    Found by running the pipeline on DRD2: the only transformations touching an
    aminotetralin scaffold are enantiomer swaps, and a parent already in the
    produced configuration yields nothing. That is a real finding about the SAR
    of that scaffold, and a bare empty list hides it.
    """
    from chilecule.tools.design import diagnose_no_analogs

    swap = Transformation(
        lhs="Oc1cccc2c1CC[C@@H]([*:1])C2", rhs="Oc1cccc2c1CC[C@H]([*:1])C2",
        context="N", n_pairs=5, median_delta=1.0, std_delta=0.2,
        min_delta=0.5, max_delta=1.5,
    )
    diagnosis = diagnose_no_analogs("CCCN(CCC)[C@H]1CCc2c(O)cccc2C1", [swap])
    assert diagnosis.n_enantiomer_swaps == 1
    assert "enantiomer" in diagnosis.explanation
    assert diagnosis.examples


def test_no_analog_diagnosis_reports_fragment_mismatch():
    from chilecule.tools.design import diagnose_no_analogs

    unrelated = Transformation(
        lhs="[*:1]C1CCCCC1", rhs="[*:1]C1CCCC1", context="c",
        n_pairs=5, median_delta=0.3, std_delta=0.1, min_delta=0.1, max_delta=0.5,
    )
    diagnosis = diagnose_no_analogs("CC(=O)Oc1ccccc1C(=O)O", [unrelated])
    assert diagnosis.n_fragment_mismatch == 1
    assert "does not have" in diagnosis.explanation


def test_ignore_stereo_is_opt_in():
    """Enantiomers differ in potency by orders of magnitude; transferring a
    matched-pair statistic across them must be a deliberate choice."""
    swap = Transformation(
        lhs="Oc1cccc2c1CC[C@@H]([*:1])C2", rhs="CC[*:1]",
        context="N", n_pairs=5, median_delta=1.0, std_delta=0.2,
        min_delta=0.5, max_delta=1.5,
    )
    parent = "CCCN(CCC)[C@H]1CCc2c(O)cccc2C1"
    assert apply_transformation(parent, swap) == []
    assert apply_transformation(parent, swap, ignore_stereo=True)
