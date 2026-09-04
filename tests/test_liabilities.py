"""Tests for structure-based developability liability flagging.

Cases are pinned to compounds whose clinical behaviour is documented, so a
regression shows up as a chemically wrong answer rather than a changed number.
"""

import pytest

from chilecule.tools.liabilities import screen

CHLOROQUINE = "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12"
GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"


def categories(smiles: str) -> set[str]:
    return {item.category for item in screen(smiles).liabilities}


def test_chloroquine_flags_herg_and_phospholipidosis():
    """Chloroquine is the textbook cationic amphiphile and prolongs QT.

    A liability screen that misses it is not screening for anything.
    """
    flagged = categories(CHLOROQUINE)
    assert "hERG" in flagged
    assert "phospholipidosis" in flagged
    assert screen(CHLOROQUINE).highest_severity == "high"


def test_small_polar_drugs_are_not_flagged():
    """Aspirin and caffeine must come back clean, or the screen fires on
    everything and tells a chemist nothing."""
    assert not screen(ASPIRIN).liabilities
    assert not screen(CAFFEINE).liabilities


def test_herg_requires_a_basic_amine_not_just_lipophilicity():
    """An amide nitrogen is not basic and must not trigger the hERG flag."""
    amide = "CCCCCCCCc1ccc(C(=O)Nc2ccccc2)cc1"  # greasy, two aryls, no basic centre
    assert "hERG" not in categories(amide)


def test_anilines_and_nitroaromatics_are_flagged_as_bioactivation_risks():
    assert "reactive metabolite" in categories("Nc1ccc(Cl)cc1")
    assert "reactive metabolite" in categories("O=[N+]([O-])c1ccccc1")


def test_flat_aromatic_compounds_are_flagged_for_solubility():
    assert "solubility" in categories("c1ccc(-c2ccc(-c3ccc(-c4ccccc4)cc3)cc2)cc1")


def test_high_polarity_is_flagged_for_permeability():
    assert "permeability" in categories("NC(=O)C(N)CC(=O)NC(CO)C(=O)NC(CO)C(=O)O")


def test_every_flag_names_an_assay():
    """A flag without a next action is not useful."""
    for item in screen(CHLOROQUINE).liabilities + screen(GEFITINIB).liabilities:
        assert item.suggested_assay
        assert item.rationale
        assert item.severity in {"high", "medium", "low"}


def test_report_carries_the_do_not_delete_caveat():
    """Marketed drugs carry these flags; the report must say so."""
    payload = screen(CHLOROQUINE).to_dict()
    assert "not predictions" in payload["caveat"]
    assert "assay, not a deletion" in payload["caveat"]


def test_unparseable_input_returns_none():
    assert screen("not a molecule") is None


@pytest.mark.parametrize("smiles", [ASPIRIN, GEFITINIB, CHLOROQUINE, CAFFEINE])
def test_screen_is_deterministic(smiles):
    assert screen(smiles).to_dict() == screen(smiles).to_dict()
