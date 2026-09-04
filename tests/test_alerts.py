"""Tests for structural alert screening."""

from chilecule.tools.alerts import screen


def test_clean_molecule_matches_nothing():
    report = screen("CN1C=NC2=C1C(=O)N(C)C(=O)N2C")  # caffeine
    assert report is not None
    assert report.clean
    assert report.max_severity is None


def test_alerts_are_attributed_to_the_right_catalog():
    """REGRESSION: attribution originally guessed from the description string
    and filed every Brenk hit under PAINS_A, inflating its severity.

    RDKit tags each entry with a FilterSet property; use it. phenol_ester is a
    Brenk alert, not a PAINS one.
    """
    report = screen("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    assert not report.clean
    catalogs = {alert.catalog for alert in report.alerts}
    assert catalogs == {"BRENK"}
    assert report.max_severity == "medium"


def test_approved_drugs_do_match_alerts():
    """Aspirin matching a Brenk alert is why this module annotates rather than
    filters. A cascade that deletes on any alert deletes marketed drugs."""
    assert not screen("CC(=O)Oc1ccccc1C(=O)O").clean


def test_pains_tiers_are_distinguished():
    report = screen("O=C1CSC(=S)N1")  # rhodanine
    assert not report.clean
    assert any(alert.catalog.startswith("PAINS") for alert in report.alerts)


def test_alerts_carry_a_literature_reference():
    """An exclusion the user cannot chase to a paper is not auditable."""
    report = screen("O=C1CSC(=S)N1")
    assert any(alert.reference for alert in report.alerts)


def test_unparseable_input_returns_none():
    assert screen("not a molecule") is None
