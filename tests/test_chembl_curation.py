"""Tests for ChEMBL curation. No network access -- fixtures are synthetic frames
shaped exactly like the API's response."""

import pandas as pd
import pytest

from chilecule.tools.chembl import censored_inactives, curate_activities


def record(**overrides):
    base = {
        "molecule_chembl_id": "CHEMBL1",
        "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "standard_type": "IC50",
        "standard_relation": "=",
        "standard_value": 100.0,
        "standard_units": "nM",
        "target_organism": "Homo sapiens",
        "data_validity_comment": None,
        "document_chembl_id": "DOC1",
    }
    return {**base, **overrides}


def test_censored_records_are_removed_not_treated_as_values():
    """'>10000 nM' must not become a data point.

    Keeping it teaches a regression model that every inactive compound has
    exactly 10 uM potency.
    """
    frame = pd.DataFrame([
        record(molecule_chembl_id="A"),
        record(molecule_chembl_id="B", standard_relation=">", standard_value=10000.0),
    ])
    curated, report = curate_activities(frame)
    assert len(curated) == 1
    assert curated.iloc[0]["molecule_chembl_id"] == "A"
    assert any("censored" in step for step, _, _ in report.steps)


def test_censored_records_are_recoverable_as_negatives():
    frame = pd.DataFrame([
        record(molecule_chembl_id="B", standard_relation=">", standard_value=10000.0,
               canonical_smiles="c1ccccc1CCN"),
    ])
    inactives = censored_inactives(frame)
    assert len(inactives) == 1
    assert inactives.iloc[0]["value_nm"] == pytest.approx(10000.0)


def test_units_are_converted_not_assumed():
    frame = pd.DataFrame([
        record(molecule_chembl_id="A", standard_value=1.0, standard_units="uM"),
        record(molecule_chembl_id="B", standard_value=1000.0, standard_units="nM",
               canonical_smiles="c1ccccc1CCN"),
    ])
    curated, _ = curate_activities(frame)
    # 1 uM and 1000 nM are the same concentration, so both land at pChEMBL 6.
    assert set(curated["pchembl"].round(2)) == {6.0}


def test_unconvertible_units_are_dropped():
    """ug.mL-1 cannot become molar without a molecular weight, and guessing one
    is how unit bugs reach publication."""
    frame = pd.DataFrame([
        record(standard_units="ug.mL-1"),
    ])
    curated, report = curate_activities(frame)
    assert curated.empty
    assert any("unconvertible" in step for step, _, _ in report.steps)


def test_curator_flagged_records_are_dropped():
    frame = pd.DataFrame([
        record(data_validity_comment="Potential transcription error"),
    ])
    curated, report = curate_activities(frame)
    assert curated.empty
    assert any("curator flag" in step for step, _, _ in report.steps)


def test_wrong_species_is_dropped():
    frame = pd.DataFrame([record(target_organism="Rattus norvegicus")])
    curated, report = curate_activities(frame, organism="Homo sapiens")
    assert curated.empty
    assert any("organism" in step for step, _, _ in report.steps)


def test_implausible_ligand_efficiency_is_rejected():
    """A tiny fragment cannot be sub-nanomolar.

    6-aminoquinazoline has 10 heavy atoms. At 0.1 nM its ligand efficiency
    would be 1.36 kcal/mol per atom, well above the thermodynamic ceiling for
    an organic ligand. Absolute range checks pass this record; only the
    efficiency check catches it.
    """
    frame = pd.DataFrame([
        record(canonical_smiles="Nc1ccc2cncnc2c1", standard_value=0.1),
    ])
    curated, report = curate_activities(frame)
    assert curated.empty
    assert any("ligand efficiency" in step for step, _, _ in report.steps)


def test_inconsistent_replicates_are_dropped():
    """30 nM and 8 uM for one compound is not a measurement with error bars."""
    frame = pd.DataFrame([
        record(standard_value=30.0),
        record(standard_value=8000.0),
    ])
    curated, report = curate_activities(frame, max_replicate_spread_log=1.0)
    assert curated.empty
    assert any("replicate" in step for step, _, _ in report.steps)


def test_consistent_replicates_are_aggregated():
    frame = pd.DataFrame([
        record(standard_value=90.0, document_chembl_id="DOC1"),
        record(standard_value=110.0, document_chembl_id="DOC2"),
    ])
    curated, _ = curate_activities(frame)
    assert len(curated) == 1
    assert curated.iloc[0]["n_measurements"] == 2
    assert curated.iloc[0]["n_documents"] == 2


def test_salts_aggregate_with_their_parent():
    """The sodium salt and the free acid are one compound."""
    frame = pd.DataFrame([
        record(molecule_chembl_id="A", canonical_smiles="CC(=O)Oc1ccccc1C(=O)O"),
        record(molecule_chembl_id="B", canonical_smiles="CC(=O)Oc1ccccc1C(=O)[O-].[Na+]"),
    ])
    curated, _ = curate_activities(frame)
    assert len(curated) == 1
    assert curated.iloc[0]["n_measurements"] == 2


def test_report_accounts_for_every_record():
    frame = pd.DataFrame([
        record(molecule_chembl_id="A"),
        record(molecule_chembl_id="B", standard_relation=">", standard_value=99999.0),
        record(molecule_chembl_id="C", data_validity_comment="Outside typical range"),
        record(molecule_chembl_id="D", target_organism="Mus musculus"),
    ])
    _, report = curate_activities(frame)
    assert report.fetched == 4
    assert report.total_removed == 3
    assert report.final_compounds == 1


def test_empty_input_returns_empty_report():
    curated, report = curate_activities(pd.DataFrame())
    assert curated.empty
    assert report.fetched == 0
