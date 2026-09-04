"""Tests for standardization and property calculation."""

import pytest

from chilecule.tools.chem import (
    ligand_efficiency,
    lipophilic_efficiency,
    parse_smiles,
    properties,
    standardize,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ASPIRIN_SODIUM = "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]"
ASPIRIN_INCHIKEY = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"


def test_parse_rejects_garbage():
    assert parse_smiles("not_a_molecule") is None
    assert parse_smiles("") is None
    assert parse_smiles("   ") is None


def test_standardize_strips_counterion_to_parent():
    """A salt and its parent acid must converge on one structure.

    This is the property the whole pipeline depends on: without it, the same
    compound appears multiple times in a dataset under different identifiers.
    """
    salt = standardize(ASPIRIN_SODIUM)
    parent = standardize(ASPIRIN)
    assert salt.ok and parent.ok
    assert salt.smiles == parent.smiles
    assert salt.inchikey == parent.inchikey == ASPIRIN_INCHIKEY
    assert salt.changed is True


def test_standardize_reports_error_without_raising():
    result = standardize("this is not a smiles")
    assert not result.ok
    assert result.smiles is None
    assert "unparseable" in result.error


def test_standardize_is_idempotent():
    once = standardize(ASPIRIN_SODIUM)
    twice = standardize(once.smiles)
    assert once.smiles == twice.smiles


def test_properties_match_published_values():
    props = properties(ASPIRIN)
    assert props is not None
    assert props.mw == pytest.approx(180.16, abs=0.1)
    assert props.heavy_atoms == 13
    assert props.hbd == 1
    assert props.aromatic_rings == 1
    assert props.rule_of_five_violations == 0
    assert props.veber_pass is True


def test_ligand_efficiency_uses_correct_thermodynamics():
    """LE = -RT ln(Kd) / heavy atoms, at 298 K."""
    # 1 nM against 20 heavy atoms: dG = -0.5961 * ln(1e-9) = 12.35 kcal/mol
    assert ligand_efficiency(1.0, 20) == pytest.approx(12.35 / 20, abs=0.01)
    # A more potent compound of the same size is more efficient.
    assert ligand_efficiency(1.0, 20) > ligand_efficiency(100.0, 20)
    # The same potency in a larger molecule is less efficient.
    assert ligand_efficiency(10.0, 20) > ligand_efficiency(10.0, 40)


def test_ligand_efficiency_guards_bad_input():
    assert ligand_efficiency(0, 20) is None
    assert ligand_efficiency(-5, 20) is None
    assert ligand_efficiency(10, 0) is None


def test_lipophilic_efficiency_is_pic50_minus_clogp():
    # 10 nM -> pIC50 8.0; with cLogP 3.0, LLE = 5.0
    assert lipophilic_efficiency(10.0, 3.0) == pytest.approx(5.0, abs=0.01)
