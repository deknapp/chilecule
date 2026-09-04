"""Integration tests that touch the network or external binaries.

Deselect with:
    pytest -m "not network and not binary"

These are separated rather than mocked because what they verify is precisely
that the real services still behave as expected. A mocked ChEMBL client tests
that the mock matches the code, which is not the failure mode that matters --
the failure mode that matters is EBI changing a field name.
"""

import shutil

import pytest

from chilecule.tools.chembl import ChemblClient, censored_inactives, curate_activities
from chilecule.tools.docking import embed_ligand, find_program
from chilecule.tools.pockets import available as fpocket_available
from chilecule.tools.structure import (
    extract_ligands,
    fetch_pdb,
    prepare_receptor,
    rank_structures,
    structures_for_uniprot,
)

EGFR_UNIPROT = "P00533"
EGFR_CHEMBL = "CHEMBL203"
EGFR_STRUCTURE = "5CNN"

requires_docking = pytest.mark.skipif(
    find_program() is None, reason="no docking program on PATH"
)
requires_fpocket = pytest.mark.skipif(
    not fpocket_available(), reason="fpocket not on PATH"
)


# ------------------------------------------------------------------- network


@pytest.mark.network
def test_chembl_target_resolution_finds_single_protein():
    targets = ChemblClient().find_targets("EGFR")
    assert not targets.empty
    top = targets.iloc[0]
    assert top["target_chembl_id"] == EGFR_CHEMBL
    # Single-protein targets must sort first; complexes and cell lines cannot
    # be docked into.
    assert top["target_type"] == "SINGLE PROTEIN"
    assert top["uniprot"] == EGFR_UNIPROT


@pytest.mark.network
def test_chembl_activity_schema_is_unchanged():
    """Guards against EBI renaming a field the curation depends on."""
    raw = ChemblClient().fetch_activities(EGFR_CHEMBL, max_records=200)
    assert not raw.empty
    for column in (
        "canonical_smiles", "standard_value", "standard_units",
        "standard_relation", "standard_type", "data_validity_comment",
    ):
        assert column in raw.columns, f"ChEMBL no longer returns {column}"


@pytest.mark.network
def test_curation_removes_records_and_yields_compounds():
    client = ChemblClient()
    raw = client.fetch_activities(EGFR_CHEMBL, max_records=2000)
    curated, report = curate_activities(
        raw, target_chembl_id=EGFR_CHEMBL, chembl_release=client.release()
    )
    assert report.fetched == len(raw)
    assert report.total_removed > 0, "curation that removes nothing is not curating"
    assert len(curated) == report.final_compounds
    assert curated["pchembl"].between(0, 14).all()
    # Aggregation must actually collapse duplicates.
    assert report.final_compounds <= report.final_records


@pytest.mark.network
def test_censored_inactives_are_recoverable():
    raw = ChemblClient().fetch_activities(EGFR_CHEMBL, max_records=2000)
    inactives = censored_inactives(raw)
    assert len(inactives) > 0
    assert (inactives["value_nm"] >= 10_000).all()


@pytest.mark.network
def test_chembl_release_is_reported_for_attribution():
    """CC BY-SA requires naming the release."""
    assert ChemblClient().release().startswith("ChEMBL")


@pytest.mark.network
def test_structure_ranking_prefers_crystal_over_cryoem_for_egfr():
    ranked = rank_structures(structures_for_uniprot(EGFR_UNIPROT))
    assert ranked
    assert ranked[0].resolution is not None
    assert ranked[0].resolution < 2.5


@pytest.mark.network
def test_uniprot_resolution_is_not_confused_by_function_text():
    """REGRESSION: a free-text UniProt search for 'EGFR' returns CBL-C first,
    because CBL-C's function annotation mentions EGFR repeatedly."""
    from chilecule.workflows.dossier import uniprot_summary

    assert uniprot_summary("EGFR")["accession"] == EGFR_UNIPROT
    assert uniprot_summary("P00533")["accession"] == EGFR_UNIPROT
    assert uniprot_summary("erbB1")["accession"] == EGFR_UNIPROT


@pytest.mark.network
def test_pdb_download_and_ligand_extraction():
    path = fetch_pdb(EGFR_STRUCTURE)
    assert path.exists() and path.stat().st_size > 10_000

    ligands = extract_ligands(path)
    categories = {lig.category for lig in ligands}
    assert "ligand" in categories
    assert "metal" in categories, "5CNN contains Mg that must be categorized as a metal"


# -------------------------------------------------------------------- binary


def test_conformer_generation_needs_no_binary():
    mol = embed_ligand("CC(=O)Oc1ccccc1C(=O)O")
    assert mol is not None
    assert mol.GetNumConformers() == 1


@pytest.mark.network
@pytest.mark.binary
@requires_docking
def test_redocking_control_passes_when_metals_are_retained():
    """The end-to-end check that gives every other docking result its meaning.

    Re-docking AMP-PNP into 5CNN with Mg stripped misses the crystallographic
    pose (2.87 A). With Mg retained it reproduces it (1.54 A). This test pins
    the passing configuration.
    """
    from rdkit import Chem

    from chilecule.tools.docking import redock_control
    from chilecule.tools.structure import box_from_ligand, extract_ligand_mol

    path = fetch_pdb(EGFR_STRUCTURE)
    native = next(lig for lig in extract_ligands(path) if lig.category == "ligand")

    mol = extract_ligand_mol(path, native)
    assert mol is not None, "native ligand needs correct bond orders for a valid RMSD"

    native_sdf = path.with_name(f"{path.stem}_{native.residue_name}_test.sdf")
    writer = Chem.SDWriter(str(native_sdf))
    writer.write(mol)
    writer.close()

    receptor = prepare_receptor(path, keep_chain=native.chain_id, keep_metals=True)
    control = redock_control(
        receptor, native_sdf, box_from_ligand(path, native), exhaustiveness=16
    )
    assert control.rmsd is not None
    assert control.passed, f"redocking control regressed to {control.rmsd:.2f} A"


@pytest.mark.network
@pytest.mark.binary
@requires_docking
def test_docking_produces_scored_poses():
    from chilecule.tools.docking import dock
    from chilecule.tools.structure import box_from_ligand

    path = fetch_pdb(EGFR_STRUCTURE)
    native = next(lig for lig in extract_ligands(path) if lig.category == "ligand")
    receptor = prepare_receptor(path, keep_chain=native.chain_id)

    erlotinib = "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1"
    result = dock(receptor, erlotinib, box_from_ligand(path, native), num_modes=3)
    assert result.error is None
    assert result.poses
    # Vina-family scores for a real inhibitor in its own site sit well below zero.
    assert result.best_score < -5.0
    # Poses must come back sorted, best first.
    assert result.poses == sorted(result.poses, key=lambda p: p.score)


@pytest.mark.network
@pytest.mark.binary
@requires_fpocket
def test_fpocket_finds_the_known_atp_site():
    """The top-druggability pocket must land on the site the structure was
    solved with a ligand in."""
    import math

    from chilecule.tools.pockets import find_pockets
    from chilecule.tools.structure import box_from_ligand

    path = fetch_pdb(EGFR_STRUCTURE)
    native = next(lig for lig in extract_ligands(path) if lig.category == "ligand")
    receptor = prepare_receptor(path, keep_chain=native.chain_id)

    found = find_pockets(receptor, max_pockets=5)
    assert found
    true_center = box_from_ligand(path, native).center
    closest = min(math.dist(p.center, true_center) for p in found)
    assert closest < 6.0, f"best pocket was {closest:.1f} A from the known site"
    # Volumes must be real cavity volumes, not fpocket's normalized score.
    assert found[0].volume > 100


def test_missing_docking_program_raises_actionable_error(monkeypatch):
    from chilecule.tools import docking as docking_module
    from chilecule.tools.structure import DockingBox

    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(docking_module.DockingUnavailable) as excinfo:
        docking_module.dock("x.pdb", "CCO", DockingBox((0, 0, 0), (20, 20, 20), "test"))
    assert "conda-forge" in str(excinfo.value)
