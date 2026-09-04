"""Tests for structure handling, including the catalytic-metal regression."""

from pathlib import Path

import pytest

from chilecule.tools.structure import (
    CATALYTIC_METALS,
    CRYSTALLIZATION_ADDITIVES,
    StructureHit,
    box_from_ligand,
    extract_ligands,
    prepare_receptor,
    rank_structures,
)

# A minimal PDB with one protein residue, a ligand, a magnesium, a sulfate and
# a water -- one of each category the preparation step has to distinguish.
MINIMAL_PDB = """\
ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  ALA A   1      11.000  10.000  10.000  1.00 20.00           C
ATOM      3  C   ALA A   1      12.000  10.000  10.000  1.00 20.00           C
HETATM    4  C1  LIG A 500      20.000  20.000  20.000  1.00 20.00           C
HETATM    5  C2  LIG A 500      21.000  20.000  20.000  1.00 20.00           C
HETATM    6  C3  LIG A 500      22.000  20.000  20.000  1.00 20.00           C
HETATM    7  C4  LIG A 500      22.000  21.000  20.000  1.00 20.00           C
HETATM    8  C5  LIG A 500      22.000  22.000  20.000  1.00 20.00           C
HETATM    9  C6  LIG A 500      22.000  22.000  21.000  1.00 20.00           C
HETATM   10 MG    MG A 600      21.000  21.000  21.000  1.00 20.00          MG
HETATM   11  S   SO4 A 700      50.000  50.000  50.000  1.00 20.00           S
HETATM   12  O   HOH A 800      60.000  60.000  60.000  1.00 20.00           O
END
"""


@pytest.fixture
def structure_file(tmp_path) -> Path:
    path = tmp_path / "test.pdb"
    path.write_text(MINIMAL_PDB)
    return path


def test_magnesium_is_not_an_additive():
    """REGRESSION: catalytic metals were being stripped as crystallization junk.

    Mg coordinates the phosphates in every kinase nucleotide site, Zn is the
    catalytic centre of the metalloproteinases, Fe sits in the P450 heme.
    Removing them deletes the electrostatics the ligand binds to. This was
    caught empirically -- re-docking AMP-PNP into a Mg-stripped 5CNN missed
    the crystallographic pose at 2.87 A RMSD; retaining Mg gives 1.54 A.
    """
    for metal in ("MG", "ZN", "FE", "MN", "CA"):
        assert metal in CATALYTIC_METALS
        assert metal not in CRYSTALLIZATION_ADDITIVES


def test_extract_ligands_categorizes_heteroatoms(structure_file):
    ligands = extract_ligands(structure_file)
    categories = {lig.residue_name: lig.category for lig in ligands}
    assert categories["LIG"] == "ligand"
    assert categories["MG"] == "metal"
    assert categories["SO4"] == "additive"
    assert "HOH" not in categories  # water is never reported


def test_prepare_receptor_keeps_metals_and_drops_additives(structure_file):
    output = prepare_receptor(structure_file, keep_metals=True)
    text = output.read_text()
    assert "ALA" in text          # protein retained
    assert " MG " in text          # catalytic metal retained
    assert "SO4" not in text       # additive removed
    assert "HOH" not in text       # water removed
    assert "LIG" not in text       # the bound ligand is what we replace


def test_prepare_receptor_can_drop_metals_when_asked(structure_file):
    output = prepare_receptor(structure_file, keep_metals=False,
                              out_path=structure_file.with_name("no_metal.pdb"))
    assert " MG " not in output.read_text()


def test_box_from_ligand_encloses_the_ligand(structure_file):
    ligand = next(lig for lig in extract_ligands(structure_file) if lig.category == "ligand")
    box = box_from_ligand(structure_file, ligand, padding=5.0)
    # Ligand spans x 20-22, y 20-22, z 20-21; centre must sit inside that.
    assert 20.0 <= box.center[0] <= 22.0
    assert 20.0 <= box.center[1] <= 22.0
    # Every edge respects the 12 A floor and includes the padding.
    assert all(size >= 12.0 for size in box.size)
    assert "LIG" in box.source


def test_structure_ranking_prefers_resolution_over_coverage():
    """A 1.9 A structure of one domain beats a 3.1 A model of the whole protein."""
    hits = [
        StructureHit("7SYD", "A", 3.1, 1.00, "Electron Microscopy"),
        StructureHit("5CNN", "A", 1.9, 0.29, "X-ray diffraction"),
    ]
    ranked = rank_structures(hits)
    assert ranked[0].pdb_id == "5CNN"
    assert "1.9" in ranked[0].rationale


def test_nmr_ensembles_are_deprioritized():
    hits = [
        StructureHit("1NMR", "A", None, 1.0, "Solution NMR"),
        StructureHit("2XRD", "A", 2.4, 0.5, "X-ray diffraction"),
    ]
    assert rank_structures(hits)[0].pdb_id == "2XRD"


def test_ranking_deduplicates_by_entry():
    hits = [
        StructureHit("5CNN", "A", 1.9, 0.29, "X-ray diffraction"),
        StructureHit("5CNN", "B", 1.9, 0.29, "X-ray diffraction"),
    ]
    assert len(rank_structures(hits)) == 1
