"""Protein structure retrieval, inspection, and preparation for docking.

Choosing a structure is a scientific decision that automated pipelines
routinely get wrong. The highest-coverage structure is often a cryo-EM model
of a full-length receptor at 3.5 A, which is excellent for understanding
architecture and useless for docking. What structure-based design needs is a
high-resolution crystal structure of the relevant domain with a bound ligand
chemically related to the series -- because the side-chain conformations in an
apo structure are frequently incompatible with any ligand at all.

:func:`rank_structures` therefore ranks on suitability for docking rather than
on coverage, and explains its ranking so the choice can be overridden.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import requests

log = logging.getLogger(__name__)

PDBE_API = "https://www.ebi.ac.uk/pdbe/api"
RCSB_FILES = "https://files.rcsb.org/download"
UNIPROT_API = "https://rest.uniprot.org/uniprotkb"

# Heteroatom codes that are crystallization additives, buffers, ions, or
# cryoprotectants rather than ligands. Treating one of these as "the bound
# ligand" is how a docking box ends up centred on a sulfate on the protein
# surface.
CRYSTALLIZATION_ADDITIVES = {
    "HOH", "DOD", "SO4", "PO4", "GOL", "EDO", "PEG", "PG4", "PGE", "MPD", "TRS",
    "ACT", "ACY", "FMT", "CIT", "MES", "EPE", "DMS", "IMD", "BME", "IPA", "MRD",
    "NA", "K", "CL", "BR", "IOD", "NO3", "AZI", "CO3", "UNX", "UNL",
}

# Catalytic and structural metals. These are NOT additives, and stripping them
# is a modelling error rather than a cleanup step: Mg(2+) coordinates the
# phosphates in every kinase nucleotide site, Zn(2+) is the catalytic centre of
# the metalloproteinases, and Fe sits in the heme of every P450. Removing them
# deletes the electrostatics that the ligand was binding to, and docking into
# the resulting site produces poses that cannot be right.
#
# They are retained by default and reported in their own category so the
# decision to keep or drop one is explicit.
CATALYTIC_METALS = {"MG", "ZN", "MN", "FE", "CA", "CD", "NI", "CU", "CO", "HG", "FE2", "3CO"}

# Cofactors: real biology, but not the ligand you dock against unless the
# target is the cofactor site itself. Flagged separately rather than discarded.
COMMON_COFACTORS = {"ATP", "ADP", "AMP", "GTP", "GDP", "NAD", "NAP", "NDP", "FAD", "FMN", "SAM", "SAH", "HEM"}


@dataclass
class StructureHit:
    pdb_id: str
    chain_id: str
    resolution: float | None
    coverage: float | None
    method: str
    score: float = 0.0
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "pdb_id": self.pdb_id, "chain_id": self.chain_id,
            "resolution": self.resolution, "coverage": self.coverage,
            "method": self.method, "score": round(self.score, 2),
            "rationale": self.rationale,
        }


@dataclass
class Ligand:
    """A heteroatom group extracted from a structure."""

    residue_name: str
    chain_id: str
    residue_number: int
    n_atoms: int
    centroid: tuple[float, float, float]
    category: str  # "ligand" | "cofactor" | "metal" | "additive"

    def to_dict(self) -> dict:
        return {
            "residue_name": self.residue_name, "chain_id": self.chain_id,
            "residue_number": self.residue_number, "n_atoms": self.n_atoms,
            "centroid": [round(c, 2) for c in self.centroid],
            "category": self.category,
        }


def structures_for_uniprot(accession: str, timeout: int = 30) -> list[StructureHit]:
    """All PDB entries mapped to a UniProt accession, via PDBe SIFTS mappings."""
    url = f"{PDBE_API}/mappings/best_structures/{accession}"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        log.warning("PDBe lookup failed for %s: %s", accession, exc)
        return []

    entries = payload.get(accession, [])
    return [
        StructureHit(
            pdb_id=e.get("pdb_id", "").upper(),
            chain_id=e.get("chain_id", ""),
            resolution=e.get("resolution"),
            coverage=e.get("coverage"),
            method=e.get("experimental_method", "unknown"),
        )
        for e in entries
    ]


def rank_structures(
    hits: list[StructureHit],
    *,
    max_resolution: float = 2.8,
    prefer_xray: bool = True,
) -> list[StructureHit]:
    """Rank structures by suitability for structure-based design.

    Scoring reflects what actually matters for docking, in order:

    * **Resolution.** Side-chain positions below ~2.5 A are reliable; above
      3.0 A they are modelled, not observed. Docking into modelled side chains
      produces confident-looking poses with no evidential basis.
    * **Method.** X-ray and, increasingly, high-resolution cryo-EM are usable.
      NMR ensembles and low-resolution EM models are not, for this purpose.
    * **Coverage.** Matters much less than resolution. A 2.0 A structure of the
      kinase domain beats a 3.5 A structure of the whole receptor for every
      question this project asks.

    Deduplicates to one entry per PDB ID, since chains of the same structure
    are the same coordinates.
    """
    best_per_entry: dict[str, StructureHit] = {}

    for hit in hits:
        resolution = hit.resolution
        reasons = []

        if resolution is None:
            score = 20.0
            reasons.append("no resolution reported")
        elif resolution <= 2.0:
            score = 100.0
            reasons.append(f"{resolution} A -- side chains directly observed")
        elif resolution <= 2.5:
            score = 85.0
            reasons.append(f"{resolution} A -- good for docking")
        elif resolution <= max_resolution:
            score = 60.0
            reasons.append(f"{resolution} A -- usable, side chains less certain")
        else:
            score = 25.0
            reasons.append(f"{resolution} A -- too low for reliable side-chain geometry")

        method = (hit.method or "").lower()
        if prefer_xray and "diffraction" in method:
            score += 10
            reasons.append("X-ray")
        elif "microscopy" in method:
            score -= 5
            reasons.append("cryo-EM")
        elif "nmr" in method:
            score -= 30
            reasons.append("NMR ensemble -- not a single docking target")

        if hit.coverage:
            score += 5 * hit.coverage
            reasons.append(f"covers {hit.coverage:.0%} of the sequence")

        scored = StructureHit(
            pdb_id=hit.pdb_id, chain_id=hit.chain_id, resolution=hit.resolution,
            coverage=hit.coverage, method=hit.method, score=score,
            rationale="; ".join(reasons),
        )
        existing = best_per_entry.get(scored.pdb_id)
        if existing is None or scored.score > existing.score:
            best_per_entry[scored.pdb_id] = scored

    return sorted(best_per_entry.values(), key=lambda h: -h.score)


def fetch_pdb(pdb_id: str, dest_dir: Path | str | None = None, timeout: int = 60) -> Path:
    """Download a PDB-format coordinate file, caching by ID."""
    pdb_id = pdb_id.lower().strip()
    dest_dir = Path(dest_dir) if dest_dir else Path.home() / ".cache" / "chilecule" / "pdb"
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{pdb_id}.pdb"
    if path.exists() and path.stat().st_size > 0:
        return path

    response = requests.get(f"{RCSB_FILES}/{pdb_id}.pdb", timeout=timeout)
    response.raise_for_status()
    path.write_text(response.text)
    return path


def extract_ligands(pdb_path: Path | str, min_atoms: int = 6) -> list[Ligand]:
    """Find candidate bound ligands in a PDB file.

    Parses HETATM records directly rather than pulling in a structure library.
    The categories matter: ``min_atoms`` alone would still admit glycerol and
    PEG fragments, which is why the additive list exists.
    """
    pdb_path = Path(pdb_path)
    groups: dict[tuple[str, str, int], list[tuple[float, float, float]]] = {}

    for line in pdb_path.read_text().splitlines():
        if not line.startswith("HETATM"):
            continue
        residue_name = line[17:20].strip()
        chain_id = line[21].strip() or "A"
        try:
            residue_number = int(line[22:26])
            coords = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        groups.setdefault((residue_name, chain_id, residue_number), []).append(coords)

    ligands = []
    for (residue_name, chain_id, residue_number), coords in groups.items():
        if residue_name in {"HOH", "DOD"}:
            continue
        if residue_name in CATALYTIC_METALS:
            category = "metal"
        elif residue_name in CRYSTALLIZATION_ADDITIVES:
            category = "additive"
        elif residue_name in COMMON_COFACTORS:
            category = "cofactor"
        else:
            category = "ligand"
        if category == "ligand" and len(coords) < min_atoms:
            category = "additive"

        n = len(coords)
        centroid = tuple(sum(c[i] for c in coords) / n for i in range(3))
        ligands.append(
            Ligand(residue_name, chain_id, residue_number, n, centroid, category)  # type: ignore[arg-type]
        )

    # Largest true ligand first -- usually the one the structure was solved for.
    return sorted(
        ligands,
        key=lambda lig: (
            {"ligand": 0, "cofactor": 1, "metal": 2, "additive": 3}[lig.category],
            -lig.n_atoms,
        ),
    )


@dataclass
class DockingBox:
    """Search volume for docking, in Angstroms."""

    center: tuple[float, float, float]
    size: tuple[float, float, float]
    source: str

    def to_dict(self) -> dict:
        return {
            "center_x": round(self.center[0], 2),
            "center_y": round(self.center[1], 2),
            "center_z": round(self.center[2], 2),
            "size_x": round(self.size[0], 1),
            "size_y": round(self.size[1], 1),
            "size_z": round(self.size[2], 1),
            "source": self.source,
        }


def box_from_ligand(pdb_path: Path | str, ligand: Ligand, padding: float = 5.0) -> DockingBox:
    """Build a docking box around a co-crystallized ligand.

    This is the most defensible way to define a site: the box is centred on
    chemical matter the protein is known to bind. Padding of 4-6 A gives a
    larger ligand room to explore without inflating the search volume, which
    costs accuracy as well as time -- an oversized box lets the sampler waste
    its budget on surface positions and report a confident score for a pose
    that is not in the site at all.
    """
    pdb_path = Path(pdb_path)
    coords = []
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("HETATM"):
            continue
        if (line[17:20].strip() != ligand.residue_name
                or (line[21].strip() or "A") != ligand.chain_id):
            continue
        try:
            if int(line[22:26]) != ligand.residue_number:
                continue
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        except ValueError:
            continue

    if not coords:
        raise ValueError(f"no atoms found for ligand {ligand.residue_name} in {pdb_path}")

    mins = [min(c[i] for c in coords) for i in range(3)]
    maxs = [max(c[i] for c in coords) for i in range(3)]
    center = tuple((mins[i] + maxs[i]) / 2 for i in range(3))
    size = tuple(max(maxs[i] - mins[i] + 2 * padding, 12.0) for i in range(3))
    return DockingBox(
        center=center,  # type: ignore[arg-type]
        size=size,  # type: ignore[arg-type]
        source=f"co-crystallized ligand {ligand.residue_name} "
               f"(chain {ligand.chain_id}, residue {ligand.residue_number})",
    )


def prepare_receptor(
    pdb_path: Path | str,
    out_path: Path | str | None = None,
    *,
    keep_chain: str | None = None,
    keep_cofactors: bool = True,
    keep_metals: bool = True,
) -> Path:
    """Strip a PDB file down to a receptor suitable for docking.

    Removes waters, crystallization additives, alternate conformations beyond
    the first, and optionally all chains but one. Keeps cofactors and catalytic
    metals by default, because removing the heme from a P450 or the magnesium
    from a kinase nucleotide site deletes the electrostatics the ligand was
    binding to. The redocking control in :mod:`chilecule.tools.docking` catches
    this class of error empirically -- re-docking a nucleotide into a
    magnesium-stripped kinase site misses the crystallographic pose.

    Deliberately does *not* add hydrogens or assign protonation states. That
    requires a pKa prediction at the assay's pH and is the responsibility of
    the docking program's own preparation step (smina and gnina do it
    internally). Doing it here badly would be worse than not doing it.
    """
    pdb_path = Path(pdb_path)
    out_path = Path(out_path) if out_path else pdb_path.with_name(f"{pdb_path.stem}_receptor.pdb")

    kept: list[str] = []
    for line in pdb_path.read_text().splitlines():
        record = line[:6]
        if record == "ATOM  ":
            if keep_chain and (line[21].strip() or "A") != keep_chain:
                continue
            altloc = line[16].strip()
            if altloc and altloc not in ("A", "1"):
                continue
            kept.append(line)
        elif record == "HETATM":
            residue_name = line[17:20].strip()
            if residue_name in CRYSTALLIZATION_ADDITIVES or residue_name in {"HOH", "DOD"}:
                continue
            is_metal = residue_name in CATALYTIC_METALS
            is_cofactor = residue_name in COMMON_COFACTORS
            if is_metal and not keep_metals:
                continue
            if is_cofactor and not keep_cofactors:
                continue
            if not (is_metal or is_cofactor):
                continue  # drop the bound ligand; it is what we are replacing
            if keep_chain and (line[21].strip() or "A") != keep_chain:
                continue
            kept.append(line)
        elif record in ("TER   ", "END   "):
            kept.append(line)

    out_path.write_text("\n".join(kept) + "\nEND\n")
    return out_path


RCSB_LIGANDS = "https://files.rcsb.org/ligands/download"


def extract_ligand_mol(
    pdb_path: Path | str, ligand: Ligand, timeout: int = 30
) -> "object | None":
    """Extract a co-crystallized ligand as a chemically correct RDKit molecule.

    PDB files store coordinates and element types but no bond orders. Reading
    HETATM records alone yields a molecule with the right shape and the wrong
    chemistry -- aromatic rings become single bonds, carboxylates lose their
    double bond -- which then fails to match the docked pose during RMSD
    comparison.

    The fix is to take geometry from the structure and connectivity from the
    PDB Chemical Component Dictionary's idealized template for that residue
    code, via :func:`AllChem.AssignBondOrdersFromTemplate`. This is the step
    that makes a redocking control meaningful rather than merely runnable.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    pdb_path = Path(pdb_path)
    lines = [
        line
        for line in pdb_path.read_text().splitlines()
        if line.startswith("HETATM")
        and line[17:20].strip() == ligand.residue_name
        and (line[21].strip() or "A") == ligand.chain_id
        and line[22:26].strip() == str(ligand.residue_number)
    ]
    if not lines:
        return None

    pose = Chem.MolFromPDBBlock("\n".join(lines) + "\nEND\n", sanitize=False, removeHs=True)
    if pose is None:
        return None

    try:
        response = requests.get(
            f"{RCSB_LIGANDS}/{ligand.residue_name}_ideal.sdf", timeout=timeout
        )
        response.raise_for_status()
        template = Chem.MolFromMolBlock(response.text, sanitize=True)
    except Exception as exc:
        log.warning("could not fetch CCD template for %s: %s", ligand.residue_name, exc)
        template = None

    if template is None:
        try:
            Chem.SanitizeMol(pose)
            return pose
        except Exception:
            return None

    try:
        corrected = AllChem.AssignBondOrdersFromTemplate(Chem.RemoveHs(template), pose)
        corrected.SetProp("_Name", ligand.residue_name)
        return corrected
    except Exception as exc:
        log.warning("bond order assignment failed for %s: %s", ligand.residue_name, exc)
        return None
