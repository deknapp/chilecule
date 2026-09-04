"""Molecular docking via smina, AutoDock Vina, or gnina.

**What a docking score is.** It is a rough estimate of binding pose quality
from an empirical scoring function fitted to a few hundred crystal structures.
Correlation between docking scores and measured binding affinity, across
diverse chemical series, is weak -- typically Pearson r of 0.3-0.5 in
published assessments, which is not enough to rank a series.

**What that means in practice.** Docking is good at generating plausible poses
and at coarse enrichment: it will usually place true binders somewhere in the
top decile of a large library. It is bad at telling you which of two analogs
is more potent. Any workflow presenting docking scores as predicted potency is
misleading its reader, so this module reports scores as ``score`` -- never as
``predicted_affinity`` -- and the workflows treat them as a ranking signal
alongside other evidence, never as the answer.

**The control that makes results interpretable.** :func:`redock_control`
re-docks the co-crystallized ligand into its own structure and measures RMSD
to the crystallographic pose. Under 2.0 A means the protocol can reproduce a
known answer for this site. Above that, nothing else the protocol says about
this target should be believed. This check costs one docking run and is
skipped by almost every automated pipeline.

Licensing: smina, AutoDock Vina 1.2+, and gnina are all Apache-2.0 as source.
They are invoked here as external processes over files -- never imported, never
linked -- so their licenses, and those of anything they link (conda-forge
builds of smina link Open Babel, GPL-2.0), do not reach this codebase. See
docs/LICENSING.md.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from .chem import parse_smiles
from .structure import DockingBox

log = logging.getLogger(__name__)

DOCKING_PROGRAMS = ("smina", "gnina", "vina")


class DockingUnavailable(RuntimeError):
    """Raised when no docking program is on PATH."""


@dataclass
class DockedPose:
    smiles: str
    score: float
    rank: int
    sdf_block: str = ""

    def to_dict(self) -> dict:
        return {"smiles": self.smiles, "score": round(self.score, 2), "rank": self.rank}


@dataclass
class DockingResult:
    smiles: str
    poses: list[DockedPose]
    program: str
    error: str | None = None

    @property
    def best_score(self) -> float | None:
        return self.poses[0].score if self.poses else None

    def to_dict(self) -> dict:
        return {
            "smiles": self.smiles,
            "program": self.program,
            "best_score": round(self.best_score, 2) if self.best_score is not None else None,
            "n_poses": len(self.poses),
            "error": self.error,
        }


def find_program(preferred: str | None = None) -> str | None:
    """Locate a docking program, preferring the caller's choice."""
    candidates = (preferred,) + DOCKING_PROGRAMS if preferred else DOCKING_PROGRAMS
    for name in candidates:
        if name and shutil.which(name):
            return name
    return None


def embed_ligand(smiles: str, n_conformers: int = 1, seed: int = 0xF00D) -> Chem.Mol | None:
    """Generate a 3D conformer with ETKDGv3, then minimize with MMFF.

    Docking programs sample ligand torsions themselves but need a sane starting
    geometry with correct stereochemistry and reasonable bond lengths. ETKDG
    supplies that; the MMFF step removes the strained local geometry embedding
    occasionally produces, which otherwise shows up as an implausible pose that
    the scoring function nonetheless rates well.
    """
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True
    if AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params) == -1:
        return None

    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
    except Exception:
        pass  # an unminimized conformer still docks; a missing one does not
    return mol


def dock(
    receptor_pdb: Path | str,
    smiles: str,
    box: DockingBox,
    *,
    program: str | None = None,
    exhaustiveness: int = 8,
    num_modes: int = 9,
    seed: int = 42,
    timeout: int = 600,
) -> DockingResult:
    """Dock one ligand into a prepared receptor.

    ``exhaustiveness`` trades runtime for sampling completeness roughly
    linearly. The Vina default of 8 is adequate for screening; 16-32 is
    appropriate when a single pose matters, such as in the redocking control.

    ``seed`` is fixed by default. Docking is stochastic, and an unseeded run
    produces different scores each time -- which makes any downstream
    comparison, including "did my change help", uninterpretable.
    """
    executable = find_program(program)
    if executable is None:
        raise DockingUnavailable(
            "No docking program found on PATH. Install one with:\n"
            "    micromamba install -c conda-forge smina\n"
            "or run ./install.sh --tier dock"
        )

    mol = embed_ligand(smiles, seed=seed)
    if mol is None:
        return DockingResult(smiles, [], executable, error="3D embedding failed")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ligand_file = tmp_path / "ligand.sdf"
        output_file = tmp_path / "docked.sdf"

        writer = Chem.SDWriter(str(ligand_file))
        writer.write(mol)
        writer.close()

        command = [
            executable,
            "-r", str(receptor_pdb),
            "-l", str(ligand_file),
            "-o", str(output_file),
            "--center_x", str(box.center[0]),
            "--center_y", str(box.center[1]),
            "--center_z", str(box.center[2]),
            "--size_x", str(box.size[0]),
            "--size_y", str(box.size[1]),
            "--size_z", str(box.size[2]),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", str(num_modes),
            "--seed", str(seed),
            "--cpu", "1",
        ]

        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired:
            return DockingResult(smiles, [], executable, error=f"timed out after {timeout}s")

        if completed.returncode != 0 or not output_file.exists():
            return DockingResult(
                smiles, [], executable,
                error=(completed.stderr or completed.stdout or "docking failed")[:500],
            )
        return DockingResult(smiles, _parse_poses(output_file, smiles), executable)


def _parse_poses(sdf_path: Path, smiles: str) -> list[DockedPose]:
    """Read scored poses out of a docking program's SDF output.

    smina writes ``minimizedAffinity``; gnina adds ``CNNscore`` and
    ``CNNaffinity``. Both are read, preferring the physics score for the
    primary ranking so that results stay comparable across programs.
    """
    poses: list[DockedPose] = []
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    for rank, pose_mol in enumerate(supplier, start=1):
        if pose_mol is None:
            continue
        score = None
        for prop in ("minimizedAffinity", "affinity", "CNNaffinity"):
            if pose_mol.HasProp(prop):
                try:
                    score = float(pose_mol.GetProp(prop))
                    break
                except ValueError:
                    continue
        if score is None:
            continue
        poses.append(
            DockedPose(
                smiles=smiles, score=score, rank=rank,
                sdf_block=Chem.MolToMolBlock(pose_mol, kekulize=False),
            )
        )
    return sorted(poses, key=lambda p: p.score)  # more negative is better


@dataclass
class RedockResult:
    """Outcome of re-docking a crystallographic ligand into its own structure."""

    residue_name: str
    rmsd: float | None
    score: float | None
    passed: bool
    threshold: float
    detail: str

    def to_dict(self) -> dict:
        return {
            "ligand": self.residue_name,
            "rmsd_angstrom": round(self.rmsd, 2) if self.rmsd is not None else None,
            "docking_score": round(self.score, 2) if self.score is not None else None,
            "passed": self.passed,
            "threshold_angstrom": self.threshold,
            "detail": self.detail,
        }


def redock_control(
    receptor_pdb: Path | str,
    native_ligand_sdf: Path | str,
    box: DockingBox,
    *,
    threshold: float = 2.0,
    exhaustiveness: int = 16,
    program: str | None = None,
) -> RedockResult:
    """Re-dock the crystallographic ligand and measure RMSD to its known pose.

    The field standard is 2.0 A: below it, the protocol reproduced the
    experimentally observed binding mode; above it, the protocol did not find
    the right answer even when the right answer was in its training data, and
    predictions for novel ligands at this site carry no weight.

    RMSD is computed with :func:`rdMolAlign.CalcRMS`, which accounts for
    molecular symmetry. Naive atom-order RMSD reports a large deviation for a
    perfectly docked para-substituted phenyl ring that happens to be flipped,
    and would fail a protocol that worked.
    """
    native = Chem.SDMolSupplier(str(native_ligand_sdf), removeHs=True, sanitize=True)
    reference = next((m for m in native if m is not None), None)
    if reference is None:
        return RedockResult("unknown", None, None, False, threshold,
                            "could not read the native ligand")

    residue_name = reference.GetProp("_Name") if reference.HasProp("_Name") else "native"
    smiles = Chem.MolToSmiles(reference)

    result = dock(receptor_pdb, smiles, box, program=program,
                  exhaustiveness=exhaustiveness, num_modes=1)
    if result.error or not result.poses:
        return RedockResult(residue_name, None, None, False, threshold,
                            f"redocking failed: {result.error}")

    docked = Chem.MolFromMolBlock(result.poses[0].sdf_block, sanitize=True)
    if docked is None:
        return RedockResult(residue_name, None, result.best_score, False, threshold,
                            "docked pose could not be parsed")

    try:
        rmsd = rdMolAlign.CalcRMS(Chem.RemoveHs(docked), Chem.RemoveHs(reference))
    except Exception as exc:
        return RedockResult(residue_name, None, result.best_score, False, threshold,
                            f"RMSD calculation failed: {exc}")

    passed = rmsd <= threshold
    detail = (
        f"reproduced the crystallographic pose to {rmsd:.2f} A -- "
        "the protocol is valid for this site"
        if passed else
        f"failed to reproduce the crystallographic pose ({rmsd:.2f} A > {threshold} A). "
        "Do not interpret docking results for this target until this passes; "
        "check the box definition, the protonation state, and whether the site "
        "requires a cofactor that was stripped during receptor preparation."
    )
    return RedockResult(residue_name, float(rmsd), result.best_score, passed, threshold, detail)
