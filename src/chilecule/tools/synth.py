"""Can it be made?

Synthetic accessibility is the constraint that quietly kills more designed
compounds than potency does. A generated analog with a beautiful docking score
and an eight-step route through a reagent nobody sells is not a design
proposal, and any tool that enumerates structures without saying so is wasting
a chemist's time.

Two complementary signals, both free:

**SA score** (Ertl & Schuffenhauer, J Cheminform 2009) scores 1 (trivial) to 10
(intractable), from fragment frequencies in PubChem plus penalties for size,
stereocentres, macrocycles and spiro atoms. It is a *familiarity* measure, not
a route prediction -- it asks whether a molecule is built from pieces that
appear in known compounds. That makes it good at flagging exotic structures and
poor at noticing that a familiar-looking molecule needs an awkward
regiochemistry.

**Structural complexity flags** catch what SA score misses: unassigned
stereocentres that imply a separation problem, and quaternary stereocentres,
bridgeheads and macrocycles that imply a hard synthesis regardless of how
ordinary the fragments look.

Neither is retrosynthesis. When a compound matters, ask a chemist or run a
retrosynthesis tool; these are for triage across hundreds of structures.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache

from rdkit import Chem, RDConfig
from rdkit.Chem import Descriptors, rdMolDescriptors

from .chem import parse_smiles


@lru_cache(maxsize=1)
def _sascorer():
    """RDKit ships the SA scorer in Contrib, which is not on the import path."""
    contrib = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if contrib not in sys.path:
        sys.path.append(contrib)
    import sascorer  # noqa: PLC0415

    return sascorer


@dataclass(frozen=True)
class SynthesisAssessment:
    smiles: str
    sa_score: float
    n_stereocentres: int
    n_unassigned_stereocentres: int
    n_spiro: int
    n_bridgehead: int
    n_macrocycles: int
    n_rings: int
    flags: tuple[str, ...]

    @property
    def tier(self) -> str:
        """Coarse buckets, because the underlying number is not precise enough
        to support finer ones."""
        if self.sa_score < 3.5:
            return "readily accessible"
        if self.sa_score < 5.0:
            return "moderate"
        if self.sa_score < 6.5:
            return "demanding"
        return "likely impractical"

    @property
    def summary(self) -> str:
        base = f"SA score {self.sa_score:.2f} -- {self.tier}"
        return f"{base}. {' '.join(self.flags)}" if self.flags else base

    def to_dict(self) -> dict:
        return {
            "smiles": self.smiles,
            "sa_score": round(self.sa_score, 2),
            "tier": self.tier,
            "n_stereocentres": self.n_stereocentres,
            "n_unassigned_stereocentres": self.n_unassigned_stereocentres,
            "n_spiro": self.n_spiro,
            "n_bridgehead": self.n_bridgehead,
            "n_macrocycles": self.n_macrocycles,
            "flags": list(self.flags),
            "summary": self.summary,
            "caveat": (
                "SA score measures fragment familiarity, not route length. It is triage "
                "across many structures, not a substitute for retrosynthetic analysis."
            ),
        }


def assess(smiles: str) -> SynthesisAssessment | None:
    """Score how hard a molecule is likely to be to make."""
    mol = parse_smiles(smiles)
    if mol is None:
        return None

    score = float(_sascorer().calculateScore(mol))
    centres = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    )
    unassigned = sum(1 for _, label in centres if label == "?")
    spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    ring_info = mol.GetRingInfo()
    macrocycles = sum(1 for ring in ring_info.AtomRings() if len(ring) >= 12)

    flags: list[str] = []
    if unassigned:
        flags.append(
            f"{unassigned} unassigned stereocentre(s) -- the structure is a mixture "
            "unless separated or set synthetically."
        )
    if len(centres) >= 4:
        flags.append(f"{len(centres)} stereocentres; expect a chirality control problem.")
    if macrocycles:
        flags.append("Macrocyclic: ring closure is usually the yield-limiting step.")
    if spiro:
        flags.append(f"{spiro} spiro atom(s).")
    if bridgehead:
        flags.append(f"{bridgehead} bridgehead atom(s); bridged systems are rarely cheap.")
    if Descriptors.MolWt(mol) > 700:
        flags.append("Above 700 Da, where purification and characterization get harder.")

    return SynthesisAssessment(
        smiles=Chem.MolToSmiles(mol),
        sa_score=score,
        n_stereocentres=len(centres),
        n_unassigned_stereocentres=unassigned,
        n_spiro=spiro,
        n_bridgehead=bridgehead,
        n_macrocycles=macrocycles,
        n_rings=ring_info.NumRings(),
        flags=tuple(flags),
    )
