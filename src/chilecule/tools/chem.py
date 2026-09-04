"""Molecule standardization and physicochemical profiling.

Everything downstream -- SAR analysis, filtering, docking, model building --
assumes molecules have been through :func:`standardize`. Skipping it is the
single most common source of silently wrong cheminformatics: the same compound
deposited as a hydrochloride salt, a zwitterion, and a neutral tautomer will
produce three different InChIKeys and three different fingerprints, which
quietly inflates dataset size and corrupts any train/test split built on
structural identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache

from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

# RDKit logs valence and kekulization complaints to stderr for every bad input.
# Callers get structured errors instead; see parse_smiles.
RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class Standardized:
    """Result of standardizing one input structure."""

    smiles: str | None
    inchikey: str | None
    input_smiles: str
    changed: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.smiles is not None


@lru_cache(maxsize=1)
def _largest_fragment() -> rdMolStandardize.LargestFragmentChooser:
    return rdMolStandardize.LargestFragmentChooser()


@lru_cache(maxsize=1)
def _uncharger() -> rdMolStandardize.Uncharger:
    return rdMolStandardize.Uncharger()


@lru_cache(maxsize=1)
def _tautomer_enumerator() -> rdMolStandardize.TautomerEnumerator:
    enumerator = rdMolStandardize.TautomerEnumerator()
    # Guard against combinatorial blowup on polycyclic tautomer systems.
    enumerator.SetMaxTautomers(256)
    return enumerator


def parse_smiles(smiles: str) -> Chem.Mol | None:
    """Parse SMILES, returning None rather than raising on invalid input."""
    if not smiles or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles.strip())


def standardize(
    smiles: str,
    *,
    neutralize: bool = True,
    canonical_tautomer: bool = True,
) -> Standardized:
    """Normalize one structure to a canonical form suitable for comparison.

    The pipeline is: sanitize -> normalize functional groups -> keep the largest
    organic fragment (drops counterions and solvates) -> optionally neutralize
    charges -> optionally pick a canonical tautomer.

    ``neutralize`` is on by default because bioactivity databases are wildly
    inconsistent about protonation state, and the deposited state rarely
    reflects the state at assay pH. Turn it off when formal charge is the point,
    e.g. when profiling permanently charged quaternary ammoniums.

    ``canonical_tautomer`` is the expensive step. It matters for any workflow
    that deduplicates or joins on structure, because keto/enol and amide/imidic
    pairs are extremely common in medicinal chemistry series and are otherwise
    treated as distinct compounds.
    """
    mol = parse_smiles(smiles)
    if mol is None:
        return Standardized(None, None, smiles, error="unparseable SMILES")

    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = _largest_fragment().choose(mol)
        if neutralize:
            mol = _uncharger().uncharge(mol)
        if canonical_tautomer:
            mol = _tautomer_enumerator().Canonicalize(mol)
        Chem.SanitizeMol(mol)
    except Exception as exc:  # RDKit raises a variety of C++-backed exceptions
        return Standardized(None, None, smiles, error=f"{type(exc).__name__}: {exc}")

    out = Chem.MolToSmiles(mol)
    return Standardized(
        smiles=out,
        inchikey=Chem.MolToInchiKey(mol) or None,
        input_smiles=smiles,
        changed=out != (Chem.MolToSmiles(parse_smiles(smiles)) or smiles),
    )


@dataclass(frozen=True)
class Properties:
    """Physicochemical profile of a single molecule."""

    smiles: str
    mw: float
    clogp: float
    tpsa: float
    hbd: int
    hba: int
    rotatable_bonds: int
    aromatic_rings: int
    heavy_atoms: int
    fraction_csp3: float
    formal_charge: int
    stereocenters: int
    qed: float
    rule_of_five_violations: int = field(default=0)
    veber_pass: bool = field(default=True)

    def to_dict(self) -> dict:
        return asdict(self)


def properties(smiles: str) -> Properties | None:
    """Compute the physicochemical descriptors that drive early triage.

    These are the descriptors that appear on essentially every hit-assessment
    slide. They are cheap, interpretable, and -- unlike a docking score --
    they mean the same thing across targets.
    """
    mol = parse_smiles(smiles)
    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)
    clogp = Crippen.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)

    ro5 = sum([mw > 500, clogp > 5, hbd > 5, hba > 10])
    # Veber: oral bioavailability correlates with flexibility and polar surface
    # area better than with molecular weight alone.
    veber = rotb <= 10 and tpsa <= 140

    return Properties(
        smiles=Chem.MolToSmiles(mol),
        mw=round(mw, 2),
        clogp=round(clogp, 2),
        tpsa=round(tpsa, 2),
        hbd=hbd,
        hba=hba,
        rotatable_bonds=rotb,
        aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(mol),
        heavy_atoms=mol.GetNumHeavyAtoms(),
        fraction_csp3=round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        formal_charge=Chem.GetFormalCharge(mol),
        stereocenters=len(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)),
        qed=round(QED.qed(mol), 3),
        rule_of_five_violations=ro5,
        veber_pass=veber,
    )


def ligand_efficiency(potency_nm: float, heavy_atoms: int) -> float | None:
    """Ligand efficiency: binding energy per heavy atom, in kcal/mol/HA.

    LE = -RT ln(Kd) / HA, evaluated at 298 K. The convention in the literature
    is to substitute IC50 or Ki for Kd, which is an approximation but a
    universal one.

    LE is how medicinal chemists avoid rewarding compounds for being large.
    A 10 nM hit with 45 heavy atoms (LE 0.24) has far less room to grow than a
    1 uM fragment with 16 heavy atoms (LE 0.51), and the fragment is usually
    the better starting point despite being 100x weaker.
    """
    import math

    if potency_nm is None or potency_nm <= 0 or not heavy_atoms:
        return None
    # 0.5961 kcal/mol = RT at 298 K
    delta_g = -0.5961 * math.log(potency_nm * 1e-9)
    return round(delta_g / heavy_atoms, 3)


def lipophilic_efficiency(potency_nm: float, clogp: float) -> float | None:
    """LLE (a.k.a. LipE) = pIC50 - cLogP.

    Tracks whether potency is being bought with lipophilicity, which is the
    classic failure mode of hit-to-lead: potency climbs, cLogP climbs faster,
    and the series walks into solubility, promiscuity, and hERG problems.
    An LLE above 5 is generally considered a healthy lead.
    """
    import math

    if potency_nm is None or potency_nm <= 0:
        return None
    pic50 = -math.log10(potency_nm * 1e-9)
    return round(pic50 - clogp, 2)
