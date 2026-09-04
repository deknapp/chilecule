"""Developability liabilities, flagged from structure.

The risks a medicinal chemist checks for before committing a series: hERG,
metabolic soft spots, reactive metabolite precursors, phospholipidosis, and
solubility. Each is reported as a **structural risk flag with a stated reason**,
never as a prediction.

Why rules rather than a model. The usual approach is to train regressors on
public ADMET data and report a number. Those models are useful inside their
applicability domain and misleading outside it, they need heavy dependencies
(the standard toolkit pulls in torch and transformers), and shipping one here
would mean shipping predictions from data this project cannot reliably re-fetch.
A structural flag saying "basic amine plus cLogP 4.6 is the classic hERG
pharmacophore" is less precise and much more honest: it names the feature, the
reason, and what to measure. A chemist can argue with it. They cannot argue
with 0.63.

Every flag here is a hypothesis about what to test, not a result. The right
response to any of them is an assay, not a deletion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from rdkit import Chem

from .chem import parse_smiles, properties


@dataclass(frozen=True)
class Liability:
    """One flagged risk, with the evidence and what to do about it."""

    category: str
    severity: str  # "high" | "medium" | "low"
    finding: str
    rationale: str
    suggested_assay: str

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "severity": self.severity,
            "finding": self.finding,
            "rationale": self.rationale,
            "suggested_assay": self.suggested_assay,
        }


# Substructures worth naming, each with why it matters. SMARTS are deliberately
# narrow -- a pattern that fires on half a library is not information.
SOFT_SPOTS: tuple[tuple[str, str, str], ...] = (
    (
        "[CX4H2,CX4H3;$(C[c])]",
        "benzylic C-H",
        "Benzylic positions are the commonest site of CYP-mediated oxidation. "
        "Blocking with fluorine or a ring is the standard fix when clearance is high.",
    ),
    (
        "[NX3;H0;$(N(C)(C)[c,C])][CH3]",
        "N-methyl on a tertiary amine",
        "N-dealkylation is a first-pass metabolic route and often the major "
        "clearance pathway for tertiary amines.",
    ),
    (
        "[OX2]([CH3])[c]",
        "aryl methyl ether",
        "O-demethylation is a common Phase I route; the resulting phenol is "
        "usually rapidly conjugated and cleared.",
    ),
    (
        "[cH1]1[cH1][cH1][cH1][cH1][cH1]1",
        "unsubstituted phenyl",
        "Aromatic hydroxylation, typically at the para position. Substituting "
        "para is the usual block.",
    ),
)

BIOACTIVATION: tuple[tuple[str, str, str], ...] = (
    (
        "c1cc[s]c1",
        "thiophene",
        "Oxidised by CYP to a reactive S-oxide or epoxide. A recognised "
        "structural alert for idiosyncratic toxicity.",
    ),
    (
        "c1cc[o]c1",
        "furan",
        "Bioactivated to a reactive cis-enedione. Strongly associated with "
        "hepatotoxicity in marketed compounds withdrawn for that reason.",
    ),
    (
        "[NX3;H2][c]",
        "primary aniline",
        "Oxidised to hydroxylamine and nitroso species that bind protein "
        "covalently; also a route to genotoxic risk.",
    ),
    (
        "[N+](=O)[O-]",
        "nitro group",
        "Nitroreduction generates nitroso and hydroxylamine intermediates. "
        "A standard genotoxicity alert.",
    ),
    (
        "[CX3H1](=O)[#6]",
        "aldehyde",
        "Electrophilic and protein-reactive; also readily oxidised to the acid.",
    ),
    (
        "[CX3](=O)[CX3]=[CX3]",
        "Michael acceptor",
        "Covalent reactivity toward cysteine thiols. Intentional in covalent "
        "inhibitors and a liability everywhere else -- decide which this is.",
    ),
)


@lru_cache(maxsize=1)
def _compiled(patterns: tuple[tuple[str, str, str], ...]):
    return tuple(
        (Chem.MolFromSmarts(smarts), name, reason) for smarts, name, reason in patterns
    )


# A basic centre in the hERG sense: an amine likely protonated at physiological
# pH. Amides, anilines and sulfonamides are excluded because they are not basic.
BASIC_AMINE = "[NX3;H2,H1,H0;!$(N[#6]=[O,N,S]);!$(N[c]);!$(N[SX4](=O)=O);!$([N+])]"


@dataclass
class LiabilityReport:
    smiles: str
    liabilities: list[Liability] = field(default_factory=list)

    @property
    def highest_severity(self) -> str | None:
        for level in ("high", "medium", "low"):
            if any(item.severity == level for item in self.liabilities):
                return level
        return None

    @property
    def summary(self) -> str:
        if not self.liabilities:
            return "No structural liability flags raised."
        by_category: dict[str, int] = {}
        for item in self.liabilities:
            by_category[item.category] = by_category.get(item.category, 0) + 1
        parts = ", ".join(f"{count} {category}" for category, count in by_category.items())
        return (
            f"{len(self.liabilities)} flag(s): {parts}. "
            f"Highest severity {self.highest_severity}."
        )

    def to_dict(self) -> dict:
        return {
            "smiles": self.smiles,
            "n_flags": len(self.liabilities),
            "highest_severity": self.highest_severity,
            "summary": self.summary,
            "liabilities": [item.to_dict() for item in self.liabilities],
            "caveat": (
                "Structural risk flags, not predictions. Each names a feature and the "
                "reason it matters; none is a measurement. The response to a flag is an "
                "assay, not a deletion -- many marketed drugs carry several."
            ),
        }


def screen(smiles: str) -> LiabilityReport | None:
    """Flag developability risks visible from structure."""
    mol = parse_smiles(smiles)
    props = properties(smiles)
    if mol is None or props is None:
        return None

    found: list[Liability] = []
    basic_amine = Chem.MolFromSmarts(BASIC_AMINE)
    has_basic_amine = mol.HasSubstructMatch(basic_amine) if basic_amine else False

    # ---- hERG. The classic pharmacophore is a basic centre flanked by
    # lipophilic aromatic bulk; the risk tracks lipophilicity steeply.
    if has_basic_amine and props.clogp >= 3.0 and props.aromatic_rings >= 2:
        severity = "high" if props.clogp >= 4.0 else "medium"
        found.append(
            Liability(
                category="hERG",
                severity=severity,
                finding=(
                    f"basic amine with cLogP {props.clogp} and "
                    f"{props.aromatic_rings} aromatic rings"
                ),
                rationale=(
                    "The canonical hERG pharmacophore is a protonatable nitrogen with "
                    "lipophilic aromatic groups either side, binding Tyr652/Phe656 in the "
                    "channel pore. Risk rises steeply above cLogP 3.7; lowering "
                    "lipophilicity or reducing amine basicity are the standard mitigations."
                ),
                suggested_assay="hERG patch clamp, or a binding assay for early triage",
            )
        )

    # ---- Phospholipidosis: cationic amphiphilic drugs.
    if has_basic_amine and props.clogp >= 4.0:
        found.append(
            Liability(
                category="phospholipidosis",
                severity="medium",
                finding=f"cationic amphiphile (basic amine, cLogP {props.clogp})",
                rationale=(
                    "Cationic amphiphilic drugs accumulate in lysosomes and inhibit "
                    "phospholipases, causing lamellar body formation. Usually reversible "
                    "and usually a preclinical delay rather than a stop."
                ),
                suggested_assay="in vitro phospholipidosis screen in a macrophage line",
            )
        )

    # ---- Solubility.
    if props.aromatic_rings >= 4 or (props.fraction_csp3 < 0.15 and props.mw > 350):
        found.append(
            Liability(
                category="solubility",
                severity="medium" if props.aromatic_rings >= 4 else "low",
                finding=(
                    f"{props.aromatic_rings} aromatic rings, Fsp3 {props.fraction_csp3}"
                ),
                rationale=(
                    "Flat, aromatic-rich molecules crystallise well and dissolve badly. "
                    "Aromatic ring count above three correlates with poor developability, "
                    "and sp3 character tracks with solubility and clinical success."
                ),
                suggested_assay="kinetic and thermodynamic aqueous solubility",
            )
        )

    # ---- Permeability.
    if props.tpsa > 120 or props.hbd >= 4:
        found.append(
            Liability(
                category="permeability",
                severity="medium",
                finding=f"TPSA {props.tpsa}, {props.hbd} donors",
                rationale=(
                    "Polar surface area above ~120 A^2 and four or more donors both "
                    "predict poor passive permeability. Donors cost more than acceptors "
                    "because desolvation is the barrier."
                ),
                suggested_assay="Caco-2 or MDCK permeability",
            )
        )

    # ---- Metabolic soft spots and bioactivation.
    for query, name, reason in _compiled(SOFT_SPOTS):
        if query is not None and mol.HasSubstructMatch(query):
            found.append(
                Liability(
                    category="metabolic soft spot",
                    severity="low",
                    finding=name,
                    rationale=reason,
                    suggested_assay="microsomal stability, with met-ID if clearance is high",
                )
            )
    for query, name, reason in _compiled(BIOACTIVATION):
        if query is not None and mol.HasSubstructMatch(query):
            found.append(
                Liability(
                    category="reactive metabolite",
                    severity="high",
                    finding=name,
                    rationale=reason,
                    suggested_assay="glutathione trapping, and a covalent binding assay",
                )
            )

    return LiabilityReport(smiles=Chem.MolToSmiles(mol), liabilities=found)
