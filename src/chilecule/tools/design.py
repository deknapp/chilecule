"""What should I make next?

The central question of hit-to-lead, and the one this module answers -- with a
deliberate restriction on *how* it answers it.

These are not generated molecules. Every analog proposed here comes from a
transformation that medicinal chemists have already made, mined from ChEMBL
matched molecular pairs for the target or target class in question, and every
proposal carries the evidence behind it: how many times that exact swap was
made, what it did to potency, and how much the effect varied by context.

That constraint is the point. A generative model will happily propose a
molecule nobody has made for a reason. A transformation observed fourteen
times across a kinase series, with a median gain of 0.4 log units and a spread
of 0.3, is a design proposal a chemist can argue with -- and arguing with it is
what they will want to do.

The honest limitation, stated in every result: matched-pair statistics are
context-dependent. A transformation that helped at the solvent-exposed edge of
one series can do nothing, or the opposite, at a buried position in another.
The spread column is there to show when that is likely, and analogs whose
parent falls outside the chemical space the evidence came from are reported as
**no decision** rather than given a number that would look like a prediction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMMPA, rdmolops

from .chem import parse_smiles, standardize
from .sar import fingerprint, matched_pairs, transformation_summary

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Transformation:
    """One structural change, with the evidence for what it does."""

    lhs: str
    rhs: str
    n_pairs: int
    median_delta: float
    std_delta: float
    min_delta: float
    max_delta: float
    source: str = "ChEMBL matched pairs"

    @property
    def label(self) -> str:
        return f"{self.lhs} >> {self.rhs}"

    @property
    def reliability(self) -> str:
        """How much weight the evidence for this transformation can bear.

        Spread matters more than the median. A transformation with a median of
        +0.5 and a standard deviation of 1.2 has helped in some contexts and
        hurt badly in others; reporting only its median would be misleading in
        exactly the cases where it matters most.
        """
        if self.n_pairs < 3:
            return "anecdotal"
        if self.std_delta > 0.8:
            return "context-dependent"
        if self.n_pairs >= 8 and self.std_delta <= 0.4:
            return "well-supported"
        return "moderate"

    def to_dict(self) -> dict:
        return {
            "transformation": self.label,
            "n_pairs": self.n_pairs,
            "median_delta": self.median_delta,
            "std_delta": self.std_delta,
            "range": [self.min_delta, self.max_delta],
            "reliability": self.reliability,
            "source": self.source,
        }


def mine_transformations(
    activities: pd.DataFrame,
    *,
    min_occurrences: int = 4,
    max_compounds: int = 800,
    improving_only: bool = True,
    min_median_delta: float = 0.2,
) -> list[Transformation]:
    """Extract transformations with a track record from a curated activity set.

    ``improving_only`` keeps transformations whose median effect on potency is
    positive. Both directions are still mined -- the reverse of every kept
    transformation exists in the data -- but proposing a change with a track
    record of reducing potency is not a design suggestion.
    """
    pairs = matched_pairs(activities.head(max_compounds))
    summary = transformation_summary(pairs, min_occurrences=min_occurrences)
    if summary.empty:
        return []

    if improving_only:
        summary = summary[summary["median_delta"] >= min_median_delta]

    transformations = []
    for _, row in summary.iterrows():
        lhs, _, rhs = row["transformation"].partition(" >> ")
        transformations.append(
            Transformation(
                lhs=lhs.strip(),
                rhs=rhs.strip(),
                n_pairs=int(row["n_pairs"]),
                median_delta=float(row["median_delta"]),
                std_delta=float(row["std_delta"]) if pd.notna(row["std_delta"]) else 0.0,
                min_delta=float(row["min_delta"]),
                max_delta=float(row["max_delta"]),
            )
        )
    return sorted(transformations, key=lambda t: -t.median_delta)


def _canonical_fragment(smiles: str) -> str | None:
    """Canonical form of a fragment carrying an attachment point."""
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(
            mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        )
    except Exception:
        return None
    return Chem.MolToSmiles(mol)


def apply_transformation(smiles: str, transformation: Transformation) -> list[str]:
    """Apply one transformation to a molecule, returning every distinct product.

    Works by the same single-cut fragmentation used to mine the transformation
    in the first place: fragment the query, find fragments matching the
    left-hand side, substitute the right-hand side, and reassemble with
    ``molzip``. Doing it this way rather than with a reaction SMARTS guarantees
    the transformation applied is the transformation that was mined -- an
    equivalent SMARTS written by hand would drift.
    """
    mol = parse_smiles(smiles)
    if mol is None:
        return []

    wanted = _canonical_fragment(transformation.lhs)
    replacement = Chem.MolFromSmiles(transformation.rhs, sanitize=False)
    if wanted is None or replacement is None:
        return []

    try:
        fragmentations = rdMMPA.FragmentMol(mol, maxCuts=1, maxCutBonds=30, resultsAsMols=False)
    except Exception:
        return []

    products: set[str] = set()
    for core, chains in fragmentations:
        piece = chains or core
        if not piece or "." not in piece:
            continue
        left, right = piece.split(".", 1)
        for keep, changed in ((left, right), (right, left)):
            if _canonical_fragment(changed) != wanted:
                continue
            kept_mol = Chem.MolFromSmiles(keep, sanitize=False)
            if kept_mol is None:
                continue
            try:
                combined = Chem.CombineMols(kept_mol, replacement)
                product = rdmolops.molzip(combined)
                Chem.SanitizeMol(product)
            except Exception:
                continue
            result = standardize(Chem.MolToSmiles(product), canonical_tautomer=False)
            if result.ok and result.smiles != Chem.MolToSmiles(mol):
                products.add(result.smiles)
    return sorted(products)


@dataclass
class AnalogProposal:
    """One proposed compound, with the reason it is being proposed."""

    smiles: str
    parent_smiles: str
    transformation: Transformation
    similarity_to_parent: float
    in_evidence_domain: bool = True
    domain_note: str = ""
    properties: dict = field(default_factory=dict)

    @property
    def expected_effect(self) -> str:
        """What the evidence says, phrased so it cannot be read as a prediction."""
        if not self.in_evidence_domain:
            return (
                "no decision -- this parent falls outside the chemical space the "
                "transformation evidence was drawn from"
            )
        transformation = self.transformation
        return (
            f"{transformation.median_delta:+.2f} log units median across "
            f"{transformation.n_pairs} observed pairs "
            f"(SD {transformation.std_delta:.2f}, {transformation.reliability})"
        )

    def to_dict(self) -> dict:
        return {
            "smiles": self.smiles,
            "parent": self.parent_smiles,
            "transformation": self.transformation.label,
            "evidence": self.transformation.to_dict(),
            "expected_effect": self.expected_effect,
            "similarity_to_parent": self.similarity_to_parent,
            "in_evidence_domain": self.in_evidence_domain,
            "domain_note": self.domain_note,
            **self.properties,
        }


def enumerate_analogs(
    parent_smiles: str,
    transformations: list[Transformation],
    *,
    evidence_compounds: list[str] | None = None,
    domain_similarity_threshold: float = 0.4,
    max_per_transformation: int = 3,
    max_total: int = 200,
) -> list[AnalogProposal]:
    """Apply every transformation to a parent and collect the proposals.

    ``evidence_compounds`` are the structures the transformations were mined
    from. When supplied, each proposal records whether the parent resembles
    that set. A transformation mined from kinase inhibitors, applied to a
    steroid, produces a valid structure and a meaningless expectation, and the
    proposal says so instead of quoting a median.
    """
    parent = standardize(parent_smiles, canonical_tautomer=False)
    if not parent.ok:
        return []

    in_domain, domain_note = _evidence_domain(
        parent.smiles, evidence_compounds, domain_similarity_threshold
    )

    parent_mol = parse_smiles(parent.smiles)
    parent_fp = fingerprint(parent_mol)

    proposals: list[AnalogProposal] = []
    seen: set[str] = {parent.smiles}
    for transformation in transformations:
        products = apply_transformation(parent.smiles, transformation)
        for product in products[:max_per_transformation]:
            if product in seen:
                continue
            product_mol = parse_smiles(product)
            if product_mol is None:
                continue
            seen.add(product)
            proposals.append(
                AnalogProposal(
                    smiles=product,
                    parent_smiles=parent.smiles,
                    transformation=transformation,
                    similarity_to_parent=round(
                        DataStructs.TanimotoSimilarity(parent_fp, fingerprint(product_mol)), 3
                    ),
                    in_evidence_domain=in_domain,
                    domain_note=domain_note,
                )
            )
            if len(proposals) >= max_total:
                return proposals
    return proposals


def _evidence_domain(
    parent_smiles: str, evidence_compounds: list[str] | None, threshold: float
) -> tuple[bool, str]:
    """Is the parent inside the chemical space the evidence came from?

    Applicability domain, in the form that matters here. A model or a
    statistic is not reliable merely because it was computed successfully, and
    the honest output when a query sits outside the training space is 'no
    decision' rather than a number.
    """
    if not evidence_compounds:
        return True, "no evidence set supplied; applicability not assessed"

    parent_mol = parse_smiles(parent_smiles)
    if parent_mol is None:
        return False, "parent could not be parsed"

    parent_fp = fingerprint(parent_mol)
    references = [
        fingerprint(mol)
        for mol in (parse_smiles(s) for s in evidence_compounds[:1000])
        if mol is not None
    ]
    if not references:
        return True, "evidence set could not be fingerprinted"

    nearest = max(DataStructs.BulkTanimotoSimilarity(parent_fp, references))
    if nearest >= threshold:
        return True, f"parent is {nearest:.2f} similar to the closest evidence compound"
    return False, (
        f"parent is only {nearest:.2f} similar to the closest compound the "
        f"transformations were mined from (threshold {threshold}). The structures "
        "are valid; the expected effects are not transferable."
    )
