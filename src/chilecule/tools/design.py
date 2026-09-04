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
from functools import lru_cache

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMMPA, rdmolops

from .chem import parse_smiles, standardize
from .sar import attachment_context, fingerprint, matched_pairs, transformation_summary

# Substructures that cannot be made, or would not survive contact with a
# solvent. This is a backstop behind the attachment-context check, not a
# substitute for it: fragment recombination can produce chemistry that is
# formally valid to RDKit and absurd to a chemist.
IMPLAUSIBLE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("[OX2][Cl,Br,I]", "hypohalite ester (O-halogen)"),
    ("[SX2][Cl,Br,I]", "sulfenyl halide (S-halogen)"),
    ("[Cl,Br,I][Cl,Br,I]", "halogen-halogen bond"),
    ("[NX3][Cl,Br,I]", "N-halamine"),
    ("[OX2][OX2][OX2]", "trioxide"),
    ("[NX3][NX3][NX3]", "triazane"),
)


@lru_cache(maxsize=1)
def _implausible_queries() -> tuple[tuple[object, str], ...]:
    return tuple(
        (Chem.MolFromSmarts(pattern), label) for pattern, label in IMPLAUSIBLE_PATTERNS
    )


def implausibility(smiles: str) -> str | None:
    """Name the reason a structure could not exist, or None if it is plausible."""
    mol = parse_smiles(smiles)
    if mol is None:
        return "unparseable"
    for query, label in _implausible_queries():
        if query is not None and mol.HasSubstructMatch(query):
            return label
    return None

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Transformation:
    """One structural change, with the evidence for what it does."""

    lhs: str
    rhs: str
    context: str
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
    def context_label(self) -> str:
        """Human-readable attachment environment."""
        if not self.context:
            return "unspecified"
        aromatic = self.context.islower()
        return f"{'aromatic ' if aromatic else ''}{self.context.upper()}"

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
            "attached_to": self.context_label,
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
                context=str(row.get("context", "")),
                n_pairs=int(row["n_pairs"]),
                median_delta=float(row["median_delta"]),
                std_delta=float(row["std_delta"]) if pd.notna(row["std_delta"]) else 0.0,
                min_delta=float(row["min_delta"]),
                max_delta=float(row["max_delta"]),
            )
        )
    return sorted(transformations, key=lambda t: -t.median_delta)


def _canonical_fragment(smiles: str, ignore_stereo: bool = False) -> str | None:
    """Canonical form of a fragment carrying an attachment point.

    ``ignore_stereo`` strips stereochemistry before canonicalizing, which makes
    a transformation mined on one enantiomer match the other. That is off by
    default: enantiomers routinely differ in potency by two orders of magnitude,
    so transferring a matched-pair statistic across them is not justified.
    It is available because the alternative -- silently returning nothing -- is
    a worse answer than an explicitly caveated one.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(
            mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        )
    except Exception:
        return None
    if ignore_stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def apply_transformation(
    smiles: str, transformation: Transformation, *, ignore_stereo: bool = False
) -> list[str]:
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

    wanted = _canonical_fragment(transformation.lhs, ignore_stereo)
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
            if _canonical_fragment(changed, ignore_stereo) != wanted:
                continue
            # The transformation is only valid where it was observed. Applying
            # "methyl becomes bromo" to the methyl of a methoxy group yields a
            # hypobromite ester, which RDKit will happily construct and no
            # chemist can make.
            if transformation.context and attachment_context(keep) != transformation.context:
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
            if not result.ok or result.smiles == Chem.MolToSmiles(mol):
                continue
            reason = implausibility(result.smiles)
            if reason:
                log.debug("rejected %s from %s: %s", result.smiles, transformation.label, reason)
                continue
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


@dataclass
class NoAnalogDiagnosis:
    """Why nothing was proposed. An empty list is not an answer."""

    n_transformations: int
    n_enantiomer_swaps: int = 0
    n_stereo_blocked: int = 0
    n_fragment_mismatch: int = 0
    examples: list[str] = field(default_factory=list)

    @property
    def explanation(self) -> str:
        if self.n_transformations == 0:
            return (
                "No transformation was observed often enough on this target to propose "
                "from. Either there is too little data, or its chemical matter is too "
                "diverse for matched-pair analysis."
            )
        parts = []
        if self.n_enantiomer_swaps:
            parts.append(
                f"{self.n_enantiomer_swaps} of the applicable transformations are "
                "enantiomer swaps, and this parent is already the configuration they "
                "produce -- the SAR on this scaffold is largely about chirality, which "
                "is itself the finding"
            )
        if self.n_stereo_blocked:
            parts.append(
                f"{self.n_stereo_blocked} would apply but for stereochemistry: there is "
                "a rule for the opposite enantiomer of a fragment in this parent. Re-run "
                "with ignore_stereo=True to see them, bearing in mind that enantiomers "
                "routinely differ in potency by two orders of magnitude"
            )
        if self.n_fragment_mismatch and not parts:
            parts.append(
                f"none of the {self.n_transformations} transformations matches a fragment "
                "of this parent in the attachment environment it was observed in. The "
                "changes that worked on this target were made at positions this compound "
                "does not have"
            )
        return ". ".join(parts) + "." if parts else (
            f"None of the {self.n_transformations} transformations applies to this parent."
        )

    def to_dict(self) -> dict:
        return {
            "n_transformations": self.n_transformations,
            "n_enantiomer_swaps": self.n_enantiomer_swaps,
            "n_stereo_blocked": self.n_stereo_blocked,
            "n_fragment_mismatch": self.n_fragment_mismatch,
            "explanation": self.explanation,
            "examples": self.examples,
        }


def diagnose_no_analogs(
    parent_smiles: str, transformations: list[Transformation]
) -> NoAnalogDiagnosis:
    """Explain why a parent produced no proposals.

    Three distinguishable reasons, and a chemist should be told which:
    the transformation is an enantiomer swap this compound has already made,
    it would apply but for stereochemistry, or no fragment matches at all.
    """
    diagnosis = NoAnalogDiagnosis(n_transformations=len(transformations))
    for transformation in transformations:
        if apply_transformation(parent_smiles, transformation):
            continue
        loose = apply_transformation(parent_smiles, transformation, ignore_stereo=True)
        same_skeleton = _canonical_fragment(transformation.lhs, True) == _canonical_fragment(
            transformation.rhs, True
        )
        if same_skeleton:
            diagnosis.n_enantiomer_swaps += 1
            if len(diagnosis.examples) < 3:
                diagnosis.examples.append(transformation.label)
        elif loose:
            diagnosis.n_stereo_blocked += 1
            if len(diagnosis.examples) < 3:
                diagnosis.examples.append(transformation.label)
        else:
            diagnosis.n_fragment_mismatch += 1
    return diagnosis


def enumerate_analogs(
    parent_smiles: str,
    transformations: list[Transformation],
    *,
    evidence_compounds: list[str] | None = None,
    domain_similarity_threshold: float = 0.4,
    max_per_transformation: int = 3,
    max_total: int = 200,
    ignore_stereo: bool = False,
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
        products = apply_transformation(
            parent.smiles, transformation, ignore_stereo=ignore_stereo
        )
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
