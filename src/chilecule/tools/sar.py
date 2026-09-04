"""Structure-activity relationship analysis: scaffolds, matched pairs, cliffs.

The question a project team actually asks of a bioactivity table is not "what
is the average potency" but "what did changing this group do". Matched
molecular pair analysis answers that directly: it isolates single structural
transformations and reports their effect on potency across every context in
which they were made.

Activity cliffs -- pairs that are nearly identical but differ sharply in
potency -- are the highest-information records in any dataset. They are also
the records that break machine learning models, because the smoothness
assumption underlying similarity-based prediction is exactly what a cliff
violates. Finding them before you train is worth more than any hyperparameter
search afterwards.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdMMPA
from rdkit.Chem.Scaffolds import MurckoScaffold

from .chem import parse_smiles

# Morgan/ECFP4 with 2048 bits: the default that most published comparisons use.
# Reusing one generator matters -- constructing it per molecule dominates
# runtime on datasets of any size.
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprint(mol: Chem.Mol):
    return _MORGAN.GetFingerprint(mol)


def similarity(smiles_a: str, smiles_b: str) -> float | None:
    """Tanimoto similarity on ECFP4."""
    a, b = parse_smiles(smiles_a), parse_smiles(smiles_b)
    if a is None or b is None:
        return None
    return round(DataStructs.TanimotoSimilarity(fingerprint(a), fingerprint(b)), 3)


# ------------------------------------------------------------------ Scaffolds


def murcko_scaffold(smiles: str, generic: bool = False) -> str | None:
    """Bemis-Murcko scaffold: ring systems plus the linkers joining them.

    ``generic=True`` additionally strips atom and bond types, collapsing a
    pyrimidine and a benzene into the same carbon skeleton. Useful for asking
    "how many distinct topologies are in this series" rather than "how many
    distinct ring systems".
    """
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None


def scaffold_summary(df: pd.DataFrame, smiles_col: str = "smiles",
                     activity_col: str | None = "pchembl") -> pd.DataFrame:
    """Group a dataset by Bemis-Murcko scaffold.

    The output tells you whether a "diverse" dataset is actually one series
    wearing a hat. A set of 2,000 compounds spanning 30 scaffolds, with 60% of
    the mass on one of them, will produce an impressive random-split model and
    fail completely on new chemistry -- which is precisely why model evaluation
    here uses scaffold splits.
    """
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["scaffold"] = work[smiles_col].map(murcko_scaffold)
    work = work[work["scaffold"].notna()]

    agg: dict = {"n_compounds": (smiles_col, "size")}
    if activity_col and activity_col in work:
        agg |= {
            "median_activity": (activity_col, "median"),
            "max_activity": (activity_col, "max"),
            "activity_range": (activity_col, lambda s: round(float(s.max() - s.min()), 2)),
        }

    out = (
        work.groupby("scaffold")
        .agg(**agg)
        .reset_index()
        .sort_values("n_compounds", ascending=False)
        .reset_index(drop=True)
    )
    out["fraction_of_set"] = (out["n_compounds"] / len(work)).round(4)
    return out


# --------------------------------------------------- Matched molecular pairs


def attachment_context(fragment_smiles: str) -> str | None:
    """What the fragment attaches to: element plus aromaticity, e.g. 'c' or 'O'.

    A matched-pair transformation is only meaningful in the environment it was
    observed in. "methyl becomes bromo" is an ordinary aromatic substitution
    when the attachment atom is a ring carbon, and produces a hypobromite ester
    -- a compound that does not exist -- when the attachment atom is the oxygen
    of a methoxy group. Recording the context is what keeps the two apart.
    """
    mol = Chem.MolFromSmiles(fragment_smiles, sanitize=False)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        neighbours = atom.GetNeighbors()
        if not neighbours:
            return None
        neighbour = neighbours[0]
        symbol = neighbour.GetSymbol()
        return symbol.lower() if neighbour.GetIsAromatic() else symbol
    return None


@dataclass(frozen=True)
class MatchedPair:
    """One structural transformation observed between two compounds."""

    smiles_a: str
    smiles_b: str
    id_a: str
    id_b: str
    core: str
    transformation: str
    activity_a: float
    activity_b: float
    context: str = ""

    @property
    def delta(self) -> float:
        """Change in activity going from A to B, in log units."""
        return round(self.activity_b - self.activity_a, 2)

    def to_dict(self) -> dict:
        return {
            "id_a": self.id_a, "id_b": self.id_b,
            "smiles_a": self.smiles_a, "smiles_b": self.smiles_b,
            "transformation": self.transformation,
            "core": self.core,
            "activity_a": self.activity_a, "activity_b": self.activity_b,
            "delta": self.delta,
        }


def matched_pairs(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    id_col: str = "molecule_chembl_id",
    activity_col: str = "pchembl",
    max_heavy_atoms_in_change: int = 13,
    max_pairs: int | None = 50_000,
) -> list[MatchedPair]:
    """Enumerate matched molecular pairs by single-bond fragmentation.

    Implements the indexing strategy of Hussain and Rea (J Chem Inf Model 2010):
    fragment every molecule at acyclic single bonds, index the resulting
    (core, substituent) pairs by core, and pair up molecules sharing a core.

    ``max_heavy_atoms_in_change`` bounds how large a substituent swap may be
    before it stops being a "single change". Without it, the analysis pairs
    molecules that share only a phenyl ring and calls a wholesale scaffold
    replacement a matched pair, which defeats the purpose.
    """
    if df.empty:
        return []

    core_index: dict[str, list[tuple[str, str, str, float]]] = defaultdict(list)

    for _, row in df.iterrows():
        smiles = row[smiles_col]
        mol = parse_smiles(smiles)
        if mol is None:
            continue
        identifier = str(row.get(id_col, smiles))
        activity = row.get(activity_col)
        if activity is None or (isinstance(activity, float) and np.isnan(activity)):
            continue

        try:
            fragments = rdMMPA.FragmentMol(
                mol, maxCuts=1, maxCutBonds=30, resultsAsMols=False
            )
        except Exception:
            continue

        for core, chains in fragments:
            # A single cut yields an empty core field and both halves in
            # ``chains``; RDKit reports it that way for the maxCuts=1 case.
            piece = chains or core
            if not piece or "." not in piece:
                continue
            left, right = piece.split(".", 1)
            for keep, changed in ((left, right), (right, left)):
                if _attachment_heavy_atoms(changed) > max_heavy_atoms_in_change:
                    continue
                core_index[keep].append((identifier, smiles, changed, float(activity)))

    pairs: list[MatchedPair] = []
    seen: set[tuple[str, str, str]] = set()
    for core, members in core_index.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if a[0] == b[0] or a[2] == b[2]:
                    continue
                # Emit BOTH directions.
                #
                # A transformation is directional -- "Cl >> Br" and "Br >> Cl"
                # are different design decisions with equal and opposite
                # effects -- so each unordered pair contributes one record to
                # each direction's statistics.
                #
                # Emitting only i->j would make every delta a function of the
                # input frame's row order. A frame sorted by potency (which is
                # what curate_activities returns) would then yield uniformly
                # negative deltas, and the transformation table would claim
                # that every substitution ever made reduced potency.
                for (id_a, smi_a, sub_a, act_a), (id_b, smi_b, sub_b, act_b) in ((a, b), (b, a)):
                    key = (core, sub_a, sub_b)
                    if (id_a, id_b, key[1] + key[2]) in seen:
                        continue
                    seen.add((id_a, id_b, key[1] + key[2]))
                    pairs.append(
                        MatchedPair(
                            smiles_a=smi_a, smiles_b=smi_b,
                            id_a=id_a, id_b=id_b,
                            core=core,
                            transformation=f"{sub_a} >> {sub_b}",
                            activity_a=round(act_a, 2), activity_b=round(act_b, 2),
                            context=attachment_context(core) or "",
                        )
                    )
                if max_pairs and len(pairs) >= max_pairs:
                    return pairs
    return pairs


def _attachment_heavy_atoms(fragment_smiles: str) -> int:
    """Heavy atoms in a fragment, excluding attachment-point dummies."""
    mol = Chem.MolFromSmiles(fragment_smiles, sanitize=False)
    if mol is None:
        return 999
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


def transformation_summary(pairs: list[MatchedPair], min_occurrences: int = 3) -> pd.DataFrame:
    """Aggregate matched pairs into a table of transformations and their effects.

    This is the deliverable a medicinal chemist can act on: "replacing H with
    F at this position gained a median 0.4 log units across 12 pairs" is a
    design rule. A single observation is an anecdote, which is what
    ``min_occurrences`` guards against.

    Transformations are grouped by attachment context as well as by the swap
    itself, so that a change observed on an aromatic carbon is never pooled
    with the same change on an ether oxygen.
    """
    if not pairs:
        return pd.DataFrame()

    rows = [
        {"transformation": p.transformation, "context": p.context, "delta": p.delta}
        for p in pairs
    ]
    df = pd.DataFrame(rows)
    # Grouped by transformation AND attachment context: the same substituent
    # swap is a different chemical event on an aromatic carbon than on an ether
    # oxygen, and pooling them produces statistics for a transformation nobody
    # can perform.
    out = (
        df.groupby(["transformation", "context"])
        .agg(
            n_pairs=("delta", "size"),
            median_delta=("delta", "median"),
            mean_delta=("delta", "mean"),
            std_delta=("delta", "std"),
            min_delta=("delta", "min"),
            max_delta=("delta", "max"),
        )
        .reset_index()
    )
    out = out[out["n_pairs"] >= min_occurrences]
    for col in ("median_delta", "mean_delta", "std_delta", "min_delta", "max_delta"):
        out[col] = out[col].round(2)
    return out.sort_values("median_delta", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------ Activity cliffs


def activity_cliffs(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    id_col: str = "molecule_chembl_id",
    activity_col: str = "pchembl",
    similarity_threshold: float = 0.8,
    activity_threshold: float = 1.0,
    max_compounds: int = 2000,
) -> pd.DataFrame:
    """Find pairs that are structurally similar but differ sharply in potency.

    Reports SALI (structure-activity landscape index), defined as
    ``|dActivity| / (1 - similarity)``, which ranks cliffs by how abrupt they
    are rather than merely flagging them.

    Two reasons to run this before anything else:

    * Cliffs carry the SAR. A 100-fold potency swing from a single methyl is
      telling you about a specific interaction, and it is where the next design
      idea comes from.
    * Cliffs set the ceiling on model performance. A QSAR model cannot be
      simultaneously smooth and correct on a cliff pair, so a dataset dense in
      cliffs has an irreducible error floor that no amount of tuning removes.
      Reporting the cliff count alongside model R^2 is the honest way to
      present a model.
    """
    if df.empty:
        return pd.DataFrame()

    work = df.dropna(subset=[smiles_col, activity_col]).head(max_compounds).reset_index(drop=True)
    mols, keep = [], []
    for idx, smiles in enumerate(work[smiles_col]):
        mol = parse_smiles(smiles)
        if mol is not None:
            mols.append(fingerprint(mol))
            keep.append(idx)
    work = work.iloc[keep].reset_index(drop=True)
    if len(work) < 2:
        return pd.DataFrame()

    activities = work[activity_col].to_numpy(dtype=float)
    ids = work[id_col].astype(str).tolist() if id_col in work else work[smiles_col].tolist()
    smiles_list = work[smiles_col].tolist()

    rows = []
    for i in range(len(mols) - 1):
        # BulkTanimotoSimilarity over the upper triangle: one C++ call per row
        # instead of O(n^2) Python-level calls.
        sims = DataStructs.BulkTanimotoSimilarity(mols[i], mols[i + 1 :])
        for offset, sim in enumerate(sims):
            if sim < similarity_threshold:
                continue
            j = i + 1 + offset
            delta = abs(activities[i] - activities[j])
            if delta < activity_threshold:
                continue
            rows.append(
                {
                    "id_a": ids[i], "id_b": ids[j],
                    "smiles_a": smiles_list[i], "smiles_b": smiles_list[j],
                    "similarity": round(float(sim), 3),
                    "activity_a": round(float(activities[i]), 2),
                    "activity_b": round(float(activities[j]), 2),
                    "delta_activity": round(float(delta), 2),
                    # Guard the denominator: identical fingerprints (sim == 1.0)
                    # with different activities are either stereochemistry the
                    # fingerprint cannot see, or a data error worth inspecting.
                    "sali": round(float(delta / max(1.0 - sim, 1e-3)), 1),
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("sali", ascending=False).reset_index(drop=True)
