"""R-group decomposition and Free-Wilson analysis.

The deliverable a project team actually wants from a congeneric series: a table
of what each substituent does at each position, so the next round of compounds
can be chosen rather than guessed.

Free-Wilson analysis (Free & Wilson, J Med Chem 1964) is the oldest QSAR method
there is and remains one of the most useful, because it is interpretable in the
way a chemist thinks. It assumes activity is the sum of independent
contributions from each substituent position. That assumption is often wrong --
substituents interact, and when two positions cooperate the model will
underfit -- but the failure is informative: a poor Free-Wilson fit on a series
that looks congeneric is evidence of a real interaction between positions,
which is worth knowing.

Its hard limitation is that it cannot extrapolate. A substituent that appears
in no compound has no coefficient, so the model can rank what has been made and
says nothing about what has not. That is stated in the output rather than
buried, because the temptation to read a Free-Wilson table as a design
prediction is exactly where it goes wrong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFMCS, rdRGroupDecomposition

from .chem import parse_smiles

log = logging.getLogger(__name__)


def find_common_core(
    smiles_list: list[str],
    *,
    timeout: int = 30,
    threshold: float = 0.8,
    min_core_atoms: int = 6,
) -> str | None:
    """Maximum common substructure across a series, as a SMARTS core.

    ``threshold`` allows the core to be present in most rather than all
    compounds, since a real project series usually contains a few outliers that
    would otherwise shrink the core to a single ring.
    """
    mols = [m for m in (parse_smiles(s) for s in smiles_list) if m is not None]
    if len(mols) < 2:
        return None

    result = rdFMCS.FindMCS(
        mols,
        timeout=timeout,
        threshold=threshold,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
    )
    if result.canceled or result.numAtoms < min_core_atoms:
        return None
    return result.smartsString


@dataclass
class Decomposition:
    core: str
    table: pd.DataFrame
    positions: list[str]
    n_decomposed: int
    n_failed: int

    def to_dict(self) -> dict:
        return {
            "core_smarts": self.core,
            "positions": self.positions,
            "n_decomposed": self.n_decomposed,
            "n_failed": self.n_failed,
        }


def decompose(
    df: pd.DataFrame,
    *,
    smiles_col: str = "smiles",
    core: str | None = None,
    activity_col: str | None = "pchembl",
) -> Decomposition | None:
    """Split a series into a shared core plus per-position R-groups."""
    if df.empty:
        return None

    smiles_list = df[smiles_col].tolist()
    core = core or find_common_core(smiles_list)
    if core is None:
        return None

    core_mol = Chem.MolFromSmarts(core)
    if core_mol is None:
        return None

    mols, keep_index = [], []
    for idx, smiles in zip(df.index, smiles_list, strict=True):
        mol = parse_smiles(smiles)
        if mol is not None and mol.HasSubstructMatch(core_mol):
            mols.append(mol)
            keep_index.append(idx)

    if len(mols) < 3:
        return None

    params = rdRGroupDecomposition.RGroupDecompositionParameters()
    params.removeHydrogensPostMatch = True
    params.onlyMatchAtRGroups = False
    try:
        groups, unmatched = rdRGroupDecomposition.RGroupDecompose(
            [core_mol], mols, asSmiles=True, options=params
        )
    except Exception as exc:
        log.warning("R-group decomposition failed: %s", exc)
        return None

    if not groups:
        return None

    matched_index = [i for j, i in enumerate(keep_index) if j not in set(unmatched)]
    table = pd.DataFrame(groups, index=matched_index[: len(groups)])
    positions = [c for c in table.columns if c.startswith("R")]

    source = df.loc[table.index]
    table[smiles_col] = source[smiles_col].to_numpy()
    if activity_col and activity_col in source:
        table[activity_col] = source[activity_col].to_numpy()
    for extra in ("molecule_chembl_id",):
        if extra in source:
            table[extra] = source[extra].to_numpy()

    return Decomposition(
        core=core,
        table=table.reset_index(drop=True),
        positions=positions,
        n_decomposed=len(table),
        n_failed=len(smiles_list) - len(table),
    )


def substituent_table(
    decomposition: Decomposition, activity_col: str = "pchembl", min_count: int = 2
) -> pd.DataFrame:
    """Activity of every substituent at every position.

    The plain descriptive view, computed before any model is fitted. A
    substituent seen once carries no information about position preference, so
    ``min_count`` keeps the table to entries that mean something.
    """
    table = decomposition.table
    if activity_col not in table:
        return pd.DataFrame()

    rows = []
    for position in decomposition.positions:
        grouped = table.groupby(position)[activity_col]
        for substituent, values in grouped:
            if len(values) < min_count:
                continue
            rows.append(
                {
                    "position": position,
                    "substituent": substituent,
                    "n_compounds": len(values),
                    "mean_activity": round(float(values.mean()), 2),
                    "best_activity": round(float(values.max()), 2),
                    "spread": round(float(values.max() - values.min()), 2),
                }
            )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["position", "mean_activity"], ascending=[True, False])
        .reset_index(drop=True)
    )


@dataclass
class FreeWilsonModel:
    r_squared: float
    n_compounds: int
    n_terms: int
    intercept: float
    contributions: pd.DataFrame = field(default_factory=pd.DataFrame)
    warning: str | None = None

    @property
    def interpretation(self) -> str:
        if self.warning:
            return self.warning
        if self.r_squared >= 0.7:
            return (
                f"Additive model explains {self.r_squared:.0%} of the variance. The series "
                "behaves additively, so the contributions below can guide the next round."
            )
        if self.r_squared >= 0.4:
            return (
                f"Additive model explains only {self.r_squared:.0%} of the variance. Part of "
                "the SAR comes from interactions between positions, so single-substituent "
                "contributions are indicative rather than predictive."
            )
        return (
            f"Additive model explains {self.r_squared:.0%} of the variance -- the series is "
            "not additive. Substituent positions are interacting, or the activity data spans "
            "assays that are not comparable. Do not use these contributions for design; the "
            "poor fit is itself the finding."
        )

    def to_dict(self) -> dict:
        return {
            "r_squared": round(self.r_squared, 3),
            "n_compounds": self.n_compounds,
            "n_terms": self.n_terms,
            "intercept": round(self.intercept, 2),
            "interpretation": self.interpretation,
            "limitation": (
                "Free-Wilson cannot extrapolate. A substituent absent from the series has "
                "no coefficient, so this ranks what was made and is silent on what was not."
            ),
        }


def free_wilson(
    decomposition: Decomposition,
    activity_col: str = "pchembl",
    min_substituent_count: int = 2,
    alpha: float = 1.0,
) -> FreeWilsonModel | None:
    """Fit additive per-substituent contributions to activity.

    Uses ridge regression rather than ordinary least squares. Free-Wilson
    design matrices are one-hot and heavily collinear -- every compound has
    exactly one substituent per position -- and OLS on that produces enormous
    compensating coefficients that look like strong effects and are numerical
    artifacts. The ridge penalty shrinks them toward zero, which is the honest
    behaviour when the data cannot separate two terms.
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold, cross_val_predict
    except ImportError:  # pragma: no cover
        return None

    table = decomposition.table
    if activity_col not in table or len(table) < 6:
        return FreeWilsonModel(0.0, len(table), 0, 0.0,
                               warning="Too few compounds for a meaningful fit (need at least 6).")

    frequent: dict[str, set[str]] = {}
    for position in decomposition.positions:
        counts = table[position].value_counts()
        frequent[position] = set(counts[counts >= min_substituent_count].index)

    terms = [(p, s) for p, subs in frequent.items() for s in sorted(subs)]
    if len(terms) < 2:
        return FreeWilsonModel(
            0.0, len(table), 0, 0.0,
            warning="Not enough repeated substituents to fit an additive model.",
        )

    X = np.zeros((len(table), len(terms)))
    for row, (_, record) in enumerate(table.iterrows()):
        for col, (position, substituent) in enumerate(terms):
            if record[position] == substituent:
                X[row, col] = 1.0
    y = table[activity_col].to_numpy(dtype=float)

    model = Ridge(alpha=alpha)
    model.fit(X, y)

    # Cross-validated R^2, not the training fit. A one-hot model with a term
    # per substituent can memorize a small series completely, and reporting
    # that number as explanatory power would be meaningless.
    folds = min(5, len(table) // 2)
    if folds >= 2:
        predicted = cross_val_predict(
            Ridge(alpha=alpha), X, y, cv=KFold(folds, shuffle=True, random_state=0)
        )
        ss_res = float(((y - predicted) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r_squared = 1 - ss_res / ss_tot if ss_tot else 0.0
    else:
        r_squared = 0.0

    contributions = pd.DataFrame(
        {
            "position": [p for p, _ in terms],
            "substituent": [s for _, s in terms],
            "contribution": np.round(model.coef_, 3),
            "n_compounds": [int((table[p] == s).sum()) for p, s in terms],
        }
    ).sort_values("contribution", ascending=False).reset_index(drop=True)

    return FreeWilsonModel(
        r_squared=max(r_squared, 0.0),
        n_compounds=len(table),
        n_terms=len(terms),
        intercept=float(model.intercept_),
        contributions=contributions,
    )
