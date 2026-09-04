"""Property-matched decoy selection, and a check on whether the decoys are fair.

Retrospective validation needs negatives. There are two sources, and they are
not equivalent:

1. **Confirmed inactives** -- compounds actually measured against the target
   and found inactive (ChEMBL's right-censored ``>10 uM`` records). Real
   experimental outcomes, but scarce and biased toward compounds someone had a
   reason to test. They are *not* automatically a fair negative set: running
   :func:`decoy_bias_report` on EGFR's ChEMBL actives against its confirmed
   inactives returns AUC 0.96 from physicochemical descriptors alone, because
   the two groups come from different papers pursuing different chemotypes.
   Being experimentally real does not make a negative set unbiased.

2. **Property-matched decoys** -- molecules assumed inactive because nobody
   has tested them, chosen to match the actives' physicochemical profile while
   being topologically dissimilar. Plentiful, but only as good as the matching.

The matching is the whole game. If decoys differ from actives in molecular
weight or logP, then a benchmark "validating" a docking protocol is really
measuring whether the protocol prefers heavy molecules -- and docking scores
famously correlate with size. Published benchmark sets including DUD-E have
been shown to carry exactly this bias, to the point where a model trained on
DUD-E can separate actives from decoys without seeing the protein at all
(Chen et al., J Chem Inf Model 2019; Sieg et al., J Chem Inf Model 2019).

So this module ships :func:`decoy_bias_report`, which trains a classifier on
physicochemical descriptors alone and reports its AUC. If that number is much
above 0.5, the negative set is broken and any result built on it is
uninterpretable -- and as the EGFR case above shows, that applies to real
experimental inactives just as much as to synthetic decoys. Running this
before reporting a screening result is the difference between a benchmark and
a press release.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import DataStructs

from ..tools.chem import parse_smiles, properties
from ..tools.sar import fingerprint

# Descriptors matched between actives and decoys, following the DUD-E recipe.
MATCH_PROPERTIES = ("mw", "clogp", "hbd", "hba", "rotatable_bonds", "formal_charge")

# Tolerance windows for a decoy to count as "matched" to an active.
DEFAULT_TOLERANCES: dict[str, float] = {
    "mw": 25.0,
    "clogp": 1.0,
    "hbd": 1.0,
    "hba": 2.0,
    "rotatable_bonds": 2.0,
    "formal_charge": 0.0,  # charge state must match exactly
}

# Maximum ECFP4 Tanimoto between a decoy and any active. Above this the "decoy"
# may well be active, and counting it as a negative penalizes a method for
# being right.
MAX_SIMILARITY_TO_ACTIVES = 0.35


@dataclass
class DecoySet:
    decoys: pd.DataFrame
    n_requested: int
    n_selected: int
    unmatched_actives: list[str]

    def to_dict(self) -> dict:
        return {
            "n_requested": self.n_requested,
            "n_selected": self.n_selected,
            "n_unmatched_actives": len(self.unmatched_actives),
            "coverage": round(self.n_selected / self.n_requested, 3) if self.n_requested else 0.0,
        }


def describe(smiles_list: list[str]) -> pd.DataFrame:
    """Descriptor table for a list of SMILES, skipping unparseable entries."""
    rows = []
    for smiles in smiles_list:
        props = properties(smiles)
        if props is not None:
            rows.append(props.to_dict())
    return pd.DataFrame(rows)


def property_matched_decoys(
    actives: pd.DataFrame,
    background: pd.DataFrame,
    *,
    smiles_col: str = "smiles",
    n_per_active: int = 30,
    tolerances: dict[str, float] | None = None,
    max_similarity: float = MAX_SIMILARITY_TO_ACTIVES,
    seed: int = 0,
) -> DecoySet:
    """Select decoys from ``background`` that match ``actives`` on properties.

    For each active, finds background molecules inside the tolerance window on
    every matched descriptor, discards any that are topologically similar to
    *any* active, and samples up to ``n_per_active`` of the survivors.
    """
    tolerances = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    rng = np.random.default_rng(seed)

    active_props = describe(actives[smiles_col].tolist())
    background_props = describe(background[smiles_col].tolist())
    if active_props.empty or background_props.empty:
        return DecoySet(pd.DataFrame(), 0, 0, [])

    # Fingerprint every active once; the similarity veto is the expensive step.
    active_fps = [fingerprint(m) for m in
                  (parse_smiles(s) for s in active_props["smiles"]) if m is not None]

    chosen: set[int] = set()
    unmatched: list[str] = []
    n_requested = len(active_props) * n_per_active

    for _, active in active_props.iterrows():
        mask = pd.Series(True, index=background_props.index)
        for prop in MATCH_PROPERTIES:
            mask &= (background_props[prop] - active[prop]).abs() <= tolerances[prop]

        candidates = [i for i in background_props.index[mask] if i not in chosen]
        if not candidates:
            unmatched.append(active["smiles"])
            continue

        rng.shuffle(candidates)
        picked = 0
        for idx in candidates:
            if picked >= n_per_active:
                break
            mol = parse_smiles(background_props.at[idx, "smiles"])
            if mol is None:
                continue
            sims = DataStructs.BulkTanimotoSimilarity(fingerprint(mol), active_fps)
            if max(sims) > max_similarity:
                continue  # too close to a known active to be a safe negative
            chosen.add(idx)
            picked += 1
        if picked == 0:
            unmatched.append(active["smiles"])

    decoys = background_props.loc[sorted(chosen)].reset_index(drop=True)
    return DecoySet(decoys, n_requested, len(decoys), unmatched)


@dataclass
class BiasReport:
    """Result of trying to separate actives from decoys using properties alone."""

    auc: float
    n_actives: int
    n_decoys: int
    top_features: list[tuple[str, float]]
    verdict: str
    detail: str

    def to_dict(self) -> dict:
        return {
            "property_only_auc": round(self.auc, 3),
            "n_actives": self.n_actives,
            "n_decoys": self.n_decoys,
            "top_features": [
                {"feature": f, "importance": round(v, 3)} for f, v in self.top_features
            ],
            "verdict": self.verdict,
            "detail": self.detail,
        }

    def summary(self) -> str:
        features = ", ".join(f"{f} ({v:.2f})" for f, v in self.top_features[:3])
        return (
            f"[{self.verdict}] property-only AUC = {self.auc:.3f} "
            f"({self.n_actives} actives vs {self.n_decoys} decoys)\n"
            f"  most discriminating descriptors: {features}\n"
            f"  {self.detail}"
        )


def decoy_bias_report(
    actives: pd.DataFrame,
    decoys: pd.DataFrame,
    smiles_col: str = "smiles",
    n_splits: int = 5,
    seed: int = 0,
) -> BiasReport:
    """Test whether actives and decoys are separable from descriptors alone.

    Trains a small random forest on physicochemical descriptors under
    cross-validation and reports the AUC. Interpretation:

    * **AUC near 0.5** -- the decoys are well matched. A screening method that
      beats chance on this set is using structural or structure-based
      information, which is the thing you set out to measure.
    * **AUC above ~0.65** -- the sets are separable without any knowledge of
      the target. Enrichment measured on this set is partly or wholly an
      artifact, and the reported numbers do not mean what they appear to. The
      fix is tighter property matching, not a different scoring function.

    This is the check that most published virtual-screening validations omit,
    and it is the reason so many of their enrichment numbers fail to reproduce
    prospectively.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold
    except ImportError:  # pragma: no cover - optional extra
        return BiasReport(
            float("nan"), len(actives), len(decoys), [],
            "unavailable",
            "scikit-learn is required for the bias check and is a core "
            "dependency; reinstall with: pip install -e .",
        )

    active_props = describe(actives[smiles_col].tolist())
    decoy_props = describe(decoys[smiles_col].tolist())
    if len(active_props) < 10 or len(decoy_props) < 10:
        return BiasReport(
            float("nan"), len(active_props), len(decoy_props), [],
            "insufficient-data",
            "need at least 10 molecules per class for a meaningful bias check",
        )

    feature_cols = [
        "mw", "clogp", "tpsa", "hbd", "hba", "rotatable_bonds",
        "aromatic_rings", "heavy_atoms", "fraction_csp3", "formal_charge", "qed",
    ]
    X = np.vstack([
        active_props[feature_cols].to_numpy(dtype=float),
        decoy_props[feature_cols].to_numpy(dtype=float),
    ])
    y = np.concatenate([np.ones(len(active_props)), np.zeros(len(decoy_props))]).astype(int)

    from .metrics import roc_auc

    aucs, importances = [], []
    splitter = StratifiedKFold(n_splits=min(n_splits, int(y.sum()), int((y == 0).sum())),
                               shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(X, y):
        model = RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=seed, n_jobs=-1,
            class_weight="balanced",
        )
        model.fit(X[train_idx], y[train_idx])
        aucs.append(roc_auc(model.predict_proba(X[test_idx])[:, 1], y[test_idx]))
        importances.append(model.feature_importances_)

    auc = float(np.mean(aucs))
    mean_importance = np.mean(importances, axis=0)
    ranked = sorted(
        zip(feature_cols, mean_importance, strict=True), key=lambda kv: -kv[1]
    )

    if auc < 0.60:
        verdict, detail = "PASS", (
            "Decoys are well matched; enrichment on this set reflects the method "
            "under test rather than a property artifact."
        )
    elif auc < 0.70:
        verdict, detail = "MARGINAL", (
            "Partial separability from descriptors alone. Report enrichment with "
            "this caveat stated, and prefer confirmed inactives where available."
        )
    else:
        verdict, detail = "FAIL", (
            f"Actives and decoys are separable at AUC {auc:.2f} without any "
            "target information. Enrichment measured on this set is an artifact; "
            "tighten the matching tolerances or use confirmed inactives."
        )

    return BiasReport(auc, len(active_props), len(decoy_props), ranked[:5], verdict, detail)
