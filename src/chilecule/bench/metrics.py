"""Early-recognition metrics for virtual screening.

A screening cascade is only worth running if it ranks actives above inactives.
That claim is testable, and this module is how the claim gets tested.

Why not ROC-AUC alone: ROC-AUC weights every position in the ranking equally.
In a real campaign nobody buys compound 40,000 of 50,000, so improving that
compound's rank is worth nothing while improving rank 12 is worth a great deal.
A method can post a respectable 0.75 AUC while putting no actives in the top
1%, which is the only region anyone screens. Early-recognition metrics -- EF
and BEDROC -- measure the region that matters.

References:
    Truchon & Bayly, J Chem Inf Model 2007, 47, 488-508 (BEDROC, RIE)
    Halgren et al., J Med Chem 2004, 47, 1750-1759 (enrichment factor)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScreeningMetrics:
    """Retrospective performance of a ranking over a labelled set."""

    n_total: int
    n_actives: int
    active_fraction: float
    roc_auc: float
    bedroc_alpha20: float
    enrichment: dict[str, float]
    hit_rate_top1pct: float

    def to_dict(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_actives": self.n_actives,
            "active_fraction": round(self.active_fraction, 4),
            "roc_auc": round(self.roc_auc, 3),
            "bedroc_alpha20": round(self.bedroc_alpha20, 3),
            "enrichment_factor": {k: round(v, 2) for k, v in self.enrichment.items()},
            "hit_rate_top1pct": round(self.hit_rate_top1pct, 4),
        }

    def summary(self) -> str:
        ef = " ".join(f"EF{k}={v:.1f}" for k, v in self.enrichment.items())
        return (
            f"{self.n_actives}/{self.n_total} actives ({self.active_fraction:.2%})  "
            f"AUC={self.roc_auc:.3f}  BEDROC(a=20)={self.bedroc_alpha20:.3f}  {ef}"
        )


def _pessimistic_order(
    scores: np.ndarray, labels: np.ndarray, higher_is_better: bool
) -> np.ndarray:
    """Ranking order in which actives lose every tie.

    Sorting on score alone leaves tie blocks in input order, and datasets are
    routinely stored actives-first. A scoring function that returns a constant
    would then appear to place every active at the top of the list and post a
    maximal enrichment factor -- from a function carrying no information at all.

    ``np.lexsort`` applies the LAST key first, so the primary sort is the score
    and the tiebreak is the label, ascending, putting inactives ahead of actives
    within each block.
    """
    primary = -scores if higher_is_better else scores
    return np.lexsort((labels, primary))


def _ranks_of_actives(scores: np.ndarray, labels: np.ndarray, higher_is_better: bool) -> np.ndarray:
    """1-indexed ranks of the actives under the given ranking.

    Ties are broken pessimistically -- an active tied with inactives is placed
    after them. Optimistic tie-breaking is the most common way a benchmark
    silently flatters itself, because scoring functions produce ties constantly
    (identical docking scores, identical filter verdicts) and ranking actives
    first within a tie block can manufacture enrichment out of nothing.
    """
    order = _pessimistic_order(scores, labels, higher_is_better)
    ordered_labels = labels[order]
    ordered_scores = scores[order]

    ranks = np.arange(1, len(scores) + 1, dtype=float)
    # Within each tie block, assign every member the worst rank in the block.
    start = 0
    for i in range(1, len(ordered_scores) + 1):
        if i == len(ordered_scores) or ordered_scores[i] != ordered_scores[start]:
            ranks[start:i] = i
            start = i
    return ranks[ordered_labels == 1]


def enrichment_factor(
    scores: np.ndarray, labels: np.ndarray, fraction: float = 0.01,
    higher_is_better: bool = True,
) -> float:
    """Enrichment factor at the top ``fraction`` of the ranked list.

    EF = (hit rate in the top slice) / (hit rate of the full set). EF = 1 is
    indistinguishable from random. The theoretical maximum is 1/fraction when
    the slice is smaller than the number of actives, so an EF of 100 at 1% is
    a perfect result, not an impossible one -- always report the ceiling
    alongside the value.
    """
    n = len(scores)
    n_actives = int(labels.sum())
    if n == 0 or n_actives == 0:
        return float("nan")

    cut = max(1, int(round(n * fraction)))
    order = _pessimistic_order(
        np.asarray(scores, dtype=float), np.asarray(labels, dtype=int), higher_is_better
    )
    found = int(np.asarray(labels, dtype=int)[order][:cut].sum())
    return (found / cut) / (n_actives / n)


def max_enrichment_factor(n_total: int, n_actives: int, fraction: float) -> float:
    """Ceiling on EF at this fraction, given the composition of the set."""
    cut = max(1, int(round(n_total * fraction)))
    return (min(cut, n_actives) / cut) / (n_actives / n_total)


def bedroc(
    scores: np.ndarray, labels: np.ndarray, alpha: float = 20.0,
    higher_is_better: bool = True,
) -> float:
    """Boltzmann-enhanced discrimination of ROC.

    An exponentially weighted early-recognition metric bounded on [0, 1].
    alpha = 20 places roughly 80% of the total weight in the top 8% of the
    ranked list, which is the setting used in most published comparisons.

    Unlike EF, BEDROC is continuous in the ranks and does not depend on an
    arbitrary cutoff, so it is the more stable number to optimize against.
    Report both: BEDROC for comparing methods, EF for telling a project team
    how many compounds they would have to buy.
    """
    n = len(scores)
    n_actives = int(labels.sum())
    if n == 0 or n_actives == 0 or n_actives == n:
        return float("nan")

    ra = n_actives / n
    ranks = _ranks_of_actives(np.asarray(scores, dtype=float),
                              np.asarray(labels, dtype=int), higher_is_better)

    rie_numerator = np.sum(np.exp(-alpha * ranks / n)) / n_actives
    rie_denominator = (1.0 / n) * (1 - math.exp(-alpha)) / (math.exp(alpha / n) - 1)
    rie = rie_numerator / rie_denominator

    scale = (ra * math.sinh(alpha / 2)) / (math.cosh(alpha / 2) - math.cosh(alpha / 2 - alpha * ra))
    offset = 1.0 / (1 - math.exp(alpha * (1 - ra)))
    return float(rie * scale + offset)


def roc_auc(scores: np.ndarray, labels: np.ndarray, higher_is_better: bool = True) -> float:
    """ROC-AUC via the Mann-Whitney U identity, with proper tie handling.

    Implemented directly rather than pulled from scikit-learn so that the core
    validation path has no optional dependency -- a benchmark that only runs
    when an extra is installed tends not to get run.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_actives, n_inactives = int(labels.sum()), int((labels == 0).sum())
    if n_actives == 0 or n_inactives == 0:
        return float("nan")

    signed = scores if higher_is_better else -scores
    order = np.argsort(signed, kind="mergesort")
    ranked = signed[order]
    # Average ranks within tie blocks -- the standard correction; without it,
    # a scoring function that returns constants would score 1.0 or 0.0.
    ranks = np.empty(len(signed), dtype=float)
    start = 0
    for i in range(1, len(ranked) + 1):
        if i == len(ranked) or ranked[i] != ranked[start]:
            ranks[start:i] = (start + i + 1) / 2.0
            start = i
    active_rank_sum = ranks[labels[order] == 1].sum()
    return float((active_rank_sum - n_actives * (n_actives + 1) / 2) / (n_actives * n_inactives))


def evaluate(
    scores, labels, higher_is_better: bool = True,
    fractions: tuple[float, ...] = (0.01, 0.05, 0.10),
) -> ScreeningMetrics:
    """Compute the full early-recognition panel for one ranking."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n, n_actives = len(scores), int(labels.sum())

    return ScreeningMetrics(
        n_total=n,
        n_actives=n_actives,
        active_fraction=n_actives / n if n else float("nan"),
        roc_auc=roc_auc(scores, labels, higher_is_better),
        bedroc_alpha20=bedroc(scores, labels, 20.0, higher_is_better),
        enrichment={
            f"{int(f * 100)}%": enrichment_factor(scores, labels, f, higher_is_better)
            for f in fractions
        },
        hit_rate_top1pct=_hit_rate(scores, labels, 0.01, higher_is_better),
    )


def _hit_rate(scores: np.ndarray, labels: np.ndarray, fraction: float,
              higher_is_better: bool) -> float:
    cut = max(1, int(round(len(scores) * fraction)))
    order = _pessimistic_order(scores, labels, higher_is_better)
    return float(labels[order][:cut].mean())
