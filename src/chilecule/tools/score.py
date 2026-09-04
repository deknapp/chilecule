"""Multi-parameter scorecards.

A single number is the wrong output for a compound assessment, and it is what
most tools produce. "0.42" tells a chemist nothing they can act on. What they
need is the same thing a project team asks in a meeting: *which property is
holding this compound back?*

So every scorecard here reports the overall score, the per-property
desirability that produced it, and -- the part that matters -- the **limiting
property**. "Score 0.42, limited by cLogP at 5.8 (target below 4)" is a design
instruction. The number alone is a ranking key.

Desirability functions follow the standard Derringer-Suich shape used in
multi-parameter optimization: 1.0 inside the ideal window, falling linearly to
0.0 at a stated boundary. They are made explicit and overridable rather than
hidden, because the right window depends on the project. An inhaled compound,
an oral drug and a CNS agent do not share a TPSA target, and a scorecard that
pretends otherwise is worse than none.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .chem import properties

Direction = Literal["window", "lower_better", "higher_better"]


@dataclass(frozen=True)
class Criterion:
    """One property, its ideal window, and how fast desirability falls off."""

    name: str
    direction: Direction
    ideal_low: float | None = None
    ideal_high: float | None = None
    hard_low: float | None = None
    hard_high: float | None = None
    weight: float = 1.0
    rationale: str = ""

    def desirability(self, value: float) -> float:
        """Map a measured value onto [0, 1]."""
        if value is None:
            return 0.0
        if self.direction == "lower_better":
            if self.ideal_high is not None and value <= self.ideal_high:
                return 1.0
            if self.hard_high is None or value >= self.hard_high:
                return 0.0
            span = self.hard_high - (self.ideal_high or self.hard_high)
            return 0.0 if span <= 0 else 1.0 - (value - self.ideal_high) / span
        if self.direction == "higher_better":
            if self.ideal_low is not None and value >= self.ideal_low:
                return 1.0
            if self.hard_low is None or value <= self.hard_low:
                return 0.0
            span = (self.ideal_low or self.hard_low) - self.hard_low
            return 0.0 if span <= 0 else (value - self.hard_low) / span
        # window
        if self.ideal_low is not None and self.ideal_high is not None:
            if self.ideal_low <= value <= self.ideal_high:
                return 1.0
        if self.hard_low is not None and value < self.hard_low:
            return 0.0
        if self.hard_high is not None and value > self.hard_high:
            return 0.0
        if self.ideal_low is not None and value < self.ideal_low and self.hard_low is not None:
            span = self.ideal_low - self.hard_low
            return 0.0 if span <= 0 else (value - self.hard_low) / span
        if self.ideal_high is not None and value > self.ideal_high and self.hard_high is not None:
            span = self.hard_high - self.ideal_high
            return 0.0 if span <= 0 else 1.0 - (value - self.ideal_high) / span
        return 1.0

    def target_text(self) -> str:
        if self.direction == "lower_better":
            return f"below {self.ideal_high:g}"
        if self.direction == "higher_better":
            return f"above {self.ideal_low:g}"
        return f"{self.ideal_low:g}-{self.ideal_high:g}"


# Windows are drawn from the standard oral-drug literature -- Lipinski, Veber,
# and the Hughes et al. (Bioorg Med Chem Lett 2008) analysis linking high
# cLogP with low TPSA to in vivo toxicity. They are starting points to be
# overridden per project, not laws.
ORAL = (
    Criterion("mw", "window", 250, 450, 150, 600, 1.0,
              "Oral absorption falls off above ~500 Da; below ~250 there is usually "
              "not enough surface for selective binding."),
    Criterion("clogp", "window", 1.0, 3.5, -1.0, 5.5, 1.5,
              "The single most predictive property for downstream attrition. Above 4, "
              "promiscuity, hERG and metabolic clearance all rise together."),
    Criterion("tpsa", "window", 40, 100, 0, 140, 1.0,
              "Veber: above 140 A^2 oral absorption is unlikely."),
    Criterion("hbd", "lower_better", ideal_high=3, hard_high=6, weight=0.8,
              rationale="Donors cost permeability more than acceptors do."),
    Criterion("rotatable_bonds", "lower_better", ideal_high=7, hard_high=12, weight=0.8,
              rationale="Veber: flexibility reduces oral bioavailability."),
    Criterion("fraction_csp3", "higher_better", ideal_low=0.30, hard_low=0.05, weight=0.7,
              rationale="Escape from flatland: sp3 character tracks with solubility "
              "and clinical success (Lovering et al., J Med Chem 2009)."),
    Criterion("aromatic_rings", "lower_better", ideal_high=3, hard_high=5, weight=0.7,
              rationale="More than three aromatic rings correlates with poor "
              "developability (Ritchie & Macdonald, Drug Discov Today 2009)."),
)

# Fragment-to-lead space. The point of a lead is room to grow, so the windows
# sit deliberately below oral drug space -- optimization adds weight and
# lipophilicity, and a lead that starts at the oral limit has nowhere to go.
LEAD_LIKE = (
    Criterion("mw", "window", 200, 350, 120, 400, 1.0,
              "Leave room for the ~100 Da that optimization typically adds."),
    Criterion("clogp", "window", 0.0, 3.0, -1.0, 4.0, 1.5,
              "Leave lipophilicity headroom; potency is usually bought with it."),
    Criterion("tpsa", "window", 40, 90, 0, 120, 0.8),
    Criterion("hbd", "lower_better", ideal_high=2, hard_high=4, weight=0.7),
    Criterion("rotatable_bonds", "lower_better", ideal_high=5, hard_high=8, weight=0.8),
    Criterion("fraction_csp3", "higher_better", ideal_low=0.35, hard_low=0.05, weight=0.8),
)

PROFILES: dict[str, tuple[Criterion, ...]] = {"oral": ORAL, "lead_like": LEAD_LIKE}


@dataclass
class ScorecardResult:
    smiles: str
    profile: str
    score: float
    components: list[dict] = field(default_factory=list)

    @property
    def limiting(self) -> dict | None:
        """The property costing this compound the most, weight included.

        Reported instead of just the score, because it is the part a chemist
        can act on. The largest weighted shortfall is the change that would
        most improve the compound.
        """
        penalised = [c for c in self.components if c["desirability"] < 1.0]
        if not penalised:
            return None
        return max(penalised, key=lambda c: (1.0 - c["desirability"]) * c["weight"])

    @property
    def verdict(self) -> str:
        limiting = self.limiting
        band = (
            "good" if self.score >= 0.8
            else "acceptable" if self.score >= 0.6
            else "marginal" if self.score >= 0.4
            else "poor"
        )
        head = f"{self.profile} score {self.score:.2f} -- {band}"
        if limiting is None:
            return f"{head}. Every property is inside its target window."
        return (
            f"{head}. Limited by {limiting['property']} at {limiting['value']:g} "
            f"(target {limiting['target']})."
        )

    def to_dict(self) -> dict:
        return {
            "smiles": self.smiles,
            "profile": self.profile,
            "score": round(self.score, 3),
            "verdict": self.verdict,
            "limiting_property": self.limiting["property"] if self.limiting else None,
            "components": self.components,
        }


def score(
    smiles: str,
    profile: str = "oral",
    criteria: tuple[Criterion, ...] | None = None,
) -> ScorecardResult | None:
    """Score a molecule against a multi-parameter profile.

    The overall score is the weighted arithmetic mean of per-property
    desirabilities. Arithmetic rather than geometric on purpose: the geometric
    mean used by some MPO schemes drives the whole score to zero the moment any
    single property is out of range, which is mathematically tidy and useless
    for ranking a series where everything has one flaw. The limiting property
    is reported separately so nothing is hidden by the averaging.
    """
    props = properties(smiles)
    if props is None:
        return None

    criteria = criteria or PROFILES.get(profile)
    if criteria is None:
        raise ValueError(f"unknown profile {profile!r}; available: {sorted(PROFILES)}")

    values = props.to_dict()
    components, total, weights = [], 0.0, 0.0
    for criterion in criteria:
        value = values.get(criterion.name)
        desirability = criterion.desirability(value)
        components.append(
            {
                "property": criterion.name,
                "value": value,
                "desirability": round(desirability, 3),
                "weight": criterion.weight,
                "target": criterion.target_text(),
                "rationale": criterion.rationale,
            }
        )
        total += desirability * criterion.weight
        weights += criterion.weight

    return ScorecardResult(
        smiles=props.smiles,
        profile=profile,
        score=total / weights if weights else 0.0,
        components=components,
    )
