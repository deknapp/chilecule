"""Structural alerts: PAINS, Brenk, NIH, and reactive-group flags.

A deliberate note on how these are reported, because it is where most
automated triage goes wrong:

PAINS filters were derived from six AlphaScreen assays run against a single
compound collection. They flag substructures that were frequently promiscuous
*in that readout*. They are not a verdict on a molecule. Roughly 5% of
approved drugs match a PAINS pattern, and several highly successful chemical
series (curcuminoids aside) contain catechols, rhodanines, or quinones that a
PAINS filter rejects outright.

So this module never silently deletes anything. It annotates, records which
catalog fired and why, and leaves the decision to the caller -- which, in the
agent workflows, means the model has to justify the exclusion in writing and
the excluded set stays in the report. Filtering you cannot audit is worse than
no filtering, because it removes evidence that a hit was ever considered.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import FilterCatalog

from .chem import parse_smiles

# Catalogs bundled with RDKit, all usable under RDKit's BSD-3 license.
CATALOGS: dict[str, FilterCatalog.FilterCatalogParams.FilterCatalogs] = {
    "PAINS_A": FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A,
    "PAINS_B": FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B,
    "PAINS_C": FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C,
    "BRENK": FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK,
    "NIH": FilterCatalog.FilterCatalogParams.FilterCatalogs.NIH,
    "ZINC": FilterCatalog.FilterCatalogParams.FilterCatalogs.ZINC,
}

# PAINS_A is the highest-confidence subset (most enrichment in the original
# analysis); B and C are progressively noisier. Reporting them separately lets
# a caller apply different thresholds instead of treating all PAINS as equal.
DEFAULT_CATALOGS = ("PAINS_A", "PAINS_B", "PAINS_C", "BRENK")

# RDKit tags every catalog entry with a ``FilterSet`` property. These are the
# exact values it emits, which differ in case from the enum names above.
FILTER_SET_TO_CATALOG = {
    "PAINS_A": "PAINS_A",
    "PAINS_B": "PAINS_B",
    "PAINS_C": "PAINS_C",
    "Brenk": "BRENK",
    "NIH": "NIH",
    "ZINC": "ZINC",
}

SEVERITY = {
    "PAINS_A": "high",
    "PAINS_B": "medium",
    "PAINS_C": "low",
    "BRENK": "medium",
    "NIH": "medium",
    "ZINC": "low",
}


@dataclass(frozen=True)
class Alert:
    catalog: str
    description: str
    severity: str
    reference: str = ""


@dataclass(frozen=True)
class AlertReport:
    smiles: str
    alerts: tuple[Alert, ...]

    @property
    def clean(self) -> bool:
        return not self.alerts

    @property
    def max_severity(self) -> str | None:
        for level in ("high", "medium", "low"):
            if any(a.severity == level for a in self.alerts):
                return level
        return None

    def to_dict(self) -> dict:
        return {
            "smiles": self.smiles,
            "clean": self.clean,
            "max_severity": self.max_severity,
            "alerts": [
                {
                    "catalog": a.catalog,
                    "description": a.description,
                    "severity": a.severity,
                    "reference": a.reference,
                }
                for a in self.alerts
            ],
        }


@lru_cache(maxsize=8)
def _catalog(names: tuple[str, ...]) -> FilterCatalog.FilterCatalog:
    params = FilterCatalog.FilterCatalogParams()
    for name in names:
        try:
            params.AddCatalog(CATALOGS[name])
        except KeyError as exc:
            raise ValueError(
                f"unknown catalog {name!r}; available: {sorted(CATALOGS)}"
            ) from exc
    return FilterCatalog.FilterCatalog(params)


def screen(smiles: str, catalogs: tuple[str, ...] = DEFAULT_CATALOGS) -> AlertReport | None:
    """Report every structural alert matching a molecule.

    Returns None only if the SMILES cannot be parsed -- an empty ``alerts``
    tuple means the molecule was screened and matched nothing.
    """
    mol = parse_smiles(smiles)
    if mol is None:
        return None

    catalog = _catalog(tuple(catalogs))
    alerts = []
    for match in catalog.GetMatches(mol):
        name = _catalog_name_of(match)
        alerts.append(
            Alert(
                catalog=name,
                description=match.GetDescription() or "",
                severity=SEVERITY.get(name, "medium"),
                reference=_prop(match, "Reference"),
            )
        )
    return AlertReport(smiles=Chem.MolToSmiles(mol), alerts=tuple(alerts))


def _prop(entry, name: str) -> str:
    try:
        return entry.GetProp(name)
    except KeyError:
        return ""


def _catalog_name_of(entry) -> str:
    """Recover which catalog a match came from.

    RDKit tags each entry with ``FilterSet``. Attribution matters because the
    three PAINS tiers carry very different evidence: a PAINS_A match came from
    a substructure that was promiscuous across hundreds of assay records, while
    PAINS_C entries were fit to a handful. Collapsing them into one "PAINS"
    flag throws away the only thing that makes the filter actionable.
    """
    filter_set = _prop(entry, "FilterSet")
    if filter_set in FILTER_SET_TO_CATALOG:
        return FILTER_SET_TO_CATALOG[filter_set]
    return filter_set.upper() or "UNKNOWN"


def screen_many(
    smiles_list: list[str], catalogs: tuple[str, ...] = DEFAULT_CATALOGS
) -> list[AlertReport | None]:
    return [screen(s, catalogs) for s in smiles_list]
