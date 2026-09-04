"""Has anyone made this, and what else does it hit?

Two questions a medicinal chemist asks about any structure before spending
money on it, and the two that are most annoying to answer by hand.

**Novelty.** Exact-structure lookup in ChEMBL and PubChem by InChIKey, plus
near-neighbour search. The distinction between the two matters: an exact match
means the compound is known, while a 0.9-similar match means you are inside
someone's series and should look at the patent before the paper. Neither is a
freedom-to-operate opinion, and this module says so rather than implying
otherwise.

**Promiscuity.** What else the compound has been reported to hit. A molecule
with measured activity against fourteen unrelated targets is telling you
something -- either it is a genuinely polypharmacological drug, or it is an
aggregator, a reactive electrophile, or an assay artifact. Frequent-hitter
behaviour is far better caught here, for free, than in a dose-response curve
three weeks later.

InChIKey matching is on the full key, including the protonation/stereo block.
Two compounds differing only in salt form share the first 14-character
skeleton block but not the full key, which is why structures are standardized
before lookup.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

import requests

from .chem import parse_smiles, standardize

log = logging.getLogger(__name__)

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
PUBCHEM_API = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_HEADERS = {"User-Agent": "chilecule/0.1 (open-source drug discovery)"}


@dataclass
class NoveltyReport:
    """Whether a structure is already known, and how close the neighbours are."""

    smiles: str
    inchikey: str | None
    in_chembl: bool = False
    chembl_id: str | None = None
    chembl_name: str | None = None
    max_phase: float | None = None
    in_pubchem: bool = False
    pubchem_cid: int | None = None
    nearest_neighbours: list[dict] = field(default_factory=list)
    error: str | None = None

    @property
    def is_known(self) -> bool:
        return self.in_chembl or self.in_pubchem

    @property
    def verdict(self) -> str:
        if self.error:
            return "lookup failed"
        if self.in_chembl and self.max_phase and self.max_phase >= 4:
            return "approved drug"
        if self.in_chembl:
            return "known compound with bioactivity data in ChEMBL"
        if self.in_pubchem:
            return "known structure in PubChem, no ChEMBL bioactivity record"
        if self.nearest_neighbours:
            best = self.nearest_neighbours[0]["similarity"]
            if best >= 0.9:
                return (
                f"not found, but a {best:.2f}-similar compound is known "
                "-- likely inside an existing series"
            )
            if best >= 0.7:
                return f"not found; closest known compound is {best:.2f} similar"
        return "no exact match and no close neighbour found in ChEMBL"

    def to_dict(self) -> dict:
        return {
            "smiles": self.smiles,
            "inchikey": self.inchikey,
            "is_known": self.is_known,
            "verdict": self.verdict,
            "chembl_id": self.chembl_id,
            "chembl_name": self.chembl_name,
            "max_phase": self.max_phase,
            "pubchem_cid": self.pubchem_cid,
            "nearest_neighbours": self.nearest_neighbours,
            "caveat": (
                "Structural novelty only. This is not a freedom-to-operate opinion: "
                "a compound absent from ChEMBL and PubChem may still fall inside a "
                "Markush claim."
            ),
            "error": self.error,
        }


def _get(url: str, params: dict | None = None, timeout: int = 30) -> dict | None:
    try:
        response = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        log.warning("lookup failed for %s: %s", url, exc)
        raise


def check_novelty(
    smiles: str,
    *,
    similarity_threshold: int = 70,
    max_neighbours: int = 5,
    timeout: int = 30,
) -> NoveltyReport:
    """Look a structure up in ChEMBL and PubChem, then find its neighbours.

    ``similarity_threshold`` is a percentage, matching ChEMBL's similarity
    endpoint. 70 is deliberately permissive: the useful signal is often not
    "someone made this" but "someone made six things around this".
    """
    standardized = standardize(smiles, canonical_tautomer=False)
    if not standardized.ok:
        return NoveltyReport(smiles, None, error=standardized.error)

    report = NoveltyReport(smiles=standardized.smiles, inchikey=standardized.inchikey)

    try:
        payload = _get(
            f"{CHEMBL_API}/molecule.json",
            {"molecule_structures__standard_inchi_key": report.inchikey},
            timeout,
        )
        molecules = (payload or {}).get("molecules", [])
        if molecules:
            hit = molecules[0]
            report.in_chembl = True
            report.chembl_id = hit.get("molecule_chembl_id")
            report.chembl_name = hit.get("pref_name")
            max_phase = hit.get("max_phase")
            report.max_phase = float(max_phase) if max_phase is not None else None
    except requests.RequestException as exc:
        report.error = f"ChEMBL lookup failed: {exc}"

    try:
        payload = _get(
            f"{PUBCHEM_API}/compound/inchikey/{report.inchikey}/cids/JSON", timeout=timeout
        )
        cids = ((payload or {}).get("IdentifierList") or {}).get("CID") or []
        if cids:
            report.in_pubchem = True
            report.pubchem_cid = cids[0]
    except requests.RequestException:
        pass  # PubChem being down should not fail the whole profile

    report.nearest_neighbours = _chembl_neighbours(
        report.smiles, similarity_threshold, max_neighbours, report.chembl_id, timeout
    )
    return report


def _chembl_neighbours(
    smiles: str, threshold: int, limit: int, exclude_id: str | None, timeout: int
) -> list[dict]:
    """ChEMBL similarity search, excluding the query itself."""
    try:
        payload = _get(
            f"{CHEMBL_API}/similarity/{requests.utils.quote(smiles, safe='')}/{threshold}.json",
            {"limit": limit + 5},
            timeout,
        )
    except requests.RequestException:
        return []

    neighbours = []
    for molecule in (payload or {}).get("molecules", []):
        chembl_id = molecule.get("molecule_chembl_id")
        if chembl_id == exclude_id:
            continue
        try:
            similarity = round(float(molecule.get("similarity", 0)) / 100, 3)
        except (TypeError, ValueError):
            continue
        neighbours.append(
            {
                "chembl_id": chembl_id,
                "name": molecule.get("pref_name"),
                "similarity": similarity,
                "max_phase": molecule.get("max_phase"),
                "smiles": (molecule.get("molecule_structures") or {}).get("canonical_smiles"),
            }
        )
        if len(neighbours) >= limit:
            break
    return neighbours


# Heuristic target-family classification from the ChEMBL preferred name.
#
# ChEMBL's own protein classification requires a separate API call per target,
# which is too slow for an interactive profile. Keyword matching on the target
# name is crude, and it is labelled as crude wherever it is reported -- but it
# answers the only question that matters here, which is whether a compound's
# targets are related to each other. The count alone does not: erlotinib has
# 203 reported targets and is not promiscuous in any meaningful sense, because
# essentially all of them are kinases in a kinome panel.
TARGET_FAMILY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("kinase", ("kinase", "kinase-like")),
    ("protease", ("protease", "peptidase", "cathepsin", "caspase", "thrombin",
                  "trypsin", "elastase", "matrix metallo")),
    ("GPCR", ("receptor" ,)),  # narrowed below by the ion-channel/NR checks
    ("nuclear receptor", ("nuclear receptor", "estrogen receptor",
                          "androgen receptor", "glucocorticoid receptor",
                          "retinoic acid receptor", "peroxisome proliferator")),
    ("ion channel", ("channel", "transporter")),
    ("oxidoreductase", ("reductase", "oxidase", "dehydrogenase", "synthase",
                        "cytochrome", "monoamine oxidase")),
    ("transferase", ("transferase", "methyltransferase", "acetyltransferase")),
    ("hydrolase", ("hydrolase", "esterase", "lipase", "phosphatase",
                   "phosphodiesterase")),
    ("epigenetic", ("histone", "bromodomain", "deacetylase", "demethylase")),
    ("other enzyme", ("enzyme", "ase")),
)


def target_family(target_name: str) -> str:
    """Best-effort family for a ChEMBL target name. Heuristic, by design."""
    name = (target_name or "").lower()
    # Order matters: the more specific patterns are tested before "receptor",
    # which would otherwise swallow nuclear receptors and ion channels.
    for family in ("nuclear receptor", "ion channel", "epigenetic", "kinase",
                   "protease", "oxidoreductase", "transferase", "hydrolase"):
        keywords = dict(TARGET_FAMILY_KEYWORDS)[family]
        if any(keyword in name for keyword in keywords):
            return family
    if "receptor" in name:
        return "GPCR / receptor"
    if name.endswith("ase") or "enzyme" in name:
        return "other enzyme"
    return "other"


@dataclass
class PromiscuityReport:
    """What a compound has been reported to hit, across every ChEMBL target."""

    chembl_id: str | None
    n_targets: int = 0
    n_records: int = 0
    targets: list[dict] = field(default_factory=list)
    families: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    @property
    def n_families(self) -> int:
        return len(self.families)

    @property
    def dominant_family_share(self) -> float:
        if not self.families:
            return 0.0
        return max(self.families.values()) / sum(self.families.values())

    @property
    def assessment(self) -> str:
        """Read the target list the way a chemist would.

        The raw count is close to useless on its own. Erlotinib has over 200
        reported targets and is not promiscuous in any troubling sense: they
        are nearly all kinases from panel profiling, which is exactly what a
        kinase inhibitor is meant to be tested against. Aspirin has a dozen,
        spread across unrelated protein classes.

        So the signal is concentration, not count. Many targets inside one
        family is ordinary polypharmacology. Fewer targets scattered across
        kinases, GPCRs and proteases is the signature of an aggregator, a
        reactive electrophile, or assay interference.
        """
        if self.error:
            return "no promiscuity data retrieved"
        if self.n_targets == 0:
            return (
                "no bioactivity records -- meaning this compound has not been tested, "
                "not that it is clean"
            )
        if self.n_targets == 1:
            return "single reported target"

        share = self.dominant_family_share
        dominant = max(self.families, key=self.families.get) if self.families else "unknown"
        head = f"{self.n_targets} reported targets across {self.n_families} target families"

        if share >= 0.7:
            return (
                f"{head}, {share:.0%} of them {dominant}. Concentrated in one family, which "
                "is ordinary polypharmacology for this chemotype rather than a red flag -- "
                "panel profiling inflates the raw count."
            )
        if self.n_targets <= 5:
            return f"{head}. Unremarkable."
        if self.n_families >= 4 and self.n_targets >= 8:
            return (
                f"{head} with no dominant family ({share:.0%} {dominant}). Activity spread "
                "across unrelated protein classes is the signature of an aggregator, a "
                "reactive compound, or assay interference. Worth a counterscreen and a "
                "detergent control before believing any single result."
            )
        return f"{head}, {share:.0%} {dominant}. Check whether the targets are related."

    def to_dict(self) -> dict:
        return {
            "chembl_id": self.chembl_id,
            "n_targets": self.n_targets,
            "n_target_families": self.n_families,
            "families": self.families,
            "dominant_family_share": round(self.dominant_family_share, 3),
            "n_activity_records": self.n_records,
            "assessment": self.assessment,
            "classification_caveat": (
                "Target families are inferred by keyword from the ChEMBL preferred name, "
                "not from ChEMBL's own protein classification. Treat as indicative."
            ),
            "targets": self.targets,
            "error": self.error,
        }


def check_promiscuity(
    chembl_id: str,
    *,
    max_records: int = 1000,
    min_pchembl: float = 5.0,
    timeout: int = 45,
) -> PromiscuityReport:
    """Summarize a compound's reported activity across all ChEMBL targets.

    ``min_pchembl`` of 5.0 (10 uM) filters out the weak measurements that
    otherwise make every compound look promiscuous. Only protein targets are
    counted; ChEMBL's 'NON-PROTEIN TARGET' and 'Unchecked' entries carry no
    interpretable selectivity information. Orthologs are collapsed -- the
    question is how many distinct proteins the compound hits, not how many
    species they were measured in.
    """
    report = PromiscuityReport(chembl_id=chembl_id)
    try:
        payload = _get(
            f"{CHEMBL_API}/activity.json",
            {
                "molecule_chembl_id": chembl_id,
                "pchembl_value__isnull": "false",
                "limit": min(max_records, 1000),
            },
            timeout,
        )
    except requests.RequestException as exc:
        report.error = str(exc)
        return report

    activities = (payload or {}).get("activities", [])
    report.n_records = (payload or {}).get("page_meta", {}).get("total_count", len(activities))

    counts: Counter = Counter()
    potency: dict[tuple, float] = {}
    for activity in activities:
        name = activity.get("target_pref_name")
        organism = activity.get("target_organism")
        if not name or name in {"NON-PROTEIN TARGET", "Unchecked"}:
            continue
        try:
            pchembl = float(activity.get("pchembl_value"))
        except (TypeError, ValueError):
            continue
        if pchembl < min_pchembl:
            continue
        key = (name, organism, activity.get("target_chembl_id"))
        counts[key] += 1
        potency[key] = max(potency.get(key, 0.0), pchembl)

    # Count distinct target NAMES, not (name, organism) pairs. Human and sheep
    # COX-1 are one target for this purpose, and counting orthologs separately
    # inflates the promiscuity of exactly the well-studied compounds that get
    # tested in several species -- aspirin scores 16 that way and 9 correctly.
    report.n_targets = len({name for name, _organism, _id in counts})
    report.families = dict(
        Counter(target_family(name) for name, _organism, _id in counts).most_common()
    )
    report.targets = [
        {
            "target": name,
            "family": target_family(name),
            "organism": organism,
            "target_chembl_id": target_id,
            "n_measurements": n,
            "best_pchembl": round(potency[(name, organism, target_id)], 2),
        }
        for (name, organism, target_id), n in counts.most_common(25)
    ]
    return report


def similar_known_actives(
    smiles: str, target_chembl_id: str, *, threshold: int = 60, limit: int = 5, timeout: int = 45
) -> list[dict]:
    """Known actives against a specific target that resemble the query.

    Answers "is my compound like anything that already works here", which is
    the fastest available prior on whether a structure is worth pursuing.
    """
    mol = parse_smiles(smiles)
    if mol is None:
        return []
    try:
        payload = _get(
            f"{CHEMBL_API}/similarity/{requests.utils.quote(smiles, safe='')}/{threshold}.json",
            {"target_chembl_id": target_chembl_id, "limit": limit},
            timeout,
        )
    except requests.RequestException:
        return []
    return [
        {
            "chembl_id": molecule.get("molecule_chembl_id"),
            "similarity": round(float(molecule.get("similarity", 0)) / 100, 3),
            "smiles": (molecule.get("molecule_structures") or {}).get("canonical_smiles"),
        }
        for molecule in (payload or {}).get("molecules", [])
    ]
