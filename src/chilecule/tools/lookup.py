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


@dataclass
class PromiscuityReport:
    """What a compound has been reported to hit, across every ChEMBL target."""

    chembl_id: str | None
    n_targets: int = 0
    n_records: int = 0
    targets: list[dict] = field(default_factory=list)
    n_non_protein_excluded: int = 0
    error: str | None = None

    @property
    def selectivity_window(self) -> float | None:
        """Log units between the best target and the median of the others.

        This is the signal that the raw target count is not. A compound with a
        3-log window has a primary target and a long tail of weak off-target
        annotations, which is what a selective inhibitor looks like after it
        has been through a panel. A compound hitting fifteen unrelated proteins
        all within half a log of each other is not selective for anything, and
        that flat profile is the signature of an aggregator, a reactive
        electrophile, or assay interference.

        It needs no target classification, which is why it is used here: the
        obvious alternative -- bucketing targets into families -- requires
        either a per-target API call or keyword matching on target names, and
        keyword matching puts EGFR in the GPCR bucket.
        """
        potencies = sorted((t["best_pchembl"] for t in self.targets), reverse=True)
        if len(potencies) < 3:
            return None
        rest = potencies[1:]
        median = rest[len(rest) // 2] if len(rest) % 2 else (
            (rest[len(rest) // 2 - 1] + rest[len(rest) // 2]) / 2
        )
        return round(potencies[0] - median, 2)

    @property
    def assessment(self) -> str:
        """Read the target list the way a chemist would.

        The raw count is close to useless on its own. Erlotinib has over a
        hundred reported protein targets and is not promiscuous in any
        troubling sense -- it is a kinase inhibitor that has been through
        kinome panels, and a panel result is an annotation, not a liability.

        What separates that from a genuine frequent hitter is whether there is
        a primary target at all. See :attr:`selectivity_window`.
        """
        if self.error:
            return "no promiscuity data retrieved"
        if self.n_targets == 0:
            return (
                "no protein bioactivity records -- meaning this compound has not been "
                "tested, not that it is clean"
            )
        if self.n_targets == 1:
            return "single reported protein target"

        window = self.selectivity_window
        head = f"{self.n_targets} distinct protein targets with pChEMBL >= 5"
        if self.n_non_protein_excluded:
            head += (
                f" ({self.n_non_protein_excluded} cell-line and other non-protein "
                "entries excluded)"
            )

        if window is None:
            return f"{head}. Too few targets to judge selectivity."
        if window >= 2.0:
            return (
                f"{head}. Clear primary target: {window} log units better than the median "
                "of the rest, so the long tail is panel annotation rather than promiscuity."
            )
        if window >= 1.0:
            return (
                f"{head}. Moderate selectivity window ({window} log units over the median "
                "of the others). Ordinary polypharmacology; worth knowing what else is hit."
            )
        return (
            f"{head}, and no primary target stands out -- the best is only {window} log "
            "units above the median of the rest. A flat profile across many proteins is "
            "the signature of an aggregator, a reactive compound, or assay interference. "
            "Run a detergent control and a counterscreen before believing any single result."
        )

    def to_dict(self) -> dict:
        return {
            "chembl_id": self.chembl_id,
            "n_protein_targets": self.n_targets,
            "n_non_protein_excluded": self.n_non_protein_excluded,
            "selectivity_window_log": self.selectivity_window,
            "n_activity_records": self.n_records,
            "assessment": self.assessment,
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
    otherwise make every compound look promiscuous.

    Only ``SINGLE PROTEIN`` and ``PROTEIN COMPLEX`` targets are counted, and
    the target types are looked up from ChEMBL rather than guessed from names.
    This matters more than it sounds: a large share of a well-studied
    compound's activity records are cytotoxicity measurements against cell
    lines -- MCF7, A549, HepG2 -- which ChEMBL stores as targets. Counting
    those makes every oncology compound look wildly promiscuous when what has
    actually happened is that someone ran an NCI-60 panel.

    Orthologs are collapsed. The question is how many distinct proteins the
    compound hits, not how many species it was measured in.
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
    seen_target_ids: set[str] = set()
    for activity in activities:
        name = activity.get("target_pref_name")
        organism = activity.get("target_organism")
        target_id = activity.get("target_chembl_id")
        if not name or not target_id or name in {"NON-PROTEIN TARGET", "Unchecked"}:
            continue
        try:
            pchembl = float(activity.get("pchembl_value"))
        except (TypeError, ValueError):
            continue
        if pchembl < min_pchembl:
            continue
        seen_target_ids.add(target_id)
        key = (name, organism, target_id)
        counts[key] += 1
        potency[key] = max(potency.get(key, 0.0), pchembl)

    protein_targets = _protein_target_ids(seen_target_ids, timeout)
    if protein_targets is not None:
        excluded = {k for k in counts if k[2] not in protein_targets}
        report.n_non_protein_excluded = len({k[0] for k in excluded})
        for key in excluded:
            del counts[key]
            potency.pop(key, None)

    # Count distinct target NAMES, not (name, organism) pairs. Human and sheep
    # COX-1 are one target for this purpose, and counting orthologs separately
    # inflates the promiscuity of exactly the well-studied compounds that get
    # tested in several species -- aspirin scores 16 that way and 9 correctly.
    report.n_targets = len({name for name, _organism, _id in counts})
    report.targets = [
        {
            "target": name,
            "organism": organism,
            "target_chembl_id": target_id,
            "n_measurements": n,
            "best_pchembl": round(potency[(name, organism, target_id)], 2),
        }
        for (name, organism, target_id), n in counts.most_common(25)
    ]
    return report


# ChEMBL target types that represent a protein whose inhibition means something
# structurally. CELL-LINE, ORGANISM, TISSUE and the rest are phenotypic
# readouts and are excluded from a selectivity assessment.
PROTEIN_TARGET_TYPES = {"SINGLE PROTEIN", "PROTEIN COMPLEX", "PROTEIN FAMILY",
                        "PROTEIN COMPLEX GROUP", "CHIMERIC PROTEIN"}


def _protein_target_ids(target_ids: set[str], timeout: int) -> set[str] | None:
    """Which of these ChEMBL targets are proteins, in one batched call.

    Returns None if the lookup fails, so that a ChEMBL outage degrades to an
    unfiltered count rather than to an empty result.
    """
    if not target_ids:
        return set()
    proteins: set[str] = set()
    ids = sorted(target_ids)
    # The API caps the URL length, so batch rather than sending 200 ids at once.
    for start in range(0, len(ids), 50):
        chunk = ids[start : start + 50]
        try:
            payload = _get(
                f"{CHEMBL_API}/target.json",
                {"target_chembl_id__in": ",".join(chunk), "limit": len(chunk)},
                timeout,
            )
        except requests.RequestException:
            return None
        for target in (payload or {}).get("targets", []):
            if target.get("target_type") in PROTEIN_TARGET_TYPES:
                proteins.add(target.get("target_chembl_id"))
    return proteins


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
