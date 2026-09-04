"""ChEMBL bioactivity retrieval and curation.

ChEMBL is the backbone of open drug discovery, and it is also the place where
naive pipelines quietly produce garbage. The raw ``activity`` endpoint mixes,
in a single result set:

* binding assays and cell-based functional assays, whose numbers are not
  comparable and should never be pooled into one SAR table;
* human, rat, and mouse orthologs under targets that look identical;
* censored measurements (``>10000 nM``) recorded with a ``standard_value``,
  which a regression model will happily treat as a real number and learn that
  every inactive compound has exactly 10 uM potency;
* the same compound measured a dozen times across a decade of papers, with
  spreads that occasionally exceed two log units;
* records the curators have already flagged as suspect via
  ``data_validity_comment``.

This module fetches, then curates, and -- critically -- returns a
:class:`CurationReport` accounting for every record it removed and why. A
filtering pipeline you cannot audit is not a filtering pipeline; it is a
source of unexplained numbers.

Data licensing: ChEMBL is CC BY-SA 3.0. Attribution must name the release
version, which :meth:`ChemblClient.release` reports and every curated frame
carries in ``frame.attrs``. Share-alike applies to redistributed derivatives,
which is why this project queries ChEMBL at runtime and ships no derived
dataset. See docs/LICENSING.md.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .chem import standardize

log = logging.getLogger(__name__)

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
PAGE_SIZE = 1000

# Assay types. 'B' is a direct binding measurement against the target protein;
# 'F' is functional (cell or tissue readout). Pooling them into one SAR table
# is the most common way to produce a scatter plot with no trend in it.
ASSAY_TYPE_BINDING = "B"
ASSAY_TYPE_FUNCTIONAL = "F"

# Potency endpoints that share a -log10 scale and can be compared with care.
# Ki and Kd are thermodynamic; IC50 and EC50 depend on assay conditions
# (substrate concentration, incubation time), so mixing them across papers
# widens the spread even for a correctly measured compound.
POTENCY_TYPES = ("IC50", "Ki", "Kd", "EC50")

# Curator flags that mark a record as unreliable. Keeping these is a choice you
# should have to make explicitly, not one you make by forgetting the column exists.
INVALID_DATA_COMMENTS = {
    "potential missing data",
    "potential transcription error",
    "potential author error",
    "outside typical range",
    "non standard unit for type",
    "author confirmed error",
}


@dataclass
class CurationReport:
    """Full accounting of what curation removed, step by step."""

    target_chembl_id: str
    fetched: int = 0
    steps: list[tuple[str, int, str]] = field(default_factory=list)
    final_records: int = 0
    final_compounds: int = 0
    chembl_release: str = "unknown"

    def record(self, step: str, removed: int, reason: str) -> None:
        if removed:
            self.steps.append((step, removed, reason))

    @property
    def total_removed(self) -> int:
        return sum(n for _, n, _ in self.steps)

    def to_dict(self) -> dict:
        return {
            "target_chembl_id": self.target_chembl_id,
            "chembl_release": self.chembl_release,
            "fetched": self.fetched,
            "removed_total": self.total_removed,
            "final_records": self.final_records,
            "final_compounds": self.final_compounds,
            "steps": [{"step": s, "removed": n, "reason": r} for s, n, r in self.steps],
            "attribution": (
                f"Data from ChEMBL ({self.chembl_release}), EMBL-EBI, "
                "used under CC BY-SA 3.0. https://www.ebi.ac.uk/chembl/"
            ),
        }

    def summary(self) -> str:
        lines = [
            f"ChEMBL curation for {self.target_chembl_id} ({self.chembl_release})",
            f"  fetched {self.fetched} activity records",
        ]
        for step, n, reason in self.steps:
            lines.append(f"  -{n:>6}  {step}: {reason}")
        lines.append(
            f"  = {self.final_records} records over {self.final_compounds} unique compounds"
        )
        return "\n".join(lines)


class ChemblClient:
    """Thin, polite client over the ChEMBL REST API.

    Responses are cached on disk by request URL. ChEMBL is a public service
    funded by EMBL-EBI; re-fetching the same 17,000 activity records on every
    run because the pipeline has no cache is bad manners and also slow.
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
        session: requests.Session | None = None,
    ) -> None:
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else Path.home() / ".cache" / "chilecule" / "chembl"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "chilecule/0.1 (open-source drug discovery)"})

    # ---------------------------------------------------------------- HTTP

    def _cache_path(self, url: str) -> Path:
        import hashlib

        digest = hashlib.sha256(url.encode()).hexdigest()[:24]
        return self.cache_dir / f"{digest}.json"

    def _get(self, path: str, params: dict[str, Any] | None = None, use_cache: bool = True) -> dict:
        url = f"{BASE_URL}/{path.lstrip('/')}"
        prepared = requests.Request("GET", url, params=params or {}).prepare()
        cache_file = self._cache_path(prepared.url)

        if use_cache and cache_file.exists():
            return json.loads(cache_file.read_text())

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
                if use_cache:
                    cache_file.write_text(json.dumps(payload))
                return payload
            except requests.RequestException as exc:
                last_error = exc
                # Exponential backoff; EBI rate-limits aggressive clients.
                time.sleep(2**attempt)
        raise RuntimeError(f"ChEMBL request failed after {self.max_retries} attempts: {last_error}")

    def _paginate(self, path: str, params: dict[str, Any], key: str) -> Iterator[dict]:
        params = dict(params, limit=PAGE_SIZE, offset=0)
        while True:
            payload = self._get(path, params)
            records = payload.get(key, [])
            yield from records
            meta = payload.get("page_meta", {})
            if not meta.get("next"):
                return
            params["offset"] = meta["offset"] + meta["limit"]

    def release(self) -> str:
        """Current ChEMBL release, required for CC BY-SA attribution."""
        try:
            status = self._get("status.json", use_cache=False)
            # chembl_db_version already carries the "ChEMBL_" prefix.
            return str(status.get("chembl_db_version") or "ChEMBL (version unavailable)")
        except Exception:
            return "ChEMBL (version unavailable)"

    # ------------------------------------------------------------- Targets

    def find_targets(self, query: str, organism: str | None = "Homo sapiens") -> pd.DataFrame:
        """Resolve a gene symbol, protein name, or UniProt accession to targets.

        Returns targets ordered so that single-protein human targets come
        first. ChEMBL also stores protein complexes, cell lines, and whole
        organisms as "targets"; picking one of those by accident yields
        activity data that cannot be interpreted structurally.
        """
        params: dict[str, Any] = {"q": query}
        try:
            payload = self._get("target/search.json", params)
            targets = payload.get("targets", [])
        except Exception as exc:
            log.warning("target search failed for %r: %s", query, exc)
            return pd.DataFrame()

        rows = []
        for t in targets:
            if organism and t.get("organism") != organism:
                continue
            accessions = [
                comp.get("accession")
                for comp in (t.get("target_components") or [])
                if comp.get("accession")
            ]
            rows.append(
                {
                    "target_chembl_id": t.get("target_chembl_id"),
                    "pref_name": t.get("pref_name"),
                    "target_type": t.get("target_type"),
                    "organism": t.get("organism"),
                    "uniprot": accessions[0] if accessions else None,
                    "n_components": len(accessions),
                    "species_group_flag": t.get("species_group_flag"),
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        # SINGLE PROTEIN targets are the only ones with a well-defined structure
        # to dock into, so surface them first.
        df["_rank"] = (df["target_type"] != "SINGLE PROTEIN").astype(int)
        return df.sort_values("_rank").drop(columns="_rank").reset_index(drop=True)

    # ---------------------------------------------------------- Activities

    def fetch_activities(
        self,
        target_chembl_id: str,
        standard_types: tuple[str, ...] = POTENCY_TYPES,
        assay_type: str | None = ASSAY_TYPE_BINDING,
        max_records: int | None = None,
    ) -> pd.DataFrame:
        """Fetch raw activity records for a target. No curation applied."""
        params: dict[str, Any] = {
            "target_chembl_id": target_chembl_id,
            "standard_type__in": ",".join(standard_types),
        }
        if assay_type:
            params["assay_type"] = assay_type

        rows: list[dict] = []
        for record in self._paginate("activity.json", params, "activities"):
            rows.append(record)
            if max_records and len(rows) >= max_records:
                break

        if not rows:
            return pd.DataFrame()

        keep = [
            "molecule_chembl_id", "parent_molecule_chembl_id", "canonical_smiles",
            "standard_type", "standard_relation", "standard_value", "standard_units",
            "pchembl_value", "assay_chembl_id", "assay_type", "assay_description",
            "target_chembl_id", "target_organism", "data_validity_comment",
            "potential_duplicate", "document_chembl_id", "document_year", "activity_comment",
        ]
        df = pd.DataFrame(rows)
        return df[[c for c in keep if c in df.columns]]


# --------------------------------------------------------------- Curation


def curate_activities(
    df: pd.DataFrame,
    *,
    target_chembl_id: str = "",
    organism: str | None = "Homo sapiens",
    standard_types: tuple[str, ...] = ("IC50", "Ki"),
    drop_censored: bool = True,
    max_replicate_spread_log: float = 1.0,
    max_ligand_efficiency: float | None = 0.83,
    chembl_release: str = "unknown",
    standardize_structures: bool = True,
) -> tuple[pd.DataFrame, CurationReport]:
    """Turn raw ChEMBL activities into a defensible SAR table.

    Returns one row per unique compound, with ``pchembl`` as the aggregated
    -log10(molar potency), alongside a :class:`CurationReport` that accounts
    for every dropped record.

    ``drop_censored`` removes ``>`` and ``<`` relations. They are real
    information -- a ``>10 uM`` result genuinely means inactive -- but they are
    *right-censored*, and treating the bound as a point estimate is wrong.
    Recover them separately via :func:`censored_inactives` when you need a
    negative set for classification or decoy validation.

    ``max_ligand_efficiency`` rejects records whose potency is thermodynamically
    impossible for a molecule of that size -- see step 8b for why this catches
    errors that range checks cannot.

    ``max_replicate_spread_log`` drops compounds whose independent measurements
    disagree by more than this many log units. A compound with reported IC50s
    of 30 nM and 8 uM has not been measured; it has been measured twice in
    incompatible assays, and including it adds noise no model can fit.
    """
    report = CurationReport(target_chembl_id=target_chembl_id, chembl_release=chembl_release)
    if df.empty:
        return pd.DataFrame(), report

    df = df.copy()
    report.fetched = len(df)

    def drop(mask: pd.Series, step: str, reason: str) -> pd.DataFrame:
        nonlocal df
        removed = int(mask.sum())
        report.record(step, removed, reason)
        df = df[~mask]
        return df

    # 1. Structure must exist and parse.
    drop(df["canonical_smiles"].isna(), "missing structure", "no canonical_smiles in record")

    # 2. Curator-flagged records.
    if "data_validity_comment" in df:
        flagged = (
            df["data_validity_comment"]
            .fillna("")
            .str.strip()
            .str.lower()
            .isin(INVALID_DATA_COMMENTS)
        )
        drop(flagged, "curator flag", "data_validity_comment marks record as suspect")

    # 3. Species. Human and rodent orthologs differ enough at the binding site
    #    that pooling them is a modelling error, not a data-volume win.
    if organism and "target_organism" in df:
        drop(df["target_organism"] != organism, "wrong organism", f"target_organism != {organism}")

    # 4. Endpoint type.
    drop(~df["standard_type"].isin(standard_types), "endpoint type",
         f"standard_type not in {list(standard_types)}")

    # 5. Censored relations.
    if drop_censored and "standard_relation" in df:
        censored = ~df["standard_relation"].fillna("").isin(["=", "~"])
        drop(censored, "censored", "standard_relation is an inequality, not a point estimate")

    # 6. Units must be convertible to a molar concentration.
    df["value_nm"] = _to_nanomolar(df)
    drop(df["value_nm"].isna(), "unconvertible units",
         "standard_units not a molar concentration (e.g. ug.mL-1 without MW)")

    # 7. Physically implausible values. Below 1 pM is beyond the detection
    #    limit of essentially every biochemical assay; above 1 M is not a
    #    measurement of anything.
    implausible = (df["value_nm"] <= 1e-3) | (df["value_nm"] > 1e9)
    drop(implausible, "implausible value", "potency outside 1 pM - 1 M")

    if df.empty:
        return pd.DataFrame(), report

    # 8. Convert to pChEMBL scale (-log10 molar).
    import numpy as np

    df["pchembl"] = -np.log10(df["value_nm"] * 1e-9)

    # 8b. Thermodynamic plausibility via ligand efficiency.
    #
    #     Binding energy per heavy atom has a ceiling. Kuntz et al. (PNAS 1999)
    #     found maximal affinity per atom approaching ~1.5 kcal/mol only for
    #     the very smallest ligands (metal chelators and the like), with a
    #     practical ceiling near 0.7 for organic ligands of drug-like size.
    #
    #     This catches an error class that unit checks and range checks both
    #     miss: a 10-heavy-atom fragment reported at 0.1 nM implies an LE of
    #     1.36, which no organic fragment achieves. The number is almost always
    #     a transcription error or a mis-assigned units column that happens to
    #     land inside the plausible absolute range.
    if max_ligand_efficiency is not None:
        heavy = df["canonical_smiles"].map(_heavy_atom_count)
        le = (0.5961 * np.log(10) * df["pchembl"]) / heavy.replace(0, np.nan)
        df["_le"] = le
        drop(
            df["_le"] > max_ligand_efficiency,
            "implausible ligand efficiency",
            f"LE > {max_ligand_efficiency} kcal/mol/heavy atom exceeds the thermodynamic ceiling",
        )
        df = df.drop(columns=["_le"], errors="ignore")

    if df.empty:
        return pd.DataFrame(), report

    # 9. Standardize structures so that salts and tautomers of one compound
    #    aggregate together rather than appearing as distinct chemical matter.
    if standardize_structures:
        std = df["canonical_smiles"].map(lambda s: standardize(s, canonical_tautomer=False))
        df["smiles"] = [s.smiles for s in std]
        df["inchikey"] = [s.inchikey for s in std]
        drop(df["smiles"].isna(), "unparseable structure", "RDKit could not standardize the SMILES")
    else:
        df["smiles"] = df["canonical_smiles"]
        df["inchikey"] = None

    if df.empty:
        return pd.DataFrame(), report

    group_key = "inchikey" if standardize_structures and df["inchikey"].notna().all() else "smiles"

    # 10. Aggregate replicates and drop compounds whose measurements disagree.
    grouped = df.groupby(group_key)
    spread = grouped["pchembl"].agg(lambda s: s.max() - s.min())
    inconsistent = spread[spread > max_replicate_spread_log].index
    if len(inconsistent):
        mask = df[group_key].isin(inconsistent)
        drop(mask, "inconsistent replicates",
             f"independent measurements disagree by >{max_replicate_spread_log} log units")

    if df.empty:
        return pd.DataFrame(), report

    agg = (
        df.groupby(group_key)
        .agg(
            smiles=("smiles", "first"),
            molecule_chembl_id=("molecule_chembl_id", "first"),
            pchembl=("pchembl", "median"),
            value_nm=("value_nm", "median"),
            n_measurements=("pchembl", "size"),
            spread_log=("pchembl", lambda s: round(float(s.max() - s.min()), 2)),
            standard_types=("standard_type", lambda s: ",".join(sorted(set(s)))),
            n_documents=("document_chembl_id", "nunique"),
        )
        .reset_index()
        .sort_values("pchembl", ascending=False)
        .reset_index(drop=True)
    )
    agg["pchembl"] = agg["pchembl"].round(2)

    report.final_records = int(len(df))
    report.final_compounds = int(len(agg))
    agg.attrs["chembl_release"] = chembl_release
    agg.attrs["license"] = "CC BY-SA 3.0"
    agg.attrs["attribution"] = report.to_dict()["attribution"]
    return agg, report


def censored_inactives(df: pd.DataFrame, threshold_nm: float = 10_000) -> pd.DataFrame:
    """Recover right-censored records as a labelled inactive set.

    ``>10 uM`` records are the only large-scale source of experimentally
    confirmed negatives in ChEMBL. They are useless for regression and
    valuable for classification and for validating a screening cascade -- a
    pipeline that cannot separate these from the actives is not working.
    """
    if df.empty or "standard_relation" not in df:
        return pd.DataFrame()

    censored = df[df["standard_relation"].fillna("").isin([">", ">="])].copy()
    censored["value_nm"] = _to_nanomolar(censored)
    censored = censored[censored["value_nm"] >= threshold_nm]
    censored = censored[censored["canonical_smiles"].notna()]
    if censored.empty:
        return pd.DataFrame()
    return (
        censored[["molecule_chembl_id", "canonical_smiles", "value_nm", "standard_type"]]
        .rename(columns={"canonical_smiles": "smiles"})
        .drop_duplicates("molecule_chembl_id")
        .reset_index(drop=True)
    )


# Multipliers to nanomolar. ug.mL-1 is deliberately absent: converting it needs
# the compound's molecular weight, and silently guessing one is how unit bugs
# get into published models.
_UNIT_TO_NM = {
    "nM": 1.0,
    "uM": 1e3,
    "µM": 1e3,
    "mM": 1e6,
    "M": 1e9,
    "pM": 1e-3,
    "fM": 1e-6,
    "nmol/L": 1.0,
    "umol/L": 1e3,
}


def _heavy_atom_count(smiles: str) -> float:
    """Heavy-atom count, or NaN when the structure will not parse."""
    import numpy as np

    from .chem import parse_smiles

    mol = parse_smiles(smiles) if isinstance(smiles, str) else None
    return float(mol.GetNumHeavyAtoms()) if mol is not None else np.nan


def _to_nanomolar(df: pd.DataFrame) -> pd.Series:
    """Convert standard_value/standard_units to nM, or NaN if not possible."""
    import numpy as np

    if "standard_value" not in df or "standard_units" not in df:
        return pd.Series(np.nan, index=df.index)
    values = pd.to_numeric(df["standard_value"], errors="coerce")
    factors = df["standard_units"].map(_UNIT_TO_NM)
    return values * factors
