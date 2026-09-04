"""Target dossier: what is known about a target, assembled from public data.

The first question on any new program is "what has already been done here".
Answering it means reconciling three databases that do not agree with each
other about what a target is -- UniProt (sequences), the PDB (structures), and
ChEMBL (bioactivity) -- and then judging whether the resulting picture supports
a structure-based campaign.

Runs in under a minute on any machine. No docking, no ML, no GPU.
"""

from __future__ import annotations

import logging
import re

import pandas as pd
import requests

from ..tools.chem import properties
from ..tools.chembl import ChemblClient, censored_inactives, curate_activities
from ..tools.sar import scaffold_summary
from ..tools.structure import rank_structures, structures_for_uniprot
from .report import Report, Section, base_provenance

log = logging.getLogger(__name__)

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"


def _gene_names(entry: dict) -> set[str]:
    names = set()
    for gene in entry.get("genes", []):
        primary = gene.get("geneName", {}).get("value")
        if primary:
            names.add(primary.upper())
        for synonym in gene.get("synonyms", []):
            value = synonym.get("value")
            if value:
                names.add(value.upper())
    return names


def _protein_name(entry: dict) -> str:
    return (
        entry.get("proteinDescription", {})
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value", "")
    )


def _verify_match(results: list[dict], query: str) -> dict | None:
    """Accept a result only if it genuinely corresponds to the query.

    A match means the query equals one of the entry's gene names or symbols, or
    it matches the entry's recommended protein name. Appearing anywhere in the
    free-text function annotation is explicitly not a match -- that is the
    failure mode this function exists to prevent.
    """
    wanted = query.upper().strip()

    # Three passes, each exhausting every result before the next begins. A
    # single pass with an "exact or substring" test returns whichever entry
    # comes first, which is the wrong one whenever a less specific protein
    # ranks above the exact match: searching "Epidermal growth factor
    # receptor" puts "Epidermal growth factor receptor kinase substrate 8"
    # ahead of EGFR itself in UniProt's ranking.
    for entry in results:
        if wanted in _gene_names(entry):
            return entry
    for entry in results:
        if _protein_name(entry).upper() == wanted:
            return entry
    for entry in results:
        if len(wanted) > 8 and wanted in _protein_name(entry).upper():
            return entry
    return None


# A UniProt accession: one letter, then a defined alternation of digits and
# letters. Matching this first avoids treating "P00533" as a gene symbol.
ACCESSION_PATTERN = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$"
    r"|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$"
)

UNIPROT_FIELDS = (
    "accession,id,protein_name,gene_names,length,cc_function,ft_domain,cc_subcellular_location"
)


def uniprot_summary(query: str, timeout: int = 30) -> dict:
    """Resolve a gene symbol or accession to a reviewed human UniProt entry.

    Queries are field-scoped and tried in decreasing order of specificity:
    accession, then exact gene name, then protein name. Free-text search is
    the last resort and its result is verified against the query before being
    accepted.

    The reason for all of this: UniProt's default search matches the query
    anywhere in the record, including inside free-text function annotations.
    A bare search for "EGFR" returns E3 ubiquitin-protein ligase CBL-C ahead of
    EGFR itself, because CBL-C's function paragraph mentions EGFR several
    times. The dossier then describes the wrong protein while ChEMBL, which
    resolves targets properly, describes the right one -- and the two halves of
    the report silently disagree.

    Restricted to reviewed (Swiss-Prot) entries: TrEMBL holds many unreviewed
    predicted proteins per gene, and picking one produces a dossier about a
    sequence nobody has characterized.
    """
    query = query.strip()
    filters = "(organism_id:9606) AND (reviewed:true)"
    is_accession = bool(ACCESSION_PATTERN.match(query.upper()))
    attempts = (
        [f"(accession:{query.upper()})"]
        if is_accession
        else [
            f"(gene_exact:{query}) AND {filters}",
            f"(gene:{query}) AND {filters}",
            f"(protein_name:{query}) AND {filters}",
        ]
    )

    entry: dict | None = None
    for search in attempts:
        params = {"query": search, "fields": UNIPROT_FIELDS, "format": "json", "size": "10"}
        try:
            response = requests.get(UNIPROT_SEARCH, params=params, timeout=timeout)
            response.raise_for_status()
            results = response.json().get("results", [])
        except requests.RequestException as exc:
            log.warning("UniProt lookup failed for %r: %s", query, exc)
            return {}
        if not results:
            continue
        # An accession is unambiguous by construction; anything else must be
        # verified against gene and protein names before it is accepted.
        entry = results[0] if is_accession else _verify_match(results, query)
        if entry is not None:
            break

    if entry is None:
        return {}
    function = ""
    for comment in entry.get("comments", []):
        if comment.get("commentType") == "FUNCTION":
            texts = comment.get("texts", [])
            if texts:
                function = texts[0].get("value", "")
                break

    return {
        "matched_by": "uniprot",
        "accession": entry.get("primaryAccession"),
        "entry_name": entry.get("uniProtkbId"),
        "protein_name": (
            entry.get("proteinDescription", {})
            .get("recommendedName", {})
            .get("fullName", {})
            .get("value", "")
        ),
        "genes": [g.get("geneName", {}).get("value") for g in entry.get("genes", [])],
        "length": entry.get("sequence", {}).get("length"),
        "function": function,
    }


def build(
    target: str,
    *,
    max_activity_records: int = 5000,
    client: ChemblClient | None = None,
) -> Report:
    """Assemble a dossier for a target named by gene symbol or UniProt accession."""
    client = client or ChemblClient()
    report = Report(workflow="Target dossier", subject=target)

    # ---------------------------------------------------------- Identity
    protein = uniprot_summary(target)
    if protein:
        genes = ", ".join(g for g in protein.get("genes", []) if g)
        report.add(
            Section(
                title="Target identity",
                body=(
                    f"**{protein['protein_name']}** ({protein['accession']}, "
                    f"{protein['entry_name']})\n\n"
                    f"- Genes: {genes or 'not listed'}\n"
                    f"- Length: {protein.get('length')} residues\n\n"
                    f"{protein.get('function', '')}"
                ),
                data={"uniprot": protein},
            )
        )
    else:
        report.warn(f"No reviewed human UniProt entry matched {target!r}.")

    # -------------------------------------------------------- Structures
    accession = protein.get("accession")
    structures = rank_structures(structures_for_uniprot(accession)) if accession else []
    if structures:
        table = pd.DataFrame([s.to_dict() for s in structures[:10]])
        best = structures[0]
        report.add(
            Section(
                title="Structural coverage",
                body=(
                    f"{len(structures)} distinct PDB entries map to {accession}. "
                    f"Ranked for structure-based design, the best starting point is "
                    f"**{best.pdb_id}** ({best.resolution} A, {best.method}).\n\n"
                    "Ranking weights resolution above sequence coverage: a high-resolution "
                    "structure of the relevant domain supports docking, while a "
                    "low-resolution model of the full-length protein does not, however "
                    "much of the sequence it spans."
                ),
                table=table,
                data={"n_structures": len(structures), "recommended_pdb": best.pdb_id},
            )
        )
    else:
        report.warn(
            "No experimental structures found. Structure-based workflows are unavailable "
            "for this target; ligand-based SAR analysis still applies."
        )

    # ------------------------------------------------------- Bioactivity
    targets = client.find_targets(target)
    if targets.empty:
        report.warn(f"No ChEMBL target matched {target!r}; no bioactivity section.")
        report.provenance = base_provenance(target_query=target)
        return report

    chembl_target = targets.iloc[0]
    release = client.release()

    # Cross-check the two independent identifier resolutions. If UniProt and
    # ChEMBL landed on different proteins, every downstream section is
    # describing a different molecule than the section above it, and the report
    # must say so rather than presenting a silently incoherent picture.
    if accession and chembl_target.get("uniprot") and chembl_target["uniprot"] != accession:
        report.warn(
            f"Identifier mismatch: UniProt resolved {target!r} to {accession}, but the "
            f"best ChEMBL target {chembl_target['target_chembl_id']} maps to "
            f"{chembl_target['uniprot']}. The structural and bioactivity sections below "
            "describe different proteins. Re-run with an explicit UniProt accession."
        )
    raw = client.fetch_activities(
        chembl_target["target_chembl_id"], max_records=max_activity_records
    )
    actives, curation = curate_activities(
        raw,
        target_chembl_id=chembl_target["target_chembl_id"],
        chembl_release=release,
    )
    inactives = censored_inactives(raw)

    report.add(
        Section(
            title="Bioactivity landscape",
            body=(
                f"ChEMBL target **{chembl_target['target_chembl_id']}** "
                f"({chembl_target['pref_name']}, {chembl_target['target_type']}).\n\n"
                "```\n" + curation.summary() + "\n```\n\n"
                f"{len(inactives)} additional compounds are recorded as inactive "
                "(right-censored measurements), usable as experimental negatives."
            ),
            data={"curation": curation.to_dict(), "n_confirmed_inactives": len(inactives)},
        )
    )

    if actives.empty:
        report.warn("No activity records survived curation; the target is under-characterized.")
        report.provenance = base_provenance(target_query=target, chembl_release=release)
        return report

    # ------------------------------------------------------- Chemotypes
    scaffolds = scaffold_summary(actives)
    if not scaffolds.empty:
        top_share = scaffolds.iloc[0]["fraction_of_set"]
        concentration = (
            f"The single most common scaffold accounts for {top_share:.0%} of the set."
        )
        if top_share > 0.25:
            report.warn(
                f"Chemical matter is concentrated: {top_share:.0%} of actives share one "
                "Bemis-Murcko scaffold. Any model trained on a random split of this set "
                "will overstate its ability to generalize to new chemistry -- use a "
                "scaffold split."
            )
        report.add(
            Section(
                title="Known chemotypes",
                body=(
                    f"{len(scaffolds)} distinct Bemis-Murcko scaffolds across "
                    f"{len(actives)} compounds. {concentration}"
                ),
                table=scaffolds.head(10),
                data={"n_scaffolds": len(scaffolds), "top_scaffold_share": float(top_share)},
            )
        )

    # ------------------------------------------------ Property envelope
    profile = pd.DataFrame(
        [p.to_dict() for p in (properties(s) for s in actives["smiles"].head(500)) if p]
    )
    if not profile.empty:
        potent = actives[actives["pchembl"] >= 8]
        stats = profile[["mw", "clogp", "tpsa", "hbd", "hba", "rotatable_bonds", "qed"]].describe()
        report.add(
            Section(
                title="Property envelope of known actives",
                body=(
                    f"{len(potent)} compounds reach pChEMBL >= 8 (10 nM or better). "
                    "The distribution below describes the chemical space that has already "
                    "produced activity, and is the sensible envelope for a generated or "
                    "purchased library."
                ),
                table=stats.round(2).reset_index().rename(columns={"index": "statistic"}),
                data={
                    "n_potent": int(len(potent)),
                    "median_mw": float(profile["mw"].median()),
                    "median_clogp": float(profile["clogp"].median()),
                },
            )
        )

    report.add(
        Section(
            title="Most potent known compounds",
            body="Top compounds by curated median potency.",
            table=actives.head(15)[
                ["molecule_chembl_id", "smiles", "pchembl", "n_measurements", "standard_types"]
            ],
        )
    )

    report.provenance = base_provenance(
        target_query=target,
        uniprot=accession,
        chembl_target=chembl_target["target_chembl_id"],
        chembl_release=release,
        chembl_license="CC BY-SA 3.0",
        n_actives=len(actives),
        n_confirmed_inactives=len(inactives),
    )
    return report
