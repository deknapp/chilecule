"""Everything worth knowing about a compound, before you spend money on it.

The daily workhorse. Takes one structure or a list and answers the questions a
medicinal chemist asks about anything that lands on their desk: what is it,
is it drug-like, does it carry known liabilities, can it be made, has anyone
made it, and what else does it hit.

Runs on a laptop. The only slow part is the network, and everything is cached.
"""

from __future__ import annotations

import pandas as pd

from ..parallel import pmap
from ..tools import alerts as alerts_module
from ..tools import lookup, score, synth
from ..tools.chem import properties, standardize
from .report import Report, Section, base_provenance


def profile_one(smiles: str, *, check_databases: bool = True, profile: str = "oral") -> dict:
    """Full profile of a single structure, as a flat record."""
    standardized = standardize(smiles, canonical_tautomer=False)
    if not standardized.ok:
        return {"input": smiles, "error": standardized.error}

    canonical = standardized.smiles
    props = properties(canonical)
    alert_report = alerts_module.screen(canonical)
    synthesis = synth.assess(canonical)
    scorecard = score.score(canonical, profile=profile)

    record: dict = {
        "input": smiles,
        "smiles": canonical,
        "inchikey": standardized.inchikey,
        "salt_stripped": standardized.changed,
        **(props.to_dict() if props else {}),
        "alerts": len(alert_report.alerts) if alert_report else None,
        "alert_severity": alert_report.max_severity if alert_report else None,
        "alert_detail": (
            "; ".join(f"{a.catalog}:{a.description}" for a in alert_report.alerts)
            if alert_report else ""
        ),
        "sa_score": round(synthesis.sa_score, 2) if synthesis else None,
        "synthesis_tier": synthesis.tier if synthesis else None,
        "synthesis_flags": "; ".join(synthesis.flags) if synthesis else "",
        f"{profile}_score": round(scorecard.score, 3) if scorecard else None,
        "limiting_property": (
            scorecard.limiting["property"] if scorecard and scorecard.limiting else None
        ),
    }

    if check_databases:
        novelty = lookup.check_novelty(canonical)
        record.update(
            {
                "known": novelty.is_known,
                "novelty": novelty.verdict,
                "chembl_id": novelty.chembl_id,
                "chembl_name": novelty.chembl_name,
                "nearest_known": (
                    novelty.nearest_neighbours[0]["similarity"]
                    if novelty.nearest_neighbours else None
                ),
            }
        )
        if novelty.chembl_id:
            promiscuity = lookup.check_promiscuity(novelty.chembl_id)
            record.update(
                {
                    "n_protein_targets": promiscuity.n_targets,
                    "selectivity_window": promiscuity.selectivity_window,
                    "promiscuity": promiscuity.assessment,
                }
            )
    return record


def build(
    smiles_list: list[str],
    *,
    check_databases: bool = True,
    profile: str = "oral",
) -> Report:
    """Profile one compound or a list of them."""
    subject = smiles_list[0] if len(smiles_list) == 1 else f"{len(smiles_list)} compounds"
    report = Report(workflow="Compound profile", subject=subject)

    records = pmap(
        lambda s: profile_one(s, check_databases=check_databases, profile=profile),
        smiles_list,
    )
    frame = pd.DataFrame(records)

    failed = frame[frame.get("error").notna()] if "error" in frame else pd.DataFrame()
    frame = frame[frame["smiles"].notna()] if "smiles" in frame else frame
    if not failed.empty:
        report.warn(f"{len(failed)} structure(s) could not be parsed and were dropped.")
    if frame.empty:
        report.warn("No structures could be profiled.")
        return report

    # ---- single-compound view reads as a narrative; a list reads as a table
    if len(frame) == 1:
        record = frame.iloc[0]
        report.add(Section(title="Identity", body=_identity_text(record), data=record.to_dict()))
        report.add(Section(title="Physicochemical profile", body=_property_text(record)))
        report.add(Section(title="Liabilities and accessibility", body=_liability_text(record)))
        if check_databases:
            report.add(Section(title="Prior art and selectivity", body=_prior_art_text(record)))
    else:
        columns = [
            c for c in (
                "smiles", "mw", "clogp", "tpsa", f"{profile}_score", "limiting_property",
                "alerts", "alert_severity", "sa_score", "synthesis_tier", "known", "novelty",
            ) if c in frame
        ]
        report.add(
            Section(
                title="Profile",
                body=(
                    f"{len(frame)} compounds profiled, ranked by {profile} scorecard.\n\n"
                    "`limiting_property` names the property costing each compound the most, "
                    "which is more actionable than the score. Alerts are annotations, not "
                    "verdicts -- roughly 5% of approved drugs match a PAINS pattern."
                ),
                table=frame.sort_values(f"{profile}_score", ascending=False)[columns],
                max_rows=40,
            )
        )
        report.add(Section(title="Summary", body=_batch_summary(frame, profile)))

    report.provenance = base_provenance(
        n_compounds=len(frame), scorecard_profile=profile, database_lookup=check_databases
    )
    return report


def _identity_text(record) -> str:
    lines = [f"**SMILES** `{record['smiles']}`", f"**InChIKey** `{record.get('inchikey')}`"]
    if record.get("salt_stripped"):
        lines.append(
            "\nThe input was standardized -- a counterion, solvate or charge state was "
            "removed. Comparisons downstream use the parent structure."
        )
    return "\n\n".join(lines)


def _property_text(record) -> str:
    return (
        f"MW {record.get('mw')} · cLogP {record.get('clogp')} · TPSA {record.get('tpsa')} · "
        f"HBD {record.get('hbd')} · HBA {record.get('hba')} · "
        f"rotatable bonds {record.get('rotatable_bonds')} · "
        f"aromatic rings {record.get('aromatic_rings')} · "
        f"Fsp3 {record.get('fraction_csp3')} · QED {record.get('qed')}\n\n"
        f"**Scorecard:** {record.get('oral_score', record.get('lead_like_score'))} "
        f"— limited by **{record.get('limiting_property') or 'nothing; all properties in range'}**"
    )


def _liability_text(record) -> str:
    alert_count = record.get("alerts") or 0
    if alert_count:
        alert_text = (
            f"**{alert_count} structural alert(s)**, highest severity "
            f"{record.get('alert_severity')}: {record.get('alert_detail')}\n\n"
            "Alerts are annotations, not verdicts. Aspirin matches a Brenk alert and "
            "roughly 5% of approved drugs match a PAINS pattern; if you exclude on one, "
            "name the catalog and say why it applies here."
        )
    else:
        alert_text = "**No structural alerts** in the PAINS, Brenk, NIH or ZINC catalogs."

    synthesis = (
        f"**Synthetic accessibility:** SA score {record.get('sa_score')} "
        f"({record.get('synthesis_tier')})."
    )
    if record.get("synthesis_flags"):
        synthesis += f" {record['synthesis_flags']}"
    return f"{alert_text}\n\n{synthesis}"


def _prior_art_text(record) -> str:
    lines = [f"**Novelty:** {record.get('novelty')}"]
    if record.get("chembl_id"):
        name = f" ({record['chembl_name']})" if record.get("chembl_name") else ""
        lines.append(f"ChEMBL: `{record['chembl_id']}`{name}")
    if record.get("promiscuity"):
        lines.append(f"**Selectivity:** {record['promiscuity']}")
    lines.append(
        "*Structural novelty only. A compound absent from ChEMBL and PubChem may still "
        "fall inside a Markush claim; this is not a freedom-to-operate opinion.*"
    )
    return "\n\n".join(lines)


def _batch_summary(frame: pd.DataFrame, profile: str) -> str:
    score_col = f"{profile}_score"
    clean = int((frame["alerts"] == 0).sum()) if "alerts" in frame else 0
    good = int((frame[score_col] >= 0.7).sum()) if score_col in frame else 0
    easy = int((frame["sa_score"] <= 4.0).sum()) if "sa_score" in frame else 0
    known = int(frame["known"].sum()) if "known" in frame else None

    lines = [
        f"- {clean}/{len(frame)} carry no structural alert",
        f"- {good}/{len(frame)} score 0.7 or better on the {profile} scorecard",
        f"- {easy}/{len(frame)} are readily accessible synthetically (SA <= 4)",
    ]
    if known is not None:
        lines.append(f"- {known}/{len(frame)} are already in ChEMBL or PubChem")
    if "limiting_property" in frame:
        limiting = frame["limiting_property"].value_counts()
        if not limiting.empty:
            top = limiting.index[0]
            lines.append(
                f"- The most common limiting property across the set is **{top}** "
                f"({limiting.iloc[0]} compounds), which is where series-wide effort would pay off"
            )
    return "\n".join(lines)
