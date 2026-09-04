"""What should I make next?

Given a hit and a target, proposes analogs built only from transformations
medicinal chemists have already made against that target, each carrying the
evidence for what it did to potency.

This is deliberately not generative design. Every proposal traces to matched
molecular pairs in ChEMBL, so a chemist can check the claim -- and disagreeing
with it is the point. A design suggestion nobody can argue with is not a
design suggestion.
"""

from __future__ import annotations

import pandas as pd

from ..parallel import pmap
from ..tools import score as score_module
from ..tools import synth
from ..tools.alerts import screen
from ..tools.chem import properties
from ..tools.chembl import ChemblClient, curate_activities
from ..tools.design import enumerate_analogs, mine_transformations
from ..tools.lookup import check_novelty
from .report import Report, Section, base_provenance


def build(
    parent_smiles: str,
    target: str,
    *,
    max_activity_records: int = 5000,
    min_occurrences: int = 4,
    max_analogs: int = 60,
    check_novelty_of_analogs: bool = True,
    scorecard: str = "oral",
    client: ChemblClient | None = None,
) -> Report:
    """Propose analogs of ``parent_smiles`` using evidence from ``target``."""
    client = client or ChemblClient()
    report = Report(workflow="Analog design", subject=parent_smiles)

    parent_props = properties(parent_smiles)
    if parent_props is None:
        report.warn(f"Could not parse the parent structure {parent_smiles!r}.")
        return report

    targets = client.find_targets(target)
    if targets.empty:
        report.warn(f"No ChEMBL target matched {target!r}; no transformations could be mined.")
        return report

    chembl_target = targets.iloc[0]
    release = client.release()
    raw = client.fetch_activities(
        chembl_target["target_chembl_id"], max_records=max_activity_records
    )
    actives, curation = curate_activities(
        raw, target_chembl_id=chembl_target["target_chembl_id"], chembl_release=release
    )
    if len(actives) < 50:
        report.warn(
            f"Only {len(actives)} curated compounds for {target}. Matched-pair evidence from "
            "a set this small is anecdotal; treat every expected effect below as a guess."
        )

    transformations = mine_transformations(actives, min_occurrences=min_occurrences)
    if not transformations:
        report.warn(
            "No transformation was observed often enough to propose from. Either the target "
            "has too little data, or its compounds are too structurally diverse for "
            "matched-pair analysis."
        )
        report.provenance = base_provenance(target=target, chembl_release=release)
        return report

    # ---- parent baseline, scored in the same protocol as the analogs
    parent_scorecard = score_module.score(parent_smiles, profile=scorecard)
    parent_synth = synth.assess(parent_smiles)
    report.add(
        Section(
            title="Parent compound",
            body=(
                f"`{parent_props.smiles}`\n\n"
                f"MW {parent_props.mw} · cLogP {parent_props.clogp} · "
                f"TPSA {parent_props.tpsa} · QED {parent_props.qed}\n\n"
                f"**Scorecard:** {parent_scorecard.verdict if parent_scorecard else 'n/a'}\n\n"
                f"**Synthesis:** {parent_synth.summary if parent_synth else 'n/a'}\n\n"
                "Every analog below is scored in this same protocol, so the deltas mean "
                "something. Scoring proposals without scoring the thing they came from is "
                "how a series appears to improve while standing still."
            ),
            data={"parent": parent_props.to_dict()},
        )
    )

    report.add(
        Section(
            title="Transformation evidence",
            body=(
                f"{len(transformations)} potency-improving transformations mined from "
                f"{len(actives)} curated compounds against "
                f"**{chembl_target['target_chembl_id']}** ({chembl_target['pref_name']}).\n\n"
                "`median_delta` is the observed change in pChEMBL when this exact swap was "
                "made, in log units. Read `std_delta` next to it: a large median with a "
                "large spread means the transformation is context-dependent and applying it "
                "at a new position will not reproduce the gain. `reliability` folds both "
                "into one word."
            ),
            table=pd.DataFrame([t.to_dict() for t in transformations[:15]]),
            data={"n_transformations": len(transformations)},
        )
    )

    proposals = enumerate_analogs(
        parent_smiles,
        transformations,
        evidence_compounds=actives["smiles"].head(500).tolist(),
        max_total=max_analogs,
    )
    if not proposals:
        report.warn(
            "None of the mined transformations apply to this parent -- no fragment of the "
            "parent matches the left-hand side of any transformation with a track record. "
            "That is a real answer: the changes that worked on this target were made at "
            "positions this compound does not have."
        )
        report.provenance = base_provenance(target=target, chembl_release=release)
        return report

    out_of_domain = [p for p in proposals if not p.in_evidence_domain]
    if out_of_domain:
        report.warn(
            f"The parent is outside the chemical space these transformations were mined "
            f"from ({out_of_domain[0].domain_note}). The structures below are valid; their "
            "expected effects are reported as **no decision** rather than as numbers, "
            "because a matched-pair median does not transfer across chemotypes."
        )

    rows = []
    for proposal in proposals:
        analog_props = properties(proposal.smiles)
        analog_score = score_module.score(proposal.smiles, profile=scorecard)
        analog_synth = synth.assess(proposal.smiles)
        alerts = screen(proposal.smiles)
        rows.append(
            {
                "smiles": proposal.smiles,
                "transformation": proposal.transformation.label,
                "evidence": proposal.expected_effect,
                "n_pairs": proposal.transformation.n_pairs,
                "median_delta": proposal.transformation.median_delta,
                "reliability": proposal.transformation.reliability,
                "similarity_to_parent": proposal.similarity_to_parent,
                "mw": analog_props.mw if analog_props else None,
                "clogp": analog_props.clogp if analog_props else None,
                "d_mw": round(analog_props.mw - parent_props.mw, 1) if analog_props else None,
                "d_clogp": (
                    round(analog_props.clogp - parent_props.clogp, 2) if analog_props else None
                ),
                "score": round(analog_score.score, 3) if analog_score else None,
                "d_score": (
                    round(analog_score.score - parent_scorecard.score, 3)
                    if analog_score and parent_scorecard else None
                ),
                "sa_score": round(analog_synth.sa_score, 2) if analog_synth else None,
                "alerts": len(alerts.alerts) if alerts else 0,
            }
        )

    frame = pd.DataFrame(rows)
    if check_novelty_of_analogs:
        # Parallel: each check is three network round trips (ChEMBL exact,
        # PubChem exact, ChEMBL similarity), and doing 25 of them in sequence
        # takes minutes of pure latency.
        novelty = pmap(check_novelty, frame["smiles"].head(25).tolist(), threshold=2)
        known = {n.smiles: n for n in novelty}
        frame["already_made"] = frame["smiles"].map(
            lambda s: known[s].is_known if s in known else None
        )
        frame["chembl_id"] = frame["smiles"].map(
            lambda s: known[s].chembl_id if s in known else None
        )

    # Rank on evidence first, then on whether the analog improves the profile.
    # Docking scores are absent here on purpose: this is a ligand-based design
    # question and adding a weak structural signal would dilute a strong
    # empirical one.
    frame = frame.sort_values(
        ["median_delta", "d_score"], ascending=[False, False]
    ).reset_index(drop=True)

    report.add(
        Section(
            title="Proposed analogs",
            body=(
                f"{len(frame)} analogs, each one substitution from the parent, ranked by the "
                "strength of the evidence for the transformation that produced it.\n\n"
                "`d_score`, `d_mw` and `d_clogp` are relative to the parent. `already_made` "
                "flags analogs that already exist in ChEMBL or PubChem -- often the most "
                "useful column on the page, since a known compound may be purchasable and "
                "may already have data.\n\n"
                "These are hypotheses supported by precedent, not predictions. The "
                "transformation moved potency by this much in the contexts where it was "
                "observed; whether it does so here is what the experiment is for."
            ),
            table=frame,
            max_rows=30,
            data={"n_proposals": len(frame), "n_out_of_domain": len(out_of_domain)},
        )
    )

    report.add(
        Section(
            title="What I would make first",
            body=_recommendation_text(frame, parent_scorecard),
        )
    )

    report.provenance = base_provenance(
        parent=parent_props.smiles,
        target=target,
        chembl_target=chembl_target["target_chembl_id"],
        chembl_release=release,
        chembl_license="CC BY-SA 3.0",
        n_evidence_compounds=len(actives),
        n_transformations=len(transformations),
        curation_removed=curation.total_removed,
    )
    return report


def _recommendation_text(frame: pd.DataFrame, parent_scorecard) -> str:
    """A short, opinionated shortlist -- the part a chemist reads first."""
    if frame.empty:
        return "No analogs to recommend."

    well_supported = frame[frame["reliability"].isin(["well-supported", "moderate"])]
    clean = well_supported[well_supported["alerts"] == 0] if not well_supported.empty else frame
    improving = clean[clean["d_score"].fillna(0) >= 0] if "d_score" in clean else clean
    shortlist = (improving if not improving.empty else clean).head(3)

    if shortlist.empty:
        return (
            "Nothing here clears the bar of a well-supported transformation that also holds "
            "the property profile. That is a finding, not a gap: the changes with precedent "
            "on this target either do not apply to this compound or cost more than they add."
        )

    lines = []
    for _, row in shortlist.iterrows():
        note = []
        if row.get("already_made"):
            note.append(f"already exists ({row.get('chembl_id')}) -- check if it is purchasable")
        if row["d_clogp"] is not None and row["d_clogp"] < -0.3:
            note.append(f"drops cLogP by {abs(row['d_clogp']):.2f}")
        if row["sa_score"] and row["sa_score"] > 4.5:
            note.append(f"harder to make (SA {row['sa_score']})")
        suffix = f" — {'; '.join(note)}" if note else ""
        lines.append(
            f"1. `{row['smiles']}`\n"
            f"   via **{row['transformation']}** ({row['evidence']}){suffix}"
        )

    return (
        "\n".join(lines)
        + "\n\nOrdered by evidence strength, then by whether the analog holds the property "
        "profile. Each is a single change from the parent, which is deliberate: on a "
        "compound worth optimizing, conservative substitutions preserve the binding mode "
        "that made it interesting. Larger jumps belong in a separate scaffold-hop exercise."
    )
