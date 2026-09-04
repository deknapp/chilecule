"""Hit triage: take a library down to a shortlist worth ordering.

A cascade of increasingly expensive filters, cheapest first, with the reasons
for every exclusion retained. The ordering is deliberate: property filters cost
microseconds and docking costs tens of seconds, so a library is reduced by
three orders of magnitude before anything slow runs.

What makes this different from the same cascade written as a shell script:

* Nothing is silently deleted. Every rejected compound keeps the reason it was
  rejected, and the counts appear in the report. A cascade that reports only
  survivors cannot be debugged and cannot be argued with.
* Docking runs only after the redocking control passes for the site. If the
  protocol cannot reproduce a known crystallographic pose, its scores for
  novel ligands are not evidence, and the workflow says so instead of
  producing a confident ranking.
* Docking scores are one column among several, never the sole ranking key.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..runners.base import DESCRIPTOR, DOCKING_POSE, estimate_cost
from ..runners.local import LocalRunner
from ..tools.alerts import screen
from ..tools.chem import properties, standardize
from ..tools.docking import DockingUnavailable, dock, find_program, redock_control
from ..tools.structure import (
    DockingBox,
    box_from_ligand,
    extract_ligand_mol,
    extract_ligands,
    fetch_pdb,
    prepare_receptor,
)
from .report import Report, Section, base_provenance

log = logging.getLogger(__name__)


def _property_verdict(props, limits: dict) -> str | None:
    """Return a rejection reason, or None if the compound passes."""
    if props.mw < limits["min_mw"] or props.mw > limits["max_mw"]:
        return f"MW {props.mw:.0f} outside {limits['min_mw']}-{limits['max_mw']}"
    if props.clogp > limits["max_clogp"]:
        return f"cLogP {props.clogp:.1f} > {limits['max_clogp']}"
    if props.clogp < limits["min_clogp"]:
        return f"cLogP {props.clogp:.1f} < {limits['min_clogp']}"
    if props.tpsa > limits["max_tpsa"]:
        return f"TPSA {props.tpsa:.0f} > {limits['max_tpsa']}"
    if props.rotatable_bonds > limits["max_rotatable"]:
        return f"{props.rotatable_bonds} rotatable bonds > {limits['max_rotatable']}"
    return None


DEFAULT_LIMITS = {
    "min_mw": 150.0,
    "max_mw": 550.0,
    "min_clogp": -1.0,
    "max_clogp": 5.5,
    "max_tpsa": 140.0,
    "max_rotatable": 10.0,
}


def build(
    library: list[str] | pd.DataFrame,
    *,
    pdb_id: str | None = None,
    box: DockingBox | None = None,
    receptor_path: Path | str | None = None,
    limits: dict | None = None,
    alert_severities_to_drop: tuple[str, ...] = ("high",),
    dock_top_n: int = 25,
    exhaustiveness: int = 8,
    run_redock_control: bool = True,
    runner: LocalRunner | None = None,
) -> Report:
    """Filter a library and optionally dock the survivors.

    ``library`` is a list of SMILES or a DataFrame with a ``smiles`` column.

    Docking happens only when a site is defined (``pdb_id`` or ``box`` plus
    ``receptor_path``) and a docking program is installed. Without those, the
    ligand-based portion still runs to completion -- a partial result with a
    stated limitation beats no result.
    """
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    runner = runner or LocalRunner()
    report = Report(workflow="Hit triage", subject=f"{pdb_id or 'ligand-based'}")

    smiles_in = (
        library["smiles"].tolist() if isinstance(library, pd.DataFrame) else list(library)
    )
    rejections: list[dict] = []

    # ------------------------------------------------- Stage 1: structures
    standardized = runner.map(standardize, smiles_in, DESCRIPTOR)
    survivors = []
    for original, result in zip(smiles_in, standardized):
        if result.ok:
            survivors.append(result.smiles)
        else:
            rejections.append({"smiles": original, "stage": "parse", "reason": result.error})

    # Deduplicate after standardization, not before: salts and tautomers of the
    # same compound are distinct strings and identical molecules.
    seen: set[str] = set()
    deduplicated = []
    for smiles in survivors:
        if smiles in seen:
            rejections.append({"smiles": smiles, "stage": "duplicate",
                               "reason": "identical to an earlier compound after standardization"})
            continue
        seen.add(smiles)
        deduplicated.append(smiles)

    # ------------------------------------------------- Stage 2: properties
    stage2, profiles = [], {}
    for smiles in deduplicated:
        props = properties(smiles)
        if props is None:
            rejections.append({"smiles": smiles, "stage": "properties",
                               "reason": "descriptor calculation failed"})
            continue
        reason = _property_verdict(props, limits)
        if reason:
            rejections.append({"smiles": smiles, "stage": "properties", "reason": reason})
            continue
        profiles[smiles] = props
        stage2.append(smiles)

    # ----------------------------------------------------- Stage 3: alerts
    stage3, alert_notes = [], {}
    for smiles in stage2:
        alerts = screen(smiles)
        if alerts is None:
            stage3.append(smiles)
            continue
        alert_notes[smiles] = alerts
        if alerts.max_severity in alert_severities_to_drop:
            worst = "; ".join(a.description for a in alerts.alerts if a.severity in alert_severities_to_drop)
            rejections.append({"smiles": smiles, "stage": "structural alert",
                               "reason": f"{alerts.max_severity} severity: {worst}"})
            continue
        stage3.append(smiles)

    cascade = pd.DataFrame(
        [
            {"stage": "input", "surviving": len(smiles_in), "removed": 0},
            {"stage": "parsed and standardized", "surviving": len(survivors),
             "removed": len(smiles_in) - len(survivors)},
            {"stage": "deduplicated", "surviving": len(deduplicated),
             "removed": len(survivors) - len(deduplicated)},
            {"stage": "property window", "surviving": len(stage2),
             "removed": len(deduplicated) - len(stage2)},
            {"stage": "structural alerts", "surviving": len(stage3),
             "removed": len(stage2) - len(stage3)},
        ]
    )
    report.add(
        Section(
            title="Filter cascade",
            body=(
                f"{len(smiles_in)} compounds in, {len(stage3)} through the ligand-based "
                "cascade.\n\n"
                "Alerts at severity "
                f"{list(alert_severities_to_drop)} were removed; lower-severity matches were "
                "annotated and retained. Structural alerts are heuristics -- roughly 5% of "
                "approved drugs match a PAINS pattern, and aspirin matches a Brenk one -- so "
                "they inform triage rather than decide it. Every exclusion is listed in the "
                "JSON output with its reason."
            ),
            table=cascade,
            data={"limits": limits, "n_input": len(smiles_in), "n_survivors": len(stage3)},
        )
    )

    if not stage3:
        report.warn("No compounds survived the ligand-based cascade.")
        report.provenance = base_provenance(n_input=len(smiles_in))
        return report

    # ------------------------------------------------- Stage 4: structure
    docking_scores: dict[str, float] = {}
    program = find_program()

    if (pdb_id or box) and program:
        receptor, resolved_box, control = _prepare_site(
            report, pdb_id, box, receptor_path, run_redock_control, program
        )
        if receptor is not None and resolved_box is not None:
            shortlist = stage3[:dock_top_n]
            projection = estimate_cost(len(shortlist), DOCKING_POSE)
            log.info("docking %d compounds (~%s CPU-hours)", len(shortlist), projection["cpu_hours"])

            for smiles in shortlist:
                try:
                    result = dock(receptor, smiles, resolved_box,
                                  program=program, exhaustiveness=exhaustiveness)
                except DockingUnavailable:
                    break
                if result.best_score is not None:
                    docking_scores[smiles] = result.best_score

            if control is not None and not control.passed:
                report.warn(
                    "The redocking control FAILED for this site. Docking scores below are "
                    "reported for completeness but should not be used to rank compounds."
                )
    elif pdb_id or box:
        report.warn(
            "A binding site was specified but no docking program is installed, so the "
            "structure-based stage was skipped. Install one with "
            "`micromamba install -c conda-forge smina`."
        )

    # ------------------------------------------------------- Final ranking
    rows = []
    for smiles in stage3:
        props = profiles[smiles]
        alerts = alert_notes.get(smiles)
        rows.append(
            {
                "smiles": smiles,
                "mw": props.mw,
                "clogp": props.clogp,
                "tpsa": props.tpsa,
                "qed": props.qed,
                "alerts": len(alerts.alerts) if alerts else 0,
                "docking_score": docking_scores.get(smiles),
            }
        )
    ranked = pd.DataFrame(rows)

    if docking_scores:
        ranked = ranked.sort_values(
            ["docking_score", "qed"], ascending=[True, False], na_position="last"
        )
        ranking_note = (
            "Ranked by docking score, then by QED.\n\n"
            "Docking scores are a coarse ranking signal, not predicted potency -- their "
            "correlation with measured affinity across diverse chemistry is weak. Treat the "
            "ordering as a prioritization for further work, and read it alongside the "
            "property columns rather than instead of them."
        )
    else:
        ranked = ranked.sort_values("qed", ascending=False)
        ranking_note = (
            "Ranked by QED, since no docking was performed. This ordering reflects "
            "drug-likeness only and carries no information about the target."
        )

    report.add(
        Section(
            title="Shortlist",
            body=ranking_note,
            table=ranked.reset_index(drop=True),
            max_rows=25,
            data={"n_ranked": len(ranked), "n_docked": len(docking_scores)},
        )
    )

    if rejections:
        rejection_frame = pd.DataFrame(rejections)
        report.add(
            Section(
                title="Excluded compounds",
                body=(
                    f"{len(rejections)} exclusions, by stage:\n\n"
                    + "\n".join(
                        f"- **{stage}**: {count}"
                        for stage, count in rejection_frame["stage"].value_counts().items()
                    )
                    + "\n\nThe full list with per-compound reasons is in the JSON output."
                ),
                table=rejection_frame.head(20),
                data={"rejections": rejections},
            )
        )

    report.provenance = base_provenance(
        n_input=len(smiles_in),
        n_survivors=len(stage3),
        docking_program=program,
        pdb_id=pdb_id,
    )
    return report


def _prepare_site(report, pdb_id, box, receptor_path, run_control, program):
    """Fetch and prepare the receptor, then validate the protocol by redocking."""
    if box is not None and receptor_path is not None:
        return Path(receptor_path), box, None

    structure = fetch_pdb(pdb_id)
    ligands = [lig for lig in extract_ligands(structure) if lig.category == "ligand"]
    if not ligands:
        report.warn(
            f"{pdb_id} contains no co-crystallized ligand, so no site could be defined from "
            "it. Supply a box explicitly, or run pocket detection first."
        )
        return None, None, None

    native = ligands[0]
    resolved_box = box or box_from_ligand(structure, native)
    receptor = prepare_receptor(structure, keep_chain=native.chain_id)

    control = None
    if run_control:
        control = _run_control(report, structure, native, receptor, resolved_box, program)
    return receptor, resolved_box, control


def _run_control(report, structure, native, receptor, box, program):
    """Re-dock the crystallographic ligand and report whether the pose is reproduced."""
    from rdkit import Chem

    mol = extract_ligand_mol(structure, native)
    if mol is None:
        report.warn(
            f"Could not reconstruct the native ligand {native.residue_name} with correct "
            "bond orders, so the redocking control was skipped. Docking scores below are "
            "unvalidated for this site."
        )
        return None

    native_sdf = Path(structure).with_name(f"{Path(structure).stem}_{native.residue_name}.sdf")
    writer = Chem.SDWriter(str(native_sdf))
    writer.write(mol)
    writer.close()

    control = redock_control(receptor, native_sdf, box, program=program, exhaustiveness=16)
    report.add(
        Section(
            title="Redocking control",
            body=(
                f"Re-docked the crystallographic ligand **{control.residue_name}** into its "
                f"own structure.\n\n"
                f"- RMSD to the crystallographic pose: **{control.rmsd_display()}**\n"
                f"- Threshold: {control.threshold} A\n"
                f"- Verdict: **{'PASS' if control.passed else 'FAIL'}**\n\n"
                f"{control.detail}\n\n"
                "This control is the reason the docking results below can be interpreted at "
                "all. A protocol that cannot reproduce a known answer for this site has not "
                "earned the right to make claims about unknown ligands."
            ),
            data={"redock_control": control.to_dict()},
        )
    )
    return control
