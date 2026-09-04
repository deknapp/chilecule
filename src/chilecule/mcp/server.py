"""MCP server exposing the chilecule tool layer.

Any MCP client -- Claude Code, Claude Desktop, or a custom agent -- can drive
these tools. Nothing here contains an LLM call; the server is the boundary
between deterministic science and whatever model is holding the other end.

Two decisions worth explaining, because both cost tokens and both pay for
themselves:

**Tool descriptions carry the caveats.** ``dock_molecule`` says in its own
description that docking scores are not predicted affinities. A model that has
only seen the function signature will confidently report a score of -9.2 as
"strong predicted binding". Putting the limitation where the model actually
reads it is the difference between a tool that informs and one that misleads.

**Results are typed, with units and provenance, never bare numbers.** A tool
that returns ``-9.2`` invites the model to invent an interpretation. One that
returns ``{"score": -9.2, "units": "kcal/mol", "interpretation": "ranking signal
only", "control_passed": true}`` does not.

**Arguments are validated before any work starts.** Inputs are pydantic-typed
(see :mod:`chilecule.mcp.models`), so an ``exhaustiveness`` of 10000 or a
``pdb_id`` of ``"the EGFR structure"`` comes back as a correctable error in
milliseconds instead of a subprocess that never returns. The constraints also
travel to the model in the published JSON Schema, which is prompt surface: a
bare ``{"type": "integer"}`` makes the model guess.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from pydantic import Field

# The MCP Python SDK renamed FastMCP to MCPServer in 2.0. Support both, since
# plenty of environments still pin 1.x, and report the two failure modes
# distinctly -- "not installed" and "installed but incompatible" have different
# fixes, and conflating them sends people to reinstall a package they have.
try:
    from mcp.server.mcpserver import MCPServer as _Server  # mcp >= 2.0
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP as _Server  # mcp 1.x
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "The MCP extra is not installed. Install it with: "
            "pip install 'chilecule[mcp]'"
        ) from exc

from .. import __version__
from ..tools import alerts as alerts_module
from ..tools import chem, chembl, liabilities, lookup, pockets, sar, structure, synth
from ..tools import score as score_module
from .models import (
    ActivesResult,
    AlertMatch,
    AlertsResult,
    AnalogDesignResult,
    AnalogModel,
    ChainId,
    ChemblId,
    ComparisonResult,
    DockingToolResult,
    EfficiencyResult,
    ErrorResult,
    Exhaustiveness,
    LiabilitiesResult,
    LiabilityModel,
    LigandModel,
    MaxRecords,
    NeighbourModel,
    NoveltyResult,
    PdbId,
    PocketModel,
    PocketsResult,
    PotencyNanomolar,
    PromiscuityResult,
    PromiscuityTargetModel,
    PropertiesResult,
    RedockControl,
    ScorecardComponent,
    ScorecardResultModel,
    Smiles,
    StandardizationResult,
    StructureHitModel,
    StructureInspection,
    StructureSearchResult,
    SynthesisResult,
    TargetHit,
    TargetSearchResult,
    TopN,
    TransformationModel,
)

mcp = _Server(
    "chilecule",
    version=__version__,
    instructions=(
        "Open-source drug discovery tools. Standardize structures before comparing or "
        "deduplicating them. Docking scores rank, they do not predict potency. Never "
        "report an enrichment figure without the ceiling it is measured against."
    ),
)

_chembl_client: chembl.ChemblClient | None = None


def _client() -> chembl.ChemblClient:
    global _chembl_client
    if _chembl_client is None:
        _chembl_client = chembl.ChemblClient()
    return _chembl_client


def _json(payload: Any) -> str:
    return json.dumps(payload, indent=2, default=str)


# ------------------------------------------------------------------ chemistry


@mcp.tool()
def standardize_molecule(smiles: Smiles) -> StandardizationResult:
    """Normalize a molecule to a canonical form: strip salts and solvates, neutralize
    charges, and canonicalize the tautomer.

    Run this before comparing, deduplicating, or joining on structure. The same
    compound deposited as a hydrochloride salt, a zwitterion, and a neutral
    tautomer produces three different InChIKeys and three different
    fingerprints, which silently inflates dataset size and corrupts any split
    built on structural identity.
    """
    result = chem.standardize(smiles)
    return StandardizationResult(
        input_smiles=result.input_smiles,
        standardized_smiles=result.smiles,
        inchikey=result.inchikey,
        changed=result.changed,
        error=result.error,
    )


@mcp.tool()
def molecule_properties(smiles: Smiles) -> PropertiesResult:
    """Compute physicochemical descriptors: MW, cLogP, TPSA, HBD/HBA, rotatable bonds,
    aromatic rings, fraction sp3, QED, Lipinski violations, and Veber pass.

    These are interpretable and comparable across targets, unlike a docking
    score. Lipinski violations are a guideline with well-known exceptions, not
    a filter -- report them, do not enforce them.
    """
    props = chem.properties(smiles)
    return PropertiesResult(**props.to_dict())


@mcp.tool()
def structural_alerts(smiles: Smiles) -> AlertsResult:
    """Screen a molecule against PAINS (A/B/C), Brenk, NIH, and ZINC alert catalogs.

    Returns every match with its catalog, severity, and literature reference.

    Treat these as annotations, not verdicts. Roughly 5% of approved drugs
    match a PAINS pattern and aspirin matches a Brenk one. PAINS_A carries the
    strongest evidence; PAINS_C was fit to very few observations. If you exclude
    a compound on an alert, say which catalog fired and why.
    """
    report = alerts_module.screen(smiles)
    return AlertsResult(
        smiles=report.smiles,
        clean=report.clean,
        max_severity=report.max_severity,
        alerts=[
            AlertMatch(
                catalog=alert.catalog,
                description=alert.description,
                severity=alert.severity,
                reference=alert.reference,
            )
            for alert in report.alerts
        ],
    )


@mcp.tool()
def ligand_efficiency_metrics(smiles: Smiles, potency_nm: PotencyNanomolar) -> EfficiencyResult:
    """Compute ligand efficiency (LE) and lipophilic efficiency (LLE) for a compound
    at a measured potency in nanomolar.

    LE is binding energy per heavy atom -- it stops you rewarding a compound for
    being large. LLE (pIC50 - cLogP) tracks whether potency is being bought with
    lipophilicity, which is the classic hit-to-lead failure mode. A moderately
    potent efficient fragment is usually a better starting point than a potent
    inefficient large molecule.
    """
    props = chem.properties(smiles)
    return EfficiencyResult(
        smiles=props.smiles,
        potency_nm=potency_nm,
        heavy_atoms=props.heavy_atoms,
        clogp=props.clogp,
        ligand_efficiency=chem.ligand_efficiency(potency_nm, props.heavy_atoms),
        lipophilic_efficiency=chem.lipophilic_efficiency(potency_nm, props.clogp),
    )


@mcp.tool()
def compare_molecules(smiles_a: Smiles, smiles_b: Smiles) -> ComparisonResult:
    """Tanimoto similarity on ECFP4 fingerprints between two molecules, with their
    Bemis-Murcko scaffolds.
    """
    return ComparisonResult(
        similarity=sar.similarity(smiles_a, smiles_b),
        scaffold_a=sar.murcko_scaffold(smiles_a),
        scaffold_b=sar.murcko_scaffold(smiles_b),
    )


# ----------------------------------------------------------------- bioactivity


@mcp.tool()
def find_chembl_target(query: str) -> TargetSearchResult | ErrorResult:
    """Resolve a gene symbol, protein name, or UniProt accession to ChEMBL targets.

    Single-protein human targets are returned first. ChEMBL also stores protein
    complexes, cell lines, and whole organisms as targets; selecting one of
    those yields activity data with no structural interpretation.
    """
    frame = _client().find_targets(query)
    if frame.empty:
        return ErrorResult(
            error=f"no ChEMBL target matched {query!r}",
            suggestion="Try an official gene symbol (EGFR), or a UniProt accession (P00533).",
        )
    return TargetSearchResult(
        results=[
            TargetHit(
                target_chembl_id=row.get("target_chembl_id"),
                pref_name=row.get("pref_name"),
                target_type=row.get("target_type"),
                organism=row.get("organism"),
                uniprot=row.get("uniprot"),
                n_components=row.get("n_components"),
            )
            for row in frame.head(10).to_dict(orient="records")
        ]
    )


@mcp.tool()
def get_target_actives(
    target_chembl_id: ChemblId, max_records: MaxRecords = 3000
) -> ActivesResult | ErrorResult:
    """Fetch and curate bioactivity data for a ChEMBL target.

    Returns one row per compound with median potency on the pChEMBL scale, plus
    a full audit of what curation removed and why -- censored measurements,
    curator-flagged records, wrong-species data, unconvertible units,
    thermodynamically implausible potencies, and compounds whose replicate
    measurements disagree.

    Read the audit. If a large fraction of records was removed as inconsistent,
    the remaining SAR is correspondingly less reliable.
    """
    client = _client()
    release = client.release()
    raw = client.fetch_activities(target_chembl_id, max_records=max_records)
    if raw.empty:
        return ErrorResult(
            error=f"no activity records for {target_chembl_id}",
            suggestion="Confirm the identifier with find_chembl_target.",
        )

    curated, report = chembl.curate_activities(
        raw, target_chembl_id=target_chembl_id, chembl_release=release
    )
    inactives = chembl.censored_inactives(raw)
    return ActivesResult(
        curation=report.to_dict(),
        n_confirmed_inactives=len(inactives),
        compounds=curated.head(100).to_dict(orient="records"),
    )


# ------------------------------------------------------------------ structure


@mcp.tool()
def find_structures(uniprot_accession: str, top: TopN = 10) -> StructureSearchResult | ErrorResult:
    """Find and rank PDB structures for a UniProt accession by suitability for docking.

    Ranks on resolution first, method second, sequence coverage last. A 1.9 A
    crystal structure of the relevant domain is a better docking target than a
    3.1 A cryo-EM model of the full-length protein, however much more of the
    sequence the latter covers -- below about 3 A, side-chain positions are
    modelled rather than observed.
    """
    hits = structure.rank_structures(structure.structures_for_uniprot(uniprot_accession))
    if not hits:
        return ErrorResult(
            error=f"no experimental structures found for {uniprot_accession}",
            suggestion=(
                "Confirm the accession is a UniProt ID such as P00533. Without a "
                "structure, only ligand-based workflows apply to this target."
            ),
        )
    return StructureSearchResult(
        results=[StructureHitModel(**h.to_dict()) for h in hits[:top]]
    )


@mcp.tool()
def inspect_structure(pdb_id: PdbId) -> StructureInspection:
    """Download a PDB entry and list its bound ligands, cofactors, and metals.

    Categories matter for receptor preparation: crystallization additives should
    be removed, but catalytic metals must be retained. Stripping the Mg from a
    kinase nucleotide site or the Zn from a metalloproteinase deletes the
    electrostatics the ligand binds to, and docking into the result produces
    poses that cannot be correct.
    """
    path = structure.fetch_pdb(pdb_id)
    ligands = structure.extract_ligands(path)
    return StructureInspection(
        pdb_id=pdb_id.upper(),
        path=str(path),
        ligands=[LigandModel(**lig.to_dict()) for lig in ligands[:25]],
    )


@mcp.tool()
def detect_pockets(
    pdb_id: PdbId, chain: ChainId = None, top: TopN = 5
) -> PocketsResult | ErrorResult:
    """Detect candidate binding sites with fpocket, ranked by druggability.

    Use this only when there is no co-crystallized ligand. A pocket defined by
    chemical matter the protein demonstrably binds beats any prediction.

    A typical protein returns 15-40 cavities of which one or two can bind a
    drug-like molecule, so the druggability score is triage, not an oracle.
    """
    path = structure.fetch_pdb(pdb_id)
    receptor = structure.prepare_receptor(path, keep_chain=chain)
    try:
        found = pockets.find_pockets(receptor, max_pockets=top)
    except pockets.FpocketUnavailable as exc:
        return ErrorResult(
            error=str(exc),
            suggestion="micromamba install -c conda-forge fpocket",
        )
    return PocketsResult(
        pdb_id=pdb_id.upper(),
        pockets=[PocketModel(**p.to_dict()) for p in found],
    )


@mcp.tool()
def dock_molecule(
    smiles: Smiles,
    pdb_id: PdbId,
    exhaustiveness: Exhaustiveness = 8,
    run_control: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "Re-dock the crystallographic ligand first and report RMSD to its known "
                "pose. Leave on: without it there is no evidence the protocol works for "
                "this site. Costs one extra docking run."
            ),
        ),
    ] = True,
) -> DockingToolResult | ErrorResult:
    """Dock a molecule into a PDB structure's co-crystallized ligand site.

    IMPORTANT -- how to report the result. A docking score is NOT a predicted
    binding affinity. Its correlation with measured affinity across diverse
    chemistry is weak (Pearson r typically 0.3-0.5), which is not enough to rank
    a congeneric series. Docking is good at generating plausible poses and at
    coarse enrichment over a large library; it is bad at telling you which of
    two analogs is more potent. Describe scores as a ranking signal, never as
    predicted potency.

    With ``run_control`` set, the crystallographic ligand is re-docked into its
    own structure first. If that fails to reproduce the known pose within 2 A,
    the protocol has not reproduced a known answer for this site and its scores
    for novel ligands should not be trusted -- the response says so.
    """
    from ..tools import docking as docking_module

    program = docking_module.find_program()
    if program is None:
        return ErrorResult(
            error="no docking program on PATH",
            suggestion="micromamba install -c conda-forge smina",
        )

    path = structure.fetch_pdb(pdb_id)
    ligands = [lig for lig in structure.extract_ligands(path) if lig.category == "ligand"]
    if not ligands:
        return ErrorResult(
            error=f"{pdb_id.upper()} has no co-crystallized ligand to define a site",
            suggestion="Call detect_pockets and dock into a predicted pocket instead.",
        )

    native = ligands[0]
    box = structure.box_from_ligand(path, native)
    receptor = structure.prepare_receptor(path, keep_chain=native.chain_id)

    control_payload = None
    if run_control:
        from pathlib import Path

        from rdkit import Chem

        mol = structure.extract_ligand_mol(path, native)
        if mol is not None:
            native_sdf = Path(path).with_name(f"{Path(path).stem}_{native.residue_name}.sdf")
            writer = Chem.SDWriter(str(native_sdf))
            writer.write(mol)
            writer.close()
            control = docking_module.redock_control(
                receptor, native_sdf, box, program=program, exhaustiveness=16
            )
            control_payload = RedockControl(**control.to_dict())

    result = docking_module.dock(
        receptor, smiles, box, program=program, exhaustiveness=exhaustiveness
    )
    return DockingToolResult(
        smiles=smiles,
        pdb_id=pdb_id.upper(),
        site=box.to_dict(),
        program=result.program,
        best_score=result.best_score,
        n_poses=len(result.poses),
        redocking_control=control_payload,
        control_verdict=(
            None
            if control_payload is None
            else (
                "protocol reproduces the crystallographic pose"
                if control_payload.passed
                else "PROTOCOL FAILED ITS CONTROL -- do not rely on this score"
            )
        ),
        error=result.error,
    )




# ------------------------------------------------ molecule-in: profiling a compound


@mcp.tool()
def check_compound_novelty(smiles: Smiles) -> NoveltyResult | ErrorResult:
    """Has anyone made this? Exact-structure lookup in ChEMBL and PubChem by InChIKey,
    plus a near-neighbour search.

    Use this whenever you are asked whether a compound is known, novel, or a drug --
    do not answer from memory. The distinction between an exact match and a close
    neighbour matters: an exact hit means the compound exists and may be purchasable
    with data attached, while a 0.9-similar hit means you are inside somebody's series.

    Structural novelty only. A compound absent from both databases can still fall inside
    a Markush claim, so never present this as a freedom-to-operate opinion.
    """
    report = lookup.check_novelty(smiles)
    if report.error:
        return ErrorResult(error=report.error, suggestion="Check that the SMILES parses.")
    return NoveltyResult(
        smiles=report.smiles,
        inchikey=report.inchikey,
        is_known=report.is_known,
        verdict=report.verdict,
        chembl_id=report.chembl_id,
        chembl_name=report.chembl_name,
        max_phase=report.max_phase,
        pubchem_cid=report.pubchem_cid,
        nearest_neighbours=[NeighbourModel(**n) for n in report.nearest_neighbours],
    )


@mcp.tool()
def check_promiscuity(chembl_id: ChemblId) -> PromiscuityResult | ErrorResult:
    """What else does this compound hit? Summarizes reported activity across every
    ChEMBL protein target.

    Read ``selectivity_window_log``, not the target count. The count is misleading:
    a kinase inhibitor that has been through kinome panels accumulates hundreds of
    target annotations and is not promiscuous in any troubling sense. What separates
    that from a frequent hitter is whether a primary target stands out. A wide window
    means a real primary target with a panel-annotation tail; a flat profile across many
    unrelated proteins is the signature of an aggregator, a reactive compound, or assay
    interference.

    Cell lines and other non-protein ChEMBL targets are excluded and counted separately.
    Get the ChEMBL id from check_compound_novelty.
    """
    report = lookup.check_promiscuity(chembl_id)
    if report.error:
        return ErrorResult(error=report.error)
    return PromiscuityResult(
        chembl_id=report.chembl_id,
        n_protein_targets=report.n_targets,
        n_non_protein_excluded=report.n_non_protein_excluded,
        selectivity_window_log=report.selectivity_window,
        n_activity_records=report.n_records,
        assessment=report.assessment,
        targets=[PromiscuityTargetModel(**t) for t in report.targets],
    )


@mcp.tool()
def assess_synthesis(smiles: Smiles) -> SynthesisResult | ErrorResult:
    """Can it be made? Synthetic accessibility plus structural complexity flags.

    SA score runs 1 (trivial) to 10 (intractable) and measures fragment familiarity --
    whether the molecule is built from pieces that appear in known compounds. It is not
    a route prediction, so it is good at flagging exotic structures and blind to a
    familiar-looking molecule that needs awkward regiochemistry.

    The flags catch what the score misses: unassigned stereocentres imply a separation
    problem, and macrocycles, spiro centres and bridgeheads imply a hard synthesis
    however ordinary the fragments look.
    """
    assessment = synth.assess(smiles)
    if assessment is None:
        return ErrorResult(error="could not parse SMILES")
    payload = assessment.to_dict()
    payload.pop("caveat", None)
    return SynthesisResult(**payload)


@mcp.tool()
def score_compound(smiles: Smiles, profile: str = "oral") -> ScorecardResultModel | ErrorResult:
    """Multi-parameter scorecard. Profiles: 'oral' (oral small molecule) or 'lead_like'
    (fragment-to-lead space, with headroom left for optimization).

    Report ``limiting_property``, not just the score. "0.42" tells a chemist nothing;
    "limited by cLogP at 5.8, target below 4" is a design instruction. The score is a
    ranking key, the limiting property is the actionable output.

    The windows are literature defaults and are project-dependent. An inhaled compound
    and a CNS agent do not share a TPSA target, so treat the numbers as a starting point
    rather than a standard.
    """
    try:
        result = score_module.score(smiles, profile=profile)
    except ValueError as exc:
        return ErrorResult(error=str(exc), suggestion="Use profile 'oral' or 'lead_like'.")
    if result is None:
        return ErrorResult(error="could not parse SMILES")
    return ScorecardResultModel(
        smiles=result.smiles,
        profile=result.profile,
        score=round(result.score, 3),
        verdict=result.verdict,
        limiting_property=result.limiting["property"] if result.limiting else None,
        components=[ScorecardComponent(**c) for c in result.components],
    )


@mcp.tool()
def profile_compound(smiles: Smiles, check_databases: bool = True) -> str:
    """Everything worth knowing about one compound, in a single call.

    Composes standardization, physicochemical properties, structural alerts, synthetic
    accessibility, the oral scorecard, database novelty and promiscuity. Prefer this over
    calling the individual tools one by one when the question is open-ended
    ("tell me about this compound"), and use the individual tools when you need one
    specific answer.

    Set check_databases=False to skip the network lookups when you only need computed
    properties and the answer needs to be fast.
    """
    from ..workflows.profile import profile_one

    return _json(profile_one(smiles, check_databases=check_databases))


# ------------------------------------------------ molecule-in: designing the next one


@mcp.tool()
def design_analogs(
    parent_smiles: Smiles,
    target: str,
    max_analogs: int = 20,
    min_occurrences: int = 4,
) -> AnalogDesignResult | ErrorResult:
    """What should I make next? Proposes analogs of a hit using transformations that
    medicinal chemists have already made against this target.

    These are NOT generated molecules. Every proposal comes from a matched molecular pair
    mined from ChEMBL for this target, and carries the evidence: how many times that exact
    swap was made and what it did to potency.

    How to report the result. ``median_delta`` is what the transformation did in the
    contexts where it was observed -- it is precedent, not a prediction for this parent.
    Always read ``reliability`` alongside it: a large median with a large spread means the
    transformation is context-dependent and will not necessarily reproduce here. When
    ``in_evidence_domain`` is false, the expected effects read "no decision" and you must
    report them that way, because a matched-pair median does not transfer across
    chemotypes.

    ``target`` is a gene symbol or ChEMBL id, e.g. 'EGFR' or 'CHEMBL203'.
    """
    from ..tools.chembl import curate_activities
    from ..tools.design import enumerate_analogs, mine_transformations

    client = _client()
    targets = client.find_targets(target)
    if targets.empty:
        return ErrorResult(
            error=f"no ChEMBL target matched {target!r}",
            suggestion="Use a gene symbol such as EGFR, or resolve it with find_chembl_target.",
        )

    chembl_target = targets.iloc[0]
    raw = client.fetch_activities(chembl_target["target_chembl_id"], max_records=5000)
    actives, _ = curate_activities(
        raw, target_chembl_id=chembl_target["target_chembl_id"],
        chembl_release=client.release(),
    )
    transformations = mine_transformations(actives, min_occurrences=min_occurrences)
    if not transformations:
        return ErrorResult(
            error=(
                f"no transformation was observed at least {min_occurrences} times in "
                f"{len(actives)} curated compounds for {target}"
            ),
            suggestion=(
                "Lower min_occurrences, or accept that this target's chemical matter is "
                "too diverse for matched-pair design. That is a real answer, not a failure."
            ),
        )

    proposals = enumerate_analogs(
        parent_smiles,
        transformations,
        evidence_compounds=actives["smiles"].head(500).tolist(),
        max_total=max_analogs,
    )
    if not proposals:
        return ErrorResult(
            error="none of the mined transformations apply to this parent",
            suggestion=(
                "No fragment of the parent matches the left-hand side of any transformation "
                "with a track record on this target. The changes that worked here were made "
                "at positions this compound does not have."
            ),
        )

    analogs = []
    for proposal in proposals:
        props = chem.properties(proposal.smiles)
        synthesis = synth.assess(proposal.smiles)
        alert_report = alerts_module.screen(proposal.smiles)
        analogs.append(
            AnalogModel(
                smiles=proposal.smiles,
                transformation=proposal.transformation.label,
                expected_effect=proposal.expected_effect,
                n_pairs=proposal.transformation.n_pairs,
                median_delta=proposal.transformation.median_delta,
                reliability=proposal.transformation.reliability,
                similarity_to_parent=proposal.similarity_to_parent,
                mw=props.mw if props else None,
                clogp=props.clogp if props else None,
                sa_score=round(synthesis.sa_score, 2) if synthesis else None,
                alerts=len(alert_report.alerts) if alert_report else 0,
            )
        )

    return AnalogDesignResult(
        parent=proposals[0].parent_smiles,
        target=f"{chembl_target['target_chembl_id']} ({chembl_target['pref_name']})",
        n_transformations_mined=len(transformations),
        n_evidence_compounds=len(actives),
        in_evidence_domain=proposals[0].in_evidence_domain,
        domain_note=proposals[0].domain_note,
        transformations=[TransformationModel(**t.to_dict()) for t in transformations[:12]],
        analogs=analogs,
    )


@mcp.tool()
def check_liabilities(smiles: Smiles) -> LiabilitiesResult | ErrorResult:
    """Developability liabilities visible from structure: hERG, phospholipidosis,
    solubility, permeability, metabolic soft spots and reactive-metabolite precursors.

    Each flag names a structural feature and why it matters -- "basic amine with cLogP
    4.8 and two aromatic rings is the canonical hERG pharmacophore" -- together with the
    assay that would settle it.

    Report these as risks to test, never as predictions, and never recommend deleting a
    compound on a flag alone. Plenty of marketed drugs carry several: chloroquine is
    both a hERG risk and a classic cationic amphiphile. The useful output is which assay
    to run, not a verdict.

    Complements structural_alerts, which covers assay-interference and reactive-group
    catalogs. This covers ADMET and safety pharmacology.
    """
    report = liabilities.screen(smiles)
    if report is None:
        return ErrorResult(error="could not parse SMILES")
    payload = report.to_dict()
    payload.pop("caveat", None)
    payload["liabilities"] = [LiabilityModel(**item) for item in payload["liabilities"]]
    return LiabilitiesResult(**payload)


# The entrypoint stays at the very bottom of this module, deliberately.
#
# Running the server as `python -m chilecule.mcp.server` executes the module top
# to bottom, so anything defined below `if __name__ == "__main__"` is never
# reached before run() blocks. Tools appended after this block register fine on
# an in-process import and are silently missing from the served tool list --
# which is exactly what happened, and what the allowlist test caught.
def run(transport: str = "stdio") -> None:
    """Start the MCP server."""
    mcp.run(transport=transport)


if __name__ == "__main__":
    run()
