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
from ..tools import chem, chembl, pockets, sar, structure
from .models import (
    ActivesResult,
    AlertMatch,
    AlertsResult,
    ChainId,
    ChemblId,
    ComparisonResult,
    DockingToolResult,
    EfficiencyResult,
    ErrorResult,
    Exhaustiveness,
    LigandModel,
    MaxRecords,
    PdbId,
    PocketModel,
    PocketsResult,
    PotencyNanomolar,
    PropertiesResult,
    RedockControl,
    Smiles,
    StandardizationResult,
    StructureHitModel,
    StructureInspection,
    StructureSearchResult,
    TargetHit,
    TargetSearchResult,
    TopN,
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


def run(transport: str = "stdio") -> None:
    """Start the MCP server."""
    mcp.run(transport=transport)


if __name__ == "__main__":
    run()
