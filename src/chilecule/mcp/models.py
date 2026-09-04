"""Typed schemas for the agent-facing tool boundary.

Why pydantic lives here and nowhere else in this project
-------------------------------------------------------

The scientific layer (:mod:`chilecule.tools`, :mod:`chilecule.bench`) uses
plain dataclasses. Its inputs come from Python code that a person wrote and a
type checker has already seen, so runtime validation there would buy nothing.

This module is the one boundary in the codebase where arguments are generated
by a language model. That is a genuinely different situation, and it justifies
two things that would be over-engineering anywhere else:

**Validation before the expensive work starts.** A model that passes
``exhaustiveness=10000`` is asking for a docking run that will not finish
today. Caught at the boundary, that is an immediate, correctable error; caught
by the wall clock, it is a hung agent. The same applies to a ``pdb_id`` of
``"the EGFR structure"``, which would otherwise become a 404 several calls
later.

**A schema the model can actually read.** Without field descriptions and
bounds, MCP publishes ``{"exhaustiveness": {"type": "integer", "default": 8}}``
and the model has to guess what a reasonable value is. With them it is told the
range and what the parameter costs. Tool descriptions and argument schemas are
prompt surface, and thin schemas are a silent, recurring accuracy tax.

A note on what this does *not* do: it validates shape, range, and parseability.
It cannot catch a semantically wrong operation on well-formed data -- stripping
a catalytic metal, or breaking tie handling in a ranking metric. Those are
caught by controls and tests, which is where this project's real safety net
lives. Types are the cheap layer, not the load-bearing one.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from ..tools.chem import parse_smiles

# --------------------------------------------------------------- input types


def _validate_smiles(value: str) -> str:
    """Reject anything RDKit cannot parse, with an error a model can act on.

    The common failure is a model passing a compound name or an InChI where a
    SMILES was requested. ``"aspirin"`` happens to be a syntactically valid
    string, so only an actual parse attempt catches it -- and saying so plainly
    is what lets the model correct itself on the next turn instead of retrying
    the same call.
    """
    text = (value or "").strip()
    if not text:
        raise ValueError("SMILES string is empty")
    if parse_smiles(text) is None:
        raise ValueError(
            f"{text!r} is not a valid SMILES string. If this is a compound name "
            "or an InChI, convert it to SMILES first -- this tool does not "
            "accept names."
        )
    return text


Smiles = Annotated[
    str,
    AfterValidator(_validate_smiles),
    Field(
        description=(
            "A SMILES string, e.g. 'CC(=O)Oc1ccccc1C(=O)O' for aspirin. "
            "Compound names and InChI are not accepted."
        ),
        examples=["CC(=O)Oc1ccccc1C(=O)O", "c1ccc(Nc2ncnc3ccccc23)cc1"],
    ),
]

PdbId = Annotated[
    str,
    Field(
        pattern=r"^[0-9][A-Za-z0-9]{3}$",
        description=(
            "A four-character PDB identifier such as '5CNN'. Always starts with "
            "a digit. Use find_structures to obtain one for a target."
        ),
        examples=["5CNN", "1M17"],
    ),
]

ChemblId = Annotated[
    str,
    Field(
        pattern=r"^CHEMBL[0-9]+$",
        description=(
            "A ChEMBL identifier such as 'CHEMBL203'. "
            "Use find_chembl_target to obtain one."
        ),
        examples=["CHEMBL203"],
    ),
]

ChainId = Annotated[
    str | None,
    Field(
        default=None,
        max_length=2,
        description="Single chain to restrict to, e.g. 'A'. Omit to use every chain.",
    ),
]

Exhaustiveness = Annotated[
    int,
    Field(
        default=8,
        ge=1,
        le=64,
        description=(
            "Docking search effort. Runtime scales roughly linearly. 8 is the "
            "screening default and is adequate for ranking; 16-32 is appropriate "
            "when a single pose matters. Above 32 the returns are negligible and "
            "the cost is not."
        ),
    ),
]

PotencyNanomolar = Annotated[
    float,
    Field(
        gt=0,
        le=1e9,
        description=(
            "Measured potency in nanomolar (IC50, Ki, or Kd). Must be positive; "
            "1e9 nM is 1 M, beyond which the number is not a measurement."
        ),
    ),
]

MaxRecords = Annotated[
    int,
    Field(
        default=3000,
        ge=100,
        le=20000,
        description=(
            "Cap on ChEMBL activity records fetched. Larger values are slower and "
            "place more load on a public EMBL-EBI service; 3000 is enough to "
            "characterize most targets."
        ),
    ),
]

TopN = Annotated[
    int,
    Field(default=5, ge=1, le=50, description="How many results to return, best first."),
]


# -------------------------------------------------------------- output types


class ToolResult(BaseModel):
    """Base for every tool response.

    ``model_config`` forbids extra fields so that a typo in a field name fails
    here, during development, rather than silently producing a response missing
    the key the model was told to expect.
    """

    model_config = ConfigDict(extra="forbid")


class StandardizationResult(ToolResult):
    input_smiles: str
    standardized_smiles: str | None = Field(
        description=(
            "Canonical SMILES after salt stripping, neutralization and tautomer "
            "canonicalization."
        )
    )
    inchikey: str | None = Field(description="InChIKey of the standardized structure.")
    changed: bool = Field(
        description="True if standardization altered the structure, e.g. a counterion was removed."
    )
    error: str | None = None


class PropertiesResult(ToolResult):
    smiles: str
    mw: float = Field(description="Molecular weight in daltons.")
    clogp: float = Field(description="Calculated octanol/water partition coefficient (Crippen).")
    tpsa: float = Field(description="Topological polar surface area in square angstroms.")
    hbd: int
    hba: int
    rotatable_bonds: int
    aromatic_rings: int
    heavy_atoms: int
    fraction_csp3: float
    formal_charge: int
    stereocenters: int
    qed: float = Field(description="Quantitative estimate of drug-likeness, 0-1.")
    rule_of_five_violations: int = Field(
        description="Lipinski violations. A guideline with well-known exceptions, not a filter."
    )
    veber_pass: bool


class AlertMatch(ToolResult):
    catalog: Literal["PAINS_A", "PAINS_B", "PAINS_C", "BRENK", "NIH", "ZINC", "UNKNOWN"]
    description: str
    severity: Literal["high", "medium", "low"]
    reference: str = ""


class AlertsResult(ToolResult):
    smiles: str
    clean: bool
    max_severity: Literal["high", "medium", "low"] | None
    alerts: list[AlertMatch]
    guidance: str = Field(
        default=(
            "Annotations, not verdicts. Roughly 5% of approved drugs match a PAINS "
            "pattern and aspirin matches a Brenk one. If you exclude a compound, name "
            "the catalog and justify it."
        )
    )


class EfficiencyResult(ToolResult):
    smiles: str
    potency_nm: float
    heavy_atoms: int
    clogp: float
    ligand_efficiency: float | None = Field(
        description="Binding energy per heavy atom, kcal/mol/HA. Above ~0.3 is a workable start."
    )
    lipophilic_efficiency: float | None = Field(
        description="pIC50 minus cLogP. Above 5 is healthy; a falling LLE signals trouble."
    )


class ComparisonResult(ToolResult):
    similarity: float | None = Field(description="Tanimoto similarity on ECFP4, 0-1.")
    fingerprint: str = "ECFP4 (Morgan radius 2, 2048 bits)"
    scaffold_a: str | None
    scaffold_b: str | None


class TargetHit(ToolResult):
    target_chembl_id: str | None
    pref_name: str | None
    target_type: str | None
    organism: str | None
    uniprot: str | None
    n_components: int | None = None


class TargetSearchResult(ToolResult):
    results: list[TargetHit]
    note: str = Field(
        default=(
            "SINGLE PROTEIN targets are listed first. Protein complexes, cell lines "
            "and whole organisms are also stored as ChEMBL targets and cannot be "
            "docked into."
        )
    )


class ActivesResult(ToolResult):
    curation: dict[str, Any] = Field(
        description="Full audit of what curation removed and why. Read it before trusting the SAR."
    )
    n_confirmed_inactives: int
    compounds: list[dict[str, Any]]
    license: str = (
        "ChEMBL data is CC BY-SA 3.0. Cite the release named in the curation audit."
    )


class StructureHitModel(ToolResult):
    pdb_id: str
    chain_id: str
    resolution: float | None
    coverage: float | None
    method: str
    score: float
    rationale: str


class StructureSearchResult(ToolResult):
    results: list[StructureHitModel]
    note: str = Field(
        default=(
            "Ranked for docking suitability: resolution first, method second, "
            "sequence coverage last. Below about 3 A, side-chain positions are "
            "modelled rather than observed."
        )
    )


class LigandModel(ToolResult):
    residue_name: str
    chain_id: str
    residue_number: int
    n_atoms: int
    centroid: list[float]
    category: Literal["ligand", "cofactor", "metal", "additive"]


class StructureInspection(ToolResult):
    pdb_id: str
    path: str
    ligands: list[LigandModel]
    note: str = Field(
        default=(
            "'ligand' defines a docking site. 'metal' and 'cofactor' are retained "
            "during receptor preparation -- stripping a catalytic metal deletes the "
            "electrostatics a ligand binds to. 'additive' is removed."
        )
    )


class PocketModel(ToolResult):
    rank: int
    score: float
    druggability: float
    volume_A3: float
    volume_score: float
    n_alpha_spheres: int
    hydrophobicity: float
    center: list[float]
    assessment: str


class PocketsResult(ToolResult):
    pdb_id: str
    pockets: list[PocketModel]
    note: str = Field(
        default=(
            "A typical protein has 15-40 cavities and one or two that bind a "
            "drug-like molecule. Use a co-crystallized ligand to define the site "
            "whenever one exists; this is for apo structures."
        )
    )


class RedockControl(ToolResult):
    ligand: str
    rmsd_angstrom: float | None
    docking_score: float | None
    passed: bool
    threshold_angstrom: float
    detail: str


class DockingToolResult(ToolResult):
    smiles: str
    pdb_id: str
    site: dict[str, Any]
    program: str
    best_score: float | None
    score_units: str = "kcal/mol (more negative is a better-scoring pose)"
    interpretation: str = (
        "Ranking signal only. NOT a predicted binding affinity -- correlation with "
        "measured potency across diverse chemistry is weak. Do not convert this to a "
        "Kd and do not compare it across different targets."
    )
    n_poses: int
    redocking_control: RedockControl | None
    control_verdict: str | None
    error: str | None = None


class ErrorResult(ToolResult):
    """A failure the model can act on, rather than a traceback."""

    error: str
    suggestion: str | None = None
