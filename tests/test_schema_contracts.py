"""Contract tests between the tool layer and the MCP schemas.

Every MCP tool converts a dataclass from the tool layer into a pydantic model.
Those models set ``extra="forbid"``, so adding a field to a dataclass's
``to_dict`` without adding it to the model turns every call to that tool into a
validation error.

That is exactly what happened. Adding the attachment context to
``Transformation.to_dict()`` broke ``design_analogs`` completely -- and no test
caught it, because the tool-layer tests never touch the MCP wrapper and the MCP
tests never call the tools that need network access. An agent found it, in
production, by trying to use the tool.

These tests close that gap without needing the network: they assert that each
model accepts exactly what its dataclass produces.
"""

import pandas as pd
import pytest

from chilecule.mcp import models
from chilecule.tools.alerts import screen
from chilecule.tools.chem import properties
from chilecule.tools.design import Transformation
from chilecule.tools.pockets import Pocket
from chilecule.tools.structure import Ligand, StructureHit, rank_structures
from chilecule.tools.synth import assess

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def test_properties_dataclass_matches_its_model():
    models.PropertiesResult(**properties(ASPIRIN).to_dict())


def test_synthesis_dataclass_matches_its_model():
    payload = assess(ASPIRIN).to_dict()
    payload.pop("caveat", None)
    models.SynthesisResult(**payload)


def test_transformation_dataclass_matches_its_model():
    """REGRESSION: adding `attached_to` to to_dict() broke design_analogs entirely."""
    transformation = Transformation(
        lhs="C[*:1]", rhs="F[*:1]", context="c",
        n_pairs=5, median_delta=0.4, std_delta=0.2, min_delta=0.0, max_delta=1.0,
    )
    model = models.TransformationModel(**transformation.to_dict())
    assert model.attached_to == "aromatic C"


def test_alert_dataclass_matches_its_model():
    report = screen("O=C1CSC(=S)N1")
    for alert in report.to_dict()["alerts"]:
        models.AlertMatch(**alert)


def test_ligand_dataclass_matches_its_model():
    ligand = Ligand("ANP", "A", 1101, 31, (1.0, 2.0, 3.0), "ligand")
    models.LigandModel(**ligand.to_dict())


def test_pocket_dataclass_matches_its_model():
    pocket = Pocket(
        rank=1, score=0.1, druggability=0.5, volume=2000.0, volume_score=4.0,
        n_alpha_spheres=100, hydrophobicity=20.0, center=(1.0, 2.0, 3.0),
    )
    models.PocketModel(**pocket.to_dict())


def test_structure_hit_dataclass_matches_its_model():
    ranked = rank_structures([StructureHit("5CNN", "A", 1.9, 0.3, "X-ray diffraction")])
    models.StructureHitModel(**ranked[0].to_dict())


@pytest.mark.parametrize(
    "model",
    [
        models.PropertiesResult, models.SynthesisResult, models.TransformationModel,
        models.AlertMatch, models.LigandModel, models.PocketModel,
        models.StructureHitModel, models.NoveltyResult, models.PromiscuityResult,
    ],
)
def test_models_forbid_extra_fields(model):
    """The guard that caught the drift must stay on.

    Allowing extras would have let design_analogs keep working while silently
    dropping the attachment context from what the agent was told -- a worse
    failure than the loud one, because the agent would have gone on proposing
    analogs with no way to know the context existed.
    """
    assert model.model_config.get("extra") == "forbid"


def test_every_mcp_tool_is_registered_and_described():
    """A cheap guard that the module imports cleanly and nothing silently vanished."""
    import asyncio

    from chilecule.mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    assert len(tools) >= 17
    assert all(t.description and t.output_schema for t in tools)


def test_dataframe_serialisation_survives_json():
    """Report tables are handed to agents as JSON; NaN and numpy types must survive."""
    import json

    from chilecule.workflows.report import Report, Section

    report = Report(workflow="T", subject="S")
    report.add(Section(title="S", table=pd.DataFrame({"a": [1.0, float("nan")]})))
    json.loads(json.dumps(report.to_dict(), default=str))
