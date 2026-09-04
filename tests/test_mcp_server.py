"""Tests for the MCP server, driven over real JSON-RPC stdio.

Calling the handlers in-process would test that the functions work; it would not
test that a client can reach them. The protocol layer -- transport,
initialization handshake, schema generation -- is where an MCP integration
actually breaks, so these tests speak the wire protocol.
"""

import json
import subprocess
import sys

import pytest

pytest.importorskip("mcp", reason="MCP extra not installed")


class MCPProbe:
    """A minimal MCP client: launch the server, complete the handshake, call tools."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "chilecule.mcp.server"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self._id = 0

    def _send(self, payload: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        self._send({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}})
        assert self.proc.stdout is not None
        return json.loads(self.proc.stdout.readline())

    def notify(self, method: str, params: dict | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def initialize(self) -> dict:
        response = self.request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "chilecule-tests", "version": "0"},
        })
        self.notify("notifications/initialized")
        return response

    def close(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover
            self.proc.kill()


@pytest.fixture(scope="module")
def probe():
    client = MCPProbe()
    client.initialize()
    yield client
    client.close()


def test_server_reports_name_and_version(probe):
    """A blank version string is what a client shows the user."""
    from chilecule import __version__

    info = MCPProbe()
    try:
        response = info.initialize()
        server = response["result"]["serverInfo"]
        assert server["name"] == "chilecule"
        assert server["version"] == __version__
    finally:
        info.close()


def test_tools_are_discoverable(probe):
    tools = probe.request("tools/list")["result"]["tools"]
    names = {tool["name"] for tool in tools}
    assert {
        "standardize_molecule", "molecule_properties", "structural_alerts",
        "find_chembl_target", "dock_molecule", "detect_pockets",
    } <= names


def test_every_tool_has_a_description_and_schema(probe):
    """A tool with no description is a tool the model will misuse."""
    for tool in probe.request("tools/list")["result"]["tools"]:
        assert tool.get("description"), f"{tool['name']} has no description"
        assert len(tool["description"]) > 80, f"{tool['name']} description is too thin"
        assert tool.get("inputSchema", {}).get("properties")


def test_every_tool_publishes_an_output_schema(probe):
    """An output schema lets a client validate and destructure the response
    instead of parsing whatever text came back."""
    for tool in probe.request("tools/list")["result"]["tools"]:
        schema = tool.get("outputSchema")
        assert schema, f"{tool['name']} publishes no output schema"
        assert schema.get("properties")


def test_argument_schemas_carry_descriptions_and_bounds(probe):
    """Argument schemas are prompt surface.

    A bare {"type": "integer"} tells the model nothing about what a sane value
    is, so it guesses -- and the guess costs a docking run.
    """
    tools = {t["name"]: t for t in probe.request("tools/list")["result"]["tools"]}

    smiles = tools["molecule_properties"]["inputSchema"]["properties"]["smiles"]
    assert "description" in smiles
    assert smiles.get("examples"), "an example SMILES saves the model a failed call"

    effort = tools["dock_molecule"]["inputSchema"]["properties"]["exhaustiveness"]
    assert effort["minimum"] == 1
    assert effort["maximum"] == 64
    assert "description" in effort

    pdb = tools["inspect_structure"]["inputSchema"]["properties"]["pdb_id"]
    assert pdb.get("pattern"), "an unconstrained pdb_id accepts prose"


def test_docking_tool_states_the_score_caveat(probe):
    """The caveat has to be where the model reads it, not only in our docs."""
    tools = {t["name"]: t for t in probe.request("tools/list")["result"]["tools"]}
    description = tools["dock_molecule"]["description"].lower()
    assert "not a predicted binding affinity" in description or "not" in description
    assert "affinity" in description


def test_tool_call_returns_structured_json(probe):
    response = probe.request("tools/call", {
        "name": "molecule_properties",
        "arguments": {"smiles": "CC(=O)Oc1ccccc1C(=O)O"},
    })
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["mw"] == pytest.approx(180.16, abs=0.1)
    assert payload["heavy_atoms"] == 13


def test_invalid_input_is_rejected_at_the_boundary(probe):
    """An unparseable SMILES is refused before any work happens, with a message
    the model can correct from -- and the server survives."""
    response = probe.request("tools/call", {
        "name": "molecule_properties",
        "arguments": {"smiles": "definitely not a molecule"},
    })
    assert response["result"]["isError"] is True
    message = response["result"]["content"][0]["text"]
    assert "not a valid SMILES" in message
    # The message must name the likely mistake, not just say "invalid".
    assert "compound name" in message
    # The server is still alive afterwards.
    assert probe.request("tools/list")["result"]["tools"]


def test_out_of_range_effort_is_refused_without_running_anything(probe):
    """REGRESSION GUARD: exhaustiveness is unbounded in a naive schema.

    A model passing 10000 would previously have launched a docking run that
    does not finish in a day. Validation turns that into a millisecond error.
    """
    import time

    start = time.monotonic()
    response = probe.request("tools/call", {
        "name": "dock_molecule",
        "arguments": {"smiles": "CCO", "pdb_id": "5CNN", "exhaustiveness": 10000},
    })
    elapsed = time.monotonic() - start

    assert response["result"]["isError"] is True
    assert "less than or equal to 64" in response["result"]["content"][0]["text"]
    # No PDB download, no subprocess. If this ever takes seconds, the check moved.
    assert elapsed < 2.0, f"validation should be instant, took {elapsed:.1f}s"


def test_prose_where_an_identifier_belongs_is_refused(probe):
    response = probe.request("tools/call", {
        "name": "inspect_structure",
        "arguments": {"pdb_id": "the EGFR structure"},
    })
    assert response["result"]["isError"] is True
    assert "pattern" in response["result"]["content"][0]["text"]


def test_nonsensical_potency_is_refused(probe):
    """A negative IC50 is not a measurement, and -log10 of it is not a number."""
    response = probe.request("tools/call", {
        "name": "ligand_efficiency_metrics",
        "arguments": {"smiles": "CCO", "potency_nm": -5},
    })
    assert response["result"]["isError"] is True
    assert "greater than 0" in response["result"]["content"][0]["text"]


def test_valid_calls_return_structured_content(probe):
    """With an output schema declared, responses carry structured content that a
    client can use without re-parsing text."""
    response = probe.request("tools/call", {
        "name": "ligand_efficiency_metrics",
        "arguments": {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "potency_nm": 50.0},
    })
    assert not response["result"].get("isError")
    structured = response["result"].get("structuredContent")
    assert structured is not None
    assert structured["heavy_atoms"] == 13
    assert structured["ligand_efficiency"] == pytest.approx(0.771, abs=0.01)


def test_standardization_tool_round_trips_a_salt(probe):
    response = probe.request("tools/call", {
        "name": "standardize_molecule",
        "arguments": {"smiles": "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]"},
    })
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["standardized_smiles"] == "CC(=O)Oc1ccccc1C(=O)O"
    assert payload["inchikey"] == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"


def test_agent_tool_allowlist_matches_the_server(probe):
    """The agent's allowlist and the server's tool set must not drift apart --
    a name in the allowlist that the server does not expose is a tool the agent
    silently cannot call."""
    from chilecule.agents.discovery import CHILECULE_TOOLS

    exposed = {t["name"] for t in probe.request("tools/list")["result"]["tools"]}
    allowlisted = {name.rsplit("__", 1)[-1] for name in CHILECULE_TOOLS}
    assert allowlisted == exposed
