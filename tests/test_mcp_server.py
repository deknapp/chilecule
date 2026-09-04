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


def test_tool_errors_are_returned_as_data_not_crashes(probe):
    """An unparseable SMILES must not take the server down."""
    response = probe.request("tools/call", {
        "name": "molecule_properties",
        "arguments": {"smiles": "definitely not a molecule"},
    })
    payload = json.loads(response["result"]["content"][0]["text"])
    assert "error" in payload
    # The server is still alive afterwards.
    assert probe.request("tools/list")["result"]["tools"]


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
