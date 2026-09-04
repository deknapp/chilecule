"""Agent layer: judgment on top of the deterministic tool layer.

The tools in :mod:`chilecule.tools` compute. This module decides. That split is
the whole design (see docs/ARCHITECTURE.md), and it shapes what belongs here:

* **Not here:** anything numerical. The agent never estimates a logP, a docking
  score, or an enrichment factor. Every number in a result came from a tool.
* **Here:** which of thirty fpocket cavities is the real site. Whether a filter
  that removed 90% of a library was too aggressive. Whether a docking score
  means anything given how the redocking control went. Which of three plausible
  structures to build a campaign on, and what to say about the trade-off.

The system prompt is where most of the value sits. A model asked to "analyze
this target" will produce something fluent and unfalsifiable. The prompt below
constrains it toward the behaviour a good computational chemist has: state
uncertainty, name controls, refuse to over-read a weak signal.

Requires ``pip install 'chilecule[agent]'`` and an Anthropic credential --
either an ``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile. Run
``chilecule doctor`` to see what is resolved.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

# The scientific standards the agent is held to. Written as prohibitions
# because the failure mode of an LLM on this material is not silence -- it is
# fluent overstatement, and fluent overstatement is what a domain reviewer
# notices first.
DISCOVERY_SYSTEM_PROMPT = """\
You are a computational chemist working with the chilecule toolkit. You have
tools for cheminformatics, ChEMBL bioactivity data, protein structures, pocket
detection, and molecular docking.

HOW TO WORK

Compute nothing yourself. Every number you report must come from a tool call.
If you find yourself about to state a logP, a potency, a docking score, or an
enrichment factor that no tool returned, stop and call the tool instead.

Standardize structures before comparing or deduplicating them. A salt and its
parent acid are one compound, and treating them as two corrupts every count
downstream.

Prefer a ligand-defined binding site over a predicted one. If a structure has a
co-crystallized ligand, the site is where that ligand is. Run pocket detection
only when there is nothing bound.

HOW TO REPORT

A docking score is not a predicted binding affinity. Its correlation with
measured affinity across diverse chemistry is weak. Describe scores as a
ranking signal. Never convert one to a Kd, and never compare scores across
different targets.

Never report enrichment without the ceiling it is measured against, and never
report a model metric without saying how the data was split. A random split on
a dataset dominated by one scaffold measures memorization.

Structural alerts are annotations, not verdicts. Roughly 5% of approved drugs
match a PAINS pattern and aspirin matches a Brenk one. If you exclude a
compound on an alert, name the catalog and say why the exclusion is justified
here.

If a redocking control failed, say so before presenting any docking result from
that site, and say that the results should not be used to rank compounds.

State what would change your conclusion. When the evidence is weak, say it is
weak rather than hedging with adverbs. "The data does not distinguish these two
series" is a finding; "the compounds show promising activity" is not.

Distinguish what the data shows from what you infer from it. Cite the tool
output for the former.
"""

# Every tool the agent is permitted to call, named as the Agent SDK addresses
# MCP tools: mcp__<server-name>__<tool-name>.
CHILECULE_TOOLS = [
    "mcp__chilecule__standardize_molecule",
    "mcp__chilecule__molecule_properties",
    "mcp__chilecule__structural_alerts",
    "mcp__chilecule__ligand_efficiency_metrics",
    "mcp__chilecule__compare_molecules",
    "mcp__chilecule__find_chembl_target",
    "mcp__chilecule__get_target_actives",
    "mcp__chilecule__find_structures",
    "mcp__chilecule__inspect_structure",
    "mcp__chilecule__detect_pockets",
    "mcp__chilecule__dock_molecule",
    "mcp__chilecule__check_compound_novelty",
    "mcp__chilecule__check_promiscuity",
    "mcp__chilecule__assess_synthesis",
    "mcp__chilecule__score_compound",
    "mcp__chilecule__profile_compound",
    "mcp__chilecule__design_analogs",
    "mcp__chilecule__check_liabilities",
]


def mcp_server_config() -> dict:
    """Configuration attaching this project's MCP server to an agent.

    Launched as a subprocess over stdio using the *current* interpreter, so the
    agent gets the environment chilecule is installed in rather than whatever
    happens to be first on PATH.
    """
    return {
        "chilecule": {
            "command": sys.executable,
            "args": ["-m", "chilecule.mcp.server"],
        }
    }


def build_options(
    *,
    model: str = "claude-opus-5",
    max_turns: int = 40,
    extra_instructions: str = "",
):
    """Assemble ``ClaudeAgentOptions`` for a discovery agent."""
    try:
        from claude_agent_sdk import ClaudeAgentOptions
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "The agent extra is not installed. Install it with: "
            "pip install 'chilecule[agent]'"
        ) from exc

    system_prompt = DISCOVERY_SYSTEM_PROMPT
    if extra_instructions:
        system_prompt = f"{system_prompt}\n\nADDITIONAL INSTRUCTIONS\n\n{extra_instructions}"

    return ClaudeAgentOptions(
        system_prompt=system_prompt,
        mcp_servers=mcp_server_config(),
        allowed_tools=CHILECULE_TOOLS,
        model=model,
        max_turns=max_turns,
    )


@dataclass
class AgentRun:
    """What an agent did, not just what it said.

    ``tools_called`` is the part that makes behaviour testable. Whether an
    answer *sounds* careful is a matter of opinion; whether the agent actually
    called the novelty tool before asserting a compound is known is a fact.
    """

    answer: str
    tools_called: list[str] = field(default_factory=list)
    n_turns: int = 0

    def called(self, tool: str) -> bool:
        return any(name.endswith(tool) for name in self.tools_called)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "tools_called": self.tools_called,
            "unique_tools": sorted(set(self.tools_called)),
            "n_turns": self.n_turns,
        }


async def run_detailed(
    task: str,
    *,
    model: str = "claude-opus-5",
    max_turns: int = 40,
    extra_instructions: str = "",
) -> AgentRun:
    """Run a task and return both the answer and the tool calls made."""
    from claude_agent_sdk import query

    options = build_options(
        model=model, max_turns=max_turns, extra_instructions=extra_instructions
    )

    chunks: list[str] = []
    tools_called: list[str] = []
    turns = 0
    async for message in query(prompt=task, options=options):
        turns += 1
        for block in getattr(message, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                chunks.append(text)
            name = getattr(block, "name", None)
            if name and type(block).__name__.startswith("ToolUse"):
                tools_called.append(name)

    return AgentRun(answer="\n".join(chunks), tools_called=tools_called, n_turns=turns)


async def run(
    task: str,
    *,
    model: str = "claude-opus-5",
    max_turns: int = 40,
    extra_instructions: str = "",
    output_path: Path | str | None = None,
) -> str:
    """Run a discovery task and return the agent's final text.

    Example::

        import asyncio
        from chilecule.agents.discovery import run

        answer = asyncio.run(run(
            "I have a hit against EGFR. What should I make next, and how much "
            "should I trust each suggestion?"
        ))
    """
    result = await run_detailed(
        task, model=model, max_turns=max_turns, extra_instructions=extra_instructions
    )
    if output_path:
        Path(output_path).write_text(result.answer)
    return result.answer
