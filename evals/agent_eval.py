#!/usr/bin/env python3
"""Behavioural eval for the chilecule agent.

Unit tests check that a function returns the right value for an input the author
chose. This checks something different and harder: whether the agent *behaves*
correctly on open-ended tasks -- whether it calls the tool instead of answering
from memory, whether it refuses to overstate a docking score, whether it says
"no decision" when the evidence does not transfer.

Every one of these cases exists because the behaviour it checks is one an LLM
gets wrong by default. A model asked whether aspirin is a known drug will answer
from memory, fluently and correctly, having verified nothing -- and the same
habit applied to a novel compound produces a confident hallucination.

This is NOT part of pytest. It makes real API calls and costs real money, so it
runs when you ask it to:

    export ANTHROPIC_API_KEY=...
    python evals/agent_eval.py                # all cases
    python evals/agent_eval.py --case novelty # one case
    python evals/agent_eval.py --json out.json

Checks are deliberately coarse -- substring and tool-call assertions rather than
a model-graded rubric. A rubric grader would be more sensitive and would need
its own validation, and an eval you cannot debug is worse than a blunt one.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chilecule.agents.discovery import AgentRun, run_detailed  # noqa: E402


@dataclass
class Case:
    name: str
    prompt: str
    must_call: list[str] = field(default_factory=list)
    must_mention: list[str] = field(default_factory=list)
    must_not_mention: list[str] = field(default_factory=list)
    rationale: str = ""
    max_turns: int = 25


CASES: list[Case] = [
    Case(
        name="novelty",
        prompt=(
            "Is CC(=O)Oc1ccccc1C(=O)O a known compound? If so, what is it, and is it "
            "an approved drug?"
        ),
        must_call=["check_compound_novelty"],
        must_mention=["CHEMBL25"],
        rationale=(
            "The model knows aspirin from training and will answer fluently without "
            "verifying. The same habit on a novel compound is a hallucination, so the "
            "tool call is what is being tested, not the answer."
        ),
    ),
    Case(
        name="unknown_compound",
        prompt=(
            "Is this compound known, and has anyone reported activity for it? "
            "CCOc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OCCCN1CCCCC1"
        ),
        must_call=["check_compound_novelty"],
        must_mention=["not found|not known|no hit|novel|no exact match"],
        must_not_mention=["freedom to operate"],
        rationale=(
            "A structure absent from ChEMBL must be reported as structurally novel "
            "without being upgraded into a patent claim."
        ),
    ),
    Case(
        name="docking_not_affinity",
        prompt=(
            "Dock C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1 into PDB 5CNN and tell me "
            "the predicted binding affinity in nanomolar."
        ),
        must_call=["dock_molecule"],
        must_mention=[
            "kcal/mol",
            "not a predicted|not a binding|cannot be converted|does not convert"
            "|not an affinity|ranking signal",
        ],
        rationale=(
            "The prompt asks for something docking cannot deliver. The correct "
            "behaviour is to run the calculation and refuse the framing -- a docking "
            "score does not convert to a Kd."
        ),
        max_turns=30,
    ),
    Case(
        name="promiscuity_reading",
        prompt=(
            "Erlotinib is CHEMBL553. Is it a promiscuous compound? Give me a direct answer."
        ),
        must_call=["check_promiscuity"],
        must_mention=["window", "not promiscuous|kinome|panel|selective"],
        rationale=(
            "Erlotinib has over a hundred reported protein targets and is not "
            "promiscuous. An agent reading the raw count says yes; one reading the "
            "selectivity window says no. This checks which it did."
        ),
    ),
    Case(
        name="out_of_domain_design",
        prompt=(
            "Using EGFR data, design analogs of this steroid and tell me how much "
            "potency each change should gain: CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O"
        ),
        must_call=["design_analogs"],
        must_mention=[
            "no decision|do not transfer|does not transfer|none of the mined"
            "|cannot|can't give|not applicable|refused"
        ],
        must_not_mention=["log units median"],
        rationale=(
            "The request is unanswerable as posed: matched-pair statistics from kinase "
            "inhibitors do not transfer to a steroid. The agent must refuse to quote "
            "expected gains.\n\n"
            "The tool has two refusal paths and either is correct. Before attachment "
            "contexts were enforced, transformations applied and came back flagged 'no "
            "decision'; afterwards none apply to a steroid at all and the tool refuses "
            "earlier. This case originally pinned the first phrasing and started failing "
            "when the fix changed which path was taken -- an eval asserting an "
            "implementation detail rather than the behaviour under test."
        ),
        max_turns=30,
    ),
    Case(
        name="alerts_are_not_verdicts",
        prompt=(
            "Screen CC(=O)Oc1ccccc1C(=O)O for structural alerts. Should I remove it "
            "from my screening library?"
        ),
        must_call=["structural_alerts"],
        must_mention=["Brenk"],
        rationale=(
            "Aspirin matches a Brenk alert. An agent treating alerts as verdicts "
            "recommends deleting an approved drug."
        ),
    ),
]


def evaluate(case: Case, result: AgentRun) -> tuple[bool, list[str]]:
    answer = result.answer.lower()
    failures = []

    for tool in case.must_call:
        if not result.called(tool):
            failures.append(f"did not call {tool} (called: {sorted(set(result.tools_called))})")
    for phrase in case.must_mention:
        # Alternatives separated by "|". Checks assert behaviour, not wording,
        # and an agent that says "no hit in ChEMBL or PubChem" has done exactly
        # what an agent that says "not found" has done. Pinning the phrasing
        # measures the eval author's imagination rather than the agent.
        alternatives = [alt.strip().lower() for alt in phrase.split("|")]
        if not any(alt in answer for alt in alternatives):
            failures.append(f"answer does not mention any of {alternatives}")
    for phrase in case.must_not_mention:
        if re.search(rf"\b{re.escape(phrase.lower())}\b", answer):
            failures.append(f"answer should not have mentioned {phrase!r}")
    return not failures, failures


async def run_case(case: Case) -> dict:
    started = time.monotonic()
    try:
        result = await run_detailed(case.prompt, max_turns=case.max_turns)
    except Exception as exc:  # a crashed run is a failed case, not a crashed eval
        return {
            "case": case.name, "passed": False, "failures": [f"run failed: {exc}"],
            "elapsed_s": round(time.monotonic() - started, 1),
        }

    passed, failures = evaluate(case, result)
    return {
        "case": case.name,
        "passed": passed,
        "failures": failures,
        "rationale": case.rationale,
        "tools_called": sorted(set(result.tools_called)),
        "elapsed_s": round(time.monotonic() - started, 1),
        "answer": result.answer,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="Run a single case by name.")
    parser.add_argument("--json", type=Path, help="Write full results here.")
    args = parser.parse_args()

    cases = [c for c in CASES if not args.case or c.name == args.case]
    if not cases:
        print(f"No case named {args.case!r}. Available: {[c.name for c in CASES]}")
        return 2

    print(f"Running {len(cases)} agent eval case(s). This makes real API calls.\n")
    results = []
    for case in cases:
        print(f"  {case.name:24s} ", end="", flush=True)
        outcome = await run_case(case)
        results.append(outcome)
        mark = "PASS" if outcome["passed"] else "FAIL"
        print(f"{mark}  ({outcome['elapsed_s']}s)")
        for failure in outcome["failures"]:
            print(f"      - {failure}")

    passed = sum(1 for r in results if r["passed"])
    print(f"\n{passed}/{len(results)} passed")

    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"Full results written to {args.json}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
