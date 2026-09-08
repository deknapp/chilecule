"""chilecule as an interactive service.

Everything here computes on the molecule you give it. There is no canned
output: a profile is RDKit run on your SMILES, a structural alert is a SMARTS
match against a named catalog, and a novelty check is a live ChEMBL lookup.
That matters because a frozen example proves nothing about a tool whose whole
job is to answer questions about compounds nobody chose in advance.

**What is exposed, and what is not.** The deterministic tool layer is here:
profiling, alerts, developability liabilities, synthetic accessibility, ChEMBL
lookup, and the retrospective validation harness. The agent-driven workflows
are not, because they call an LLM and would spend the deployer's money on every
visitor. That split is the same one the repository already draws between the
tool layer, which computes, and the agent layer, which decides.

**Docking** is installed in this image (smina, fpocket from conda-forge) but is
not wired into the interface. A docking run takes minutes and a shared free
container is the wrong place to start one; it stays a command-line workflow.
"""
from __future__ import annotations

import os
import sys
import traceback

import gradio as gr

# The package is installed into the image, but running this file directly from
# the repository root during development needs src/ on the path.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from chilecule.tools import alerts as alerts_mod  # noqa: E402
from chilecule.tools import chem  # noqa: E402
from chilecule.workflows import profile as profile_wf  # noqa: E402
from chilecule.workflows import validate as validate_wf  # noqa: E402

EXAMPLES = [
    ["CC(=O)Oc1ccccc1C(=O)O", "aspirin"],
    ["COc1cc2c(Nc3ccc(Br)cc3F)ncnc2cc1OCC1CCN(C)CC1", "a gefitinib-like kinase inhibitor"],
    ["CN1CCN(CC1)c1ccc(cc1)C(=O)Nc1ccc(C)c(Nc2nccc(n2)-c2cccnc2)c1", "imatinib"],
    ["CC(C)Cc1ccc(cc1)C(C)C(O)=O", "ibuprofen"],
]

NOTE = (
    "Everything on this page is computed when you press the button. Nothing is "
    "precomputed and nothing is cached between molecules."
)


def profile_compound(smiles: str, check_databases: bool) -> str:
    """Run the profile workflow and hand back its Markdown report."""
    smiles = (smiles or "").strip()
    if not smiles:
        return "Enter a SMILES string."

    if chem.parse_smiles(smiles) is None:
        return (
            f"**RDKit could not parse `{smiles}`.**\n\n"
            "That is the parser's verdict, not a guess — an unparseable "
            "structure is refused rather than approximated."
        )

    try:
        report = profile_wf.build([smiles], check_databases=check_databases)
        return report.to_markdown()
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        return (
            f"**{type(exc).__name__}**: {exc}\n\n"
            "```\n" + traceback.format_exc(limit=3) + "\n```"
        )


def screen_alerts(smiles: str) -> str:
    """Structural alerts alone, with the catalogue named."""
    smiles = (smiles or "").strip()
    if not smiles:
        return "Enter a SMILES string."
    molecule = chem.parse_smiles(smiles)
    if molecule is None:
        return f"RDKit could not parse `{smiles}`."

    try:
        # screen() returns an AlertReport whose .alerts is a tuple, not a list.
        found = list(alerts_mod.screen(smiles).alerts)
    except Exception as exc:  # noqa: BLE001
        return f"**{type(exc).__name__}**: {exc}"

    if not found:
        return (
            "**No structural alerts.**\n\n"
            "Worth reading as 'nothing matched these catalogues', not as "
            "'this compound is clean'."
        )

    lines = ["| Alert | Catalogue | Severity |", "|---|---|---|"]
    for alert in found:
        lines.append(
            f"| {alert.description} | {alert.catalog} | {alert.severity} |"
        )
    lines.append("")
    references = sorted({a.reference for a in found if getattr(a, "reference", "")})
    for reference in references:
        lines.append(f"> {reference}")
    lines.append("")
    lines.append(
        "Alerts are annotations, not verdicts. Aspirin matches a Brenk alert "
        "and roughly 5% of approved drugs match a PAINS pattern. If you exclude "
        "a compound on one, name the catalogue and say why it applies here."
    )
    return "\n".join(lines)


def validate_target(target: str) -> str:
    """Retrospective validation: does a ranking actually enrich for actives?"""
    target = (target or "").strip()
    if not target:
        return "Enter a gene symbol, for example EGFR."
    try:
        report = validate_wf.build(target, n_replicates=5, max_pool=400)
        return report.to_markdown()
    except Exception as exc:  # noqa: BLE001
        return (
            f"**{type(exc).__name__}**: {exc}\n\n"
            "This workflow pulls live bioactivity data from ChEMBL; if ChEMBL "
            "is slow or down it will fail here rather than return something "
            "made up."
        )


with gr.Blocks(title="chilecule", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# chilecule\n"
        "Agentic drug discovery on fully open-source tooling — RDKit, ChEMBL, "
        "AutoDock Vina, fpocket. This is the **deterministic tool layer**: "
        "everything below computes, and none of it asks a language model "
        "anything.\n\n"
        f"*{NOTE}*"
    )

    with gr.Tab("Profile a compound"):
        gr.Markdown(
            "Physicochemical properties, structural alerts, developability "
            "liabilities, synthetic accessibility, and — if you leave the "
            "database check on — whether ChEMBL has seen this structure before."
        )
        with gr.Row():
            smiles_in = gr.Textbox(
                label="SMILES", value=EXAMPLES[0][0], scale=4,
                placeholder="CC(=O)Oc1ccccc1C(=O)O",
            )
            db_check = gr.Checkbox(value=True, label="Check ChEMBL / PubChem", scale=1)
        gr.Examples([[e[0], True] for e in EXAMPLES], [smiles_in, db_check])
        profile_btn = gr.Button("Profile", variant="primary")
        profile_out = gr.Markdown()
        profile_btn.click(profile_compound, [smiles_in, db_check], profile_out)

    with gr.Tab("Structural alerts"):
        gr.Markdown(
            "PAINS, Brenk and friends, with the catalogue named on every hit — "
            "because an alert without its catalogue is not actionable."
        )
        alert_in = gr.Textbox(label="SMILES", value=EXAMPLES[1][0])
        alert_btn = gr.Button("Screen", variant="primary")
        alert_out = gr.Markdown()
        alert_btn.click(screen_alerts, alert_in, alert_out)

    with gr.Tab("Retrospective validation"):
        gr.Markdown(
            "The workflow that makes the rest answerable. Pulls curated actives "
            "and confirmed inactives for a target from ChEMBL, **checks the "
            "negative set for bias before using it**, then measures whether a "
            "ranking enriches for the actives — reported against a random "
            "baseline, a drug-likeness-only control, and the theoretical "
            "ceiling.\n\n"
            "The bias check is the step most published validations omit, and it "
            "is why enrichment figures so often fail to reproduce "
            "prospectively.\n\n"
            "*Live ChEMBL query — expect this one to take a minute.*"
        )
        target_in = gr.Textbox(label="Target (gene symbol)", value="EGFR")
        target_btn = gr.Button("Validate", variant="primary")
        target_out = gr.Markdown()
        target_btn.click(validate_target, target_in, target_out)

    gr.Markdown(
        "---\n"
        "**Not exposed here.** The agent-driven workflows call a language model "
        "and would spend the deployer's money on every visitor, so they stay on "
        "the command line. Docking is installed in this image but a run takes "
        "minutes and a shared container is the wrong place to start one.\n\n"
        "Source: [github.com/deknapp/chilecule](https://github.com/deknapp/chilecule)"
    )


if __name__ == "__main__":
    demo.queue(max_size=16).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        # gradio 5 renders server-side by default, which wants a Node runtime
        # that is not in this image. Without this the app starts and then fails
        # to serve a page, which looks like a networking problem rather than a
        # missing dependency.
        ssr_mode=False,
    )
