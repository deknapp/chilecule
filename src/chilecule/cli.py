"""Command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="chilecule",
    help="Agentic drug discovery workflows on fully open-source tooling.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()

DEFAULT_OUTPUT = Path("runs")


def _emit(report, out_dir: Path, save: bool) -> None:
    console.print(report.to_markdown())
    if save:
        paths = report.save(out_dir)
        console.print(
            f"\n[green]Saved[/green] {paths['markdown']} and {paths['json']}"
        )


@app.command()
def doctor() -> None:
    """Report what is installed, what credentials are present, and which tiers can run."""
    from .config import detect

    env = detect()
    console.print(Panel.fit(f"chilecule environment  ·  Python {env.python_version}"))

    table = Table("Component", "Status", "Detail", box=None, pad_edge=False)
    for capability in (env.anthropic, env.aws):
        mark = "[green]yes[/green]" if capability.available else "[yellow]no[/yellow]"
        table.add_row(capability.name, mark, capability.detail)
    for name, capability in env.packages.items():
        mark = "[green]yes[/green]" if capability.available else "[red]no[/red]"
        table.add_row(f"py:{name}", mark, capability.detail)
    for name, capability in env.binaries.items():
        mark = "[green]yes[/green]" if capability.available else "[yellow]no[/yellow]"
        table.add_row(f"bin:{name}", mark, capability.detail)
    console.print(table)

    console.print("\n[bold]Workflow tiers[/bold]")
    descriptions = {
        "data-and-sar": "dossier, sar, validate  (needs RDKit only)",
        "structure-based": "triage with docking, pockets  (needs a docking program)",
        "agent-driven": "agent-run workflows  (needs an Anthropic credential)",
        "cloud-burst": "GPU / large-scale  (scaffold only -- not implemented)",
    }
    for tier, ready in env.tiers.items():
        # Pad the plain word, then wrap it in markup. Padding the marked-up
        # string counts the tag characters and misaligns every column.
        word = "ready" if ready else "unavailable"
        colour = "green" if ready else "yellow"
        console.print(
            f"  {tier:<18} [{colour}]{word}[/{colour}]{' ' * (13 - len(word))} "
            f"{descriptions[tier]}"
        )

    if not env.tiers["structure-based"]:
        console.print(
            "\n[dim]To enable docking:  micromamba install -c conda-forge smina fpocket[/dim]"
        )


@app.command()
def dossier(
    target: str = typer.Argument(..., help="Gene symbol or UniProt accession, e.g. EGFR"),
    max_records: int = typer.Option(5000, help="Cap on ChEMBL activity records fetched."),
    out: Path = typer.Option(DEFAULT_OUTPUT, help="Directory for saved reports."),
    save: bool = typer.Option(True, help="Write Markdown and JSON alongside the terminal output."),
) -> None:
    """Assemble what is publicly known about a target."""
    from .workflows import dossier as workflow

    with console.status(f"Building dossier for {target}..."):
        report = workflow.build(target, max_activity_records=max_records)
    _emit(report, out, save)


@app.command()
def sar(
    target: str = typer.Argument(..., help="Gene symbol or UniProt accession."),
    max_records: int = typer.Option(5000, help="Cap on ChEMBL activity records fetched."),
    max_compounds: int = typer.Option(800, help="Compounds included in matched-pair analysis."),
    out: Path = typer.Option(DEFAULT_OUTPUT),
    save: bool = typer.Option(True),
) -> None:
    """Analyse scaffolds, matched molecular pairs, and activity cliffs."""
    from .workflows import sar as workflow

    with console.status(f"Analysing SAR for {target}..."):
        report = workflow.build(
            target, max_activity_records=max_records, max_compounds_for_pairs=max_compounds
        )
    _emit(report, out, save)


@app.command()
def triage(
    smiles_file: Path = typer.Argument(..., help="File with one SMILES per line, or a CSV with a 'smiles' column."),
    pdb: Optional[str] = typer.Option(None, help="PDB ID defining the binding site, e.g. 5CNN."),
    dock_top: int = typer.Option(25, help="How many survivors to dock."),
    exhaustiveness: int = typer.Option(8, help="Docking search effort."),
    out: Path = typer.Option(DEFAULT_OUTPUT),
    save: bool = typer.Option(True),
) -> None:
    """Filter a library down to a shortlist, docking the survivors if a site is given."""
    import pandas as pd

    from .workflows import triage as workflow

    if smiles_file.suffix.lower() == ".csv":
        frame = pd.read_csv(smiles_file)
        if "smiles" not in frame.columns:
            raise typer.BadParameter(f"{smiles_file} has no 'smiles' column")
        library = frame["smiles"].dropna().tolist()
    else:
        library = [line.strip() for line in smiles_file.read_text().splitlines() if line.strip()]

    console.print(f"Read {len(library)} compounds from {smiles_file}")
    with console.status("Running triage cascade..."):
        report = workflow.build(
            library, pdb_id=pdb, dock_top_n=dock_top, exhaustiveness=exhaustiveness
        )
    _emit(report, out, save)


@app.command()
def validate(
    target: str = typer.Argument(..., help="Gene symbol or UniProt accession."),
    max_records: int = typer.Option(8000),
    active_threshold: float = typer.Option(7.0, help="pChEMBL cutoff defining an active."),
    replicates: int = typer.Option(20, help="Subsampling replicates for the metrics."),
    out: Path = typer.Option(DEFAULT_OUTPUT),
    save: bool = typer.Option(True),
) -> None:
    """Measure whether a ranking method enriches for known actives on this target."""
    from .workflows import validate as workflow

    with console.status(f"Validating on {target}..."):
        report = workflow.build(
            target,
            max_activity_records=max_records,
            active_threshold=active_threshold,
            n_replicates=replicates,
        )
    _emit(report, out, save)


@app.command()
def pockets(
    pdb: str = typer.Argument(..., help="PDB ID, e.g. 5CNN."),
    chain: Optional[str] = typer.Option(None, help="Restrict to one chain."),
    top: int = typer.Option(5, help="How many pockets to report."),
) -> None:
    """Detect and rank candidate binding sites in a structure."""
    from .tools.pockets import FpocketUnavailable, find_pockets
    from .tools.structure import fetch_pdb, prepare_receptor

    structure = fetch_pdb(pdb)
    receptor = prepare_receptor(structure, keep_chain=chain)
    try:
        found = find_pockets(receptor, max_pockets=top)
    except FpocketUnavailable as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    table = Table("Rank", "Druggability", "Volume A^3", "Center", "Assessment")
    for pocket in found:
        table.add_row(
            str(pocket.rank),
            f"{pocket.druggability:.2f}",
            f"{pocket.volume:.0f}",
            ", ".join(f"{c:.1f}" for c in pocket.center),
            pocket.assessment,
        )
    console.print(table)


@app.command()
def serve_mcp(
    transport: str = typer.Option("stdio", help="MCP transport (stdio is what clients expect)."),
) -> None:
    """Expose the tool layer as an MCP server for Claude Code, Claude Desktop, or any MCP client."""
    try:
        from .mcp.server import run
    except ImportError as exc:
        console.print(
            "[red]The MCP extra is not installed.[/red]  pip install 'chilecule[mcp]'"
        )
        raise typer.Exit(1) from exc
    run(transport=transport)


if __name__ == "__main__":
    app()
