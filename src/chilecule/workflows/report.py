"""Workflow result container.

Every workflow returns one of these. The design constraint is that a result
must be readable by three different consumers without transformation: a person
reading a terminal, a person reading a Markdown file, and an agent reading
JSON. The same object serves all three.

``provenance`` is not decoration. A result that cannot say which ChEMBL release
it used, which PDB entry, which docking program version, and which filters were
applied is not reproducible, and an irreproducible result in this domain is
worth nothing.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


# Column width caps for rendered tables. SMILES strings routinely exceed 100
# characters and turn a readable table into an unreadable one; the full values
# are always present in the JSON output, which is what an agent consumes.
MAX_CELL_WIDTH = 44


def dataframe_to_markdown(df: pd.DataFrame, max_cell_width: int = MAX_CELL_WIDTH) -> str:
    """Render a DataFrame as a GitHub-flavored Markdown table.

    Implemented here rather than via ``DataFrame.to_markdown`` so that report
    generation does not depend on ``tabulate``. Every dependency in this project
    is license-audited by hand (docs/LICENSING.md), so one added for a
    fifteen-line formatting task is one too many.
    """

    def cell(value: object) -> str:
        # NaN reaches here as a float and would render as the literal "nan",
        # which reads as a value rather than as absence.
        if value is None or (isinstance(value, float) and value != value):
            return "--"
        text = str(value)
        text = text.replace("|", "\\|").replace("\n", " ")
        if len(text) > max_cell_width:
            text = text[: max_cell_width - 1] + "…"
        return text

    headers = [cell(c) for c in df.columns]
    rows = [[cell(v) for v in record] for record in df.itertuples(index=False, name=None)]

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    lines += [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join(lines)


@dataclass
class Section:
    title: str
    body: str = ""
    table: pd.DataFrame | None = None
    data: dict[str, Any] = field(default_factory=dict)
    max_rows: int = 15

    def to_markdown(self) -> str:
        parts = [f"## {self.title}", ""]
        if self.body:
            parts += [self.body.strip(), ""]
        if self.table is not None and not self.table.empty:
            shown = self.table.head(self.max_rows)
            parts += [dataframe_to_markdown(shown), ""]
            if len(self.table) > self.max_rows:
                parts += [f"*{len(self.table) - self.max_rows} further rows omitted.*", ""]
        return "\n".join(parts)

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"title": self.title, "body": self.body, **self.data}
        if self.table is not None and not self.table.empty:
            out["table"] = self.table.head(self.max_rows).to_dict(orient="records")
            out["table_row_count"] = len(self.table)
        return out


@dataclass
class Report:
    workflow: str
    subject: str
    sections: list[Section] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add(self, section: Section) -> Report:
        self.sections.append(section)
        return self

    def warn(self, message: str) -> Report:
        """Record a caveat that must travel with the result.

        Warnings are rendered before the findings, not after. A caveat placed
        below the conclusion is a caveat nobody reads.
        """
        self.warnings.append(message)
        return self

    def to_markdown(self) -> str:
        parts = [f"# {self.workflow}: {self.subject}", "", f"*Generated {self.created_at}*", ""]
        if self.warnings:
            parts += ["> **Read before interpreting these results**", ">"]
            parts += [f"> - {w}" for w in self.warnings]
            parts += [""]
        parts += [section.to_markdown() for section in self.sections]
        if self.provenance:
            parts += ["## Provenance", ""]
            parts += [f"- **{k}**: {v}" for k, v in self.provenance.items()]
            parts += [""]
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "workflow": self.workflow,
            "subject": self.subject,
            "created_at": self.created_at,
            "warnings": self.warnings,
            "sections": [s.to_dict() for s in self.sections],
            "provenance": self.provenance,
        }

    def save(self, out_dir: Path | str, stem: str | None = None) -> dict[str, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = stem or f"{self.workflow.lower().replace(' ', '_')}_{self.subject.replace('/', '_')}"

        markdown_path = out_dir / f"{stem}.md"
        json_path = out_dir / f"{stem}.json"
        markdown_path.write_text(self.to_markdown())
        json_path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return {"markdown": markdown_path, "json": json_path}


def base_provenance(**extra: Any) -> dict[str, Any]:
    """Environment facts every report should carry."""
    from .. import __version__

    return {
        "chilecule_version": __version__,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        **extra,
    }
