"""Binding-site detection with fpocket.

Needed when there is no co-crystallized ligand to define a site -- an apo
structure, a predicted model, or a target being assessed for a site nobody has
drugged yet.

A caution that shapes how results are reported: fpocket finds *cavities*, and
most cavities are not druggable. A typical protein returns 15-40 pockets, of
which perhaps one or two can bind a small molecule with useful affinity. The
druggability score is a useful triage signal and a poor oracle, so pockets are
returned ranked with their scores exposed rather than reduced to a single
"the binding site" answer.

When a ligand-bound structure is available, use it. A pocket defined by
chemical matter the protein demonstrably binds beats any prediction.

fpocket is MIT licensed and is invoked here as an external process.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .structure import DockingBox

log = logging.getLogger(__name__)


class FpocketUnavailable(RuntimeError):
    """Raised when fpocket is not on PATH."""


@dataclass
class Pocket:
    rank: int
    score: float
    druggability: float
    volume: float
    volume_score: float
    n_alpha_spheres: int
    hydrophobicity: float
    center: tuple[float, float, float]

    @property
    def assessment(self) -> str:
        """Plain-language read on whether this pocket is worth pursuing.

        Thresholds follow the fpocket authors' own calibration against known
        druggable sites: above 0.5 the pocket resembles sites that bind
        drug-like molecules, below 0.2 it generally does not.
        """
        if self.druggability >= 0.5:
            return "druggable -- comparable to sites known to bind small molecules"
        if self.druggability >= 0.2:
            return "borderline -- may bind fragments; verify with a known ligand"
        return "unlikely to bind a drug-like molecule"

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "score": round(self.score, 3),
            "druggability": round(self.druggability, 3),
            "volume_A3": round(self.volume, 1),
            "volume_score": round(self.volume_score, 2),
            "n_alpha_spheres": self.n_alpha_spheres,
            "hydrophobicity": round(self.hydrophobicity, 2),
            "center": [round(c, 2) for c in self.center],
            "assessment": self.assessment,
        }

    def to_box(self, padding: float = 4.0, min_size: float = 18.0) -> DockingBox:
        """Docking box centred on this pocket.

        Sized from pocket volume assuming a roughly cubic cavity, with a floor:
        a box smaller than about 18 A cannot accommodate a typical drug-like
        ligand in any orientation, so a tiny pocket still gets a usable box.
        """
        edge = max(self.volume ** (1 / 3) + 2 * padding, min_size)
        return DockingBox(
            center=self.center,
            size=(edge, edge, edge),
            source=f"fpocket rank {self.rank} (druggability {self.druggability:.2f})",
        )


def available() -> bool:
    return shutil.which("fpocket") is not None


def find_pockets(
    pdb_path: Path | str,
    *,
    min_druggability: float = 0.0,
    max_pockets: int = 10,
    timeout: int = 300,
) -> list[Pocket]:
    """Run fpocket and return pockets ranked by its own scoring."""
    if not available():
        raise FpocketUnavailable(
            "fpocket is not on PATH. Install it with:\n"
            "    micromamba install -c conda-forge fpocket\n"
            "or run ./install.sh --tier dock"
        )

    pdb_path = Path(pdb_path)
    with tempfile.TemporaryDirectory() as tmp:
        # fpocket writes its output beside the input file, so copy the input
        # into a scratch directory rather than littering the user's cache.
        work = Path(tmp) / pdb_path.name
        work.write_bytes(pdb_path.read_bytes())

        completed = subprocess.run(
            ["fpocket", "-f", str(work)],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        info_file = Path(tmp) / f"{work.stem}_out" / f"{work.stem}_info.txt"
        if completed.returncode != 0 or not info_file.exists():
            log.warning("fpocket failed: %s", (completed.stderr or completed.stdout)[:300])
            return []

        pockets = _parse_info(info_file.read_text())
        centers = _pocket_centers(Path(tmp) / f"{work.stem}_out" / "pockets")

    resolved = [
        Pocket(**{**p, "center": centers.get(p["rank"], (0.0, 0.0, 0.0))})
        for p in pockets
        if p["druggability"] >= min_druggability
    ]
    return sorted(resolved, key=lambda p: -p.druggability)[:max_pockets]


_FIELD_PATTERN = re.compile(r"^\s*([A-Za-z][^:]*?)\s*:\s*([-\d.eE+]+)\s*$")


def _parse_info(text: str) -> list[dict]:
    """Parse fpocket's ``*_info.txt`` block format."""
    pockets: list[dict] = []
    current: dict | None = None

    for line in text.splitlines():
        header = re.match(r"^Pocket\s+(\d+)\s*:", line)
        if header:
            if current:
                pockets.append(current)
            current = {
                "rank": int(header.group(1)), "score": 0.0, "druggability": 0.0,
                "volume": 0.0, "volume_score": 0.0, "n_alpha_spheres": 0,
                "hydrophobicity": 0.0,
            }
            continue
        if current is None:
            continue
        field = _FIELD_PATTERN.match(line)
        if not field:
            continue
        key, raw = field.group(1).strip().lower(), field.group(2)
        try:
            value = float(raw)
        except ValueError:
            continue
        if key == "score":
            current["score"] = value
        elif "druggability" in key:
            current["druggability"] = value
        elif key == "volume":
            # Exact match required. fpocket emits BOTH "Volume : 321.291"
            # (the real cavity volume in A^3) and "Volume score: 4.364" (a
            # normalized 0-10 descriptor). A prefix match reads the second over
            # the first and reports every pocket as a few cubic angstroms,
            # which is smaller than a water molecule.
            current["volume"] = value
        elif key == "volume score":
            current["volume_score"] = value
        elif "alpha sphere" in key and "number" in key:
            current["n_alpha_spheres"] = int(value)
        elif "hydrophobicity" in key:
            current["hydrophobicity"] = value

    if current:
        pockets.append(current)
    return pockets


def _pocket_centers(pockets_dir: Path) -> dict[int, tuple[float, float, float]]:
    """Centroid of each pocket's alpha spheres, from the per-pocket PDB files."""
    centers: dict[int, tuple[float, float, float]] = {}
    if not pockets_dir.exists():
        return centers

    for path in pockets_dir.glob("pocket*_atm.pdb"):
        match = re.search(r"pocket(\d+)_atm", path.name)
        if not match:
            continue
        coords = []
        for line in path.read_text().splitlines():
            if line.startswith(("ATOM", "HETATM")):
                try:
                    coords.append(
                        (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                    )
                except ValueError:
                    continue
        if coords:
            n = len(coords)
            centers[int(match.group(1))] = tuple(
                sum(c[i] for c in coords) / n for i in range(3)
            )  # type: ignore[assignment]
    return centers
