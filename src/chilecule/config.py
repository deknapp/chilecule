"""Environment and credential detection.

This module reads credentials. It never stores, prompts for, or writes them.

The Anthropic SDK already implements a correct, documented credential
resolution chain, and anyone who has run ``ant auth login`` or exported a key
is already configured. Adding a ``config.yaml`` with an ``api_key:`` field on
top of that would create a way to leak a secret into a git repository and
would drop support for OAuth profiles and short-lived tokens. So there is no
such file.

What this module does is *detect* what is present, so ``chilecule doctor`` can
report which capabilities are available without making the user guess.

Anthropic resolution order (implemented by the SDK):
    ANTHROPIC_API_KEY -> ANTHROPIC_AUTH_TOKEN -> OAuth profile written by
    ``ant auth login`` under ~/.config/anthropic/ -> default on-disk profile.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

# External programs the structure-based workflows can use. Each is invoked as a
# separate process over files -- never imported or linked -- which is what
# keeps this Apache-2.0 codebase clear of their licenses. See docs/LICENSING.md.
EXTERNAL_BINARIES: dict[str, str] = {
    "smina": "docking (Apache-2.0 source; conda-forge binary links Open Babel, GPL-2.0)",
    "vina": "docking, AutoDock Vina 1.2+ (Apache-2.0)",
    "gnina": "docking with a CNN rescoring function (Apache-2.0)",
    "fpocket": "binding-site detection (MIT)",
    "obabel": "format conversion, optional (Open Babel, GPL-2.0 -- subprocess only)",
}


@dataclass
class Capability:
    name: str
    available: bool
    detail: str
    how_to_enable: str = ""


@dataclass
class Environment:
    """Everything ``chilecule doctor`` reports."""

    python_version: str
    anthropic: Capability
    binaries: dict[str, Capability] = field(default_factory=dict)
    packages: dict[str, Capability] = field(default_factory=dict)

    @property
    def tiers(self) -> dict[str, bool]:
        """Which workflows can actually run."""
        has_rdkit = self.packages.get("rdkit", Capability("rdkit", False, "")).available
        has_docking = any(
            self.binaries.get(b, Capability(b, False, "")).available
            for b in ("smina", "vina", "gnina")
        )
        return {
            "cheminformatics": has_rdkit,
            "structure-based": has_rdkit and has_docking,
            "agent-driven": self.anthropic.available,
        }


def anthropic_credentials_present() -> tuple[bool, str]:
    """Detect an Anthropic credential without reading its value."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True, "ANTHROPIC_API_KEY is set"
    if os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True, "ANTHROPIC_AUTH_TOKEN is set"

    profile_dir = Path.home() / ".config" / "anthropic"
    if profile_dir.exists() and any(profile_dir.iterdir()):
        active = os.environ.get("ANTHROPIC_PROFILE", "default")
        return True, f"OAuth profile in ~/.config/anthropic (profile: {active})"

    return False, "no Anthropic credential found"


def _package(name: str) -> Capability:
    import importlib.util

    spec = importlib.util.find_spec(name)
    if spec is None:
        return Capability(name, False, "not installed", f"pip install {name}")
    try:
        version = __import__(name).__version__
    except Exception:
        version = "installed"
    return Capability(name, True, str(version))


def detect() -> Environment:
    """Inspect the environment. Performs no network calls and no writes."""
    import sys

    anthropic_ok, anthropic_detail = anthropic_credentials_present()

    binaries = {}
    for binary, description in EXTERNAL_BINARIES.items():
        path = shutil.which(binary)
        binaries[binary] = Capability(
            name=binary,
            available=path is not None,
            detail=(
                f"{path}  --  {description}" if path else f"not on PATH  --  {description}"
            ),
            how_to_enable=f"micromamba install -c conda-forge {binary}",
        )

    packages = {
        name: _package(name) for name in ("rdkit", "pandas", "numpy", "sklearn", "mcp")
    }

    return Environment(
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        anthropic=Capability(
            "anthropic", anthropic_ok, anthropic_detail,
            "export ANTHROPIC_API_KEY=... or run: ant auth login",
        ),
        binaries=binaries,
        packages=packages,
    )
