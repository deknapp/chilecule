"""Environment and credential detection.

This module reads credentials. It never stores, prompts for, or writes them.

Both SDKs this project can use already implement a correct, well-documented
credential resolution chain, and every user who has run ``aws configure``,
``aws sso login``, or ``ant auth login`` is already configured. Adding a
``config.yaml`` with an ``api_key:`` field on top of that would add a way to
leak a secret into a git repository and would remove support for SSO,
instance roles, and short-lived tokens. So there is no such file.

What this module does is *detect* what is present, so ``chilecule doctor`` can
tell the user which capability tiers are unlocked without making them guess.

Anthropic resolution order (implemented by the Anthropic SDK):
    ANTHROPIC_API_KEY -> ANTHROPIC_AUTH_TOKEN -> OAuth profile written by
    ``ant auth login`` under ~/.config/anthropic/ -> workload identity
    federation -> default on-disk profile.

AWS resolution order (implemented by botocore):
    environment variables -> AWS_PROFILE / shared credentials and config files
    -> SSO cache -> container and instance metadata roles.
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
    aws: Capability
    binaries: dict[str, Capability] = field(default_factory=dict)
    packages: dict[str, Capability] = field(default_factory=dict)

    @property
    def tiers(self) -> dict[str, bool]:
        """Which workflow tiers can actually run."""
        has_rdkit = self.packages.get("rdkit", Capability("rdkit", False, "")).available
        has_docking = any(
            self.binaries.get(b, Capability(b, False, "")).available
            for b in ("smina", "vina", "gnina")
        )
        return {
            "data-and-sar": has_rdkit,
            "structure-based": has_rdkit and has_docking,
            "agent-driven": self.anthropic.available,
            "cloud-burst": False,  # scaffold only; see runners/aws_batch.py
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

    if os.environ.get("CLAUDE_CODE_USE_BEDROCK") and aws_credentials_present()[0]:
        return True, "routing to Claude on Amazon Bedrock via AWS credentials"

    return False, "no Anthropic credential found"


def aws_credentials_present() -> tuple[bool, str]:
    """Detect AWS credentials through botocore when available, env vars otherwise."""
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return True, "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are set"

    try:
        import botocore.session  # noqa: PLC0415

        session = botocore.session.get_session()
        credentials = session.get_credentials()
        if credentials is not None:
            method = getattr(credentials, "method", "unknown")
            profile = os.environ.get("AWS_PROFILE", "default")
            return True, f"botocore resolved credentials via {method} (profile: {profile})"
    except ImportError:
        pass
    except Exception as exc:
        return False, f"AWS credential lookup failed: {type(exc).__name__}"

    if (Path.home() / ".aws" / "credentials").exists():
        return True, "~/.aws/credentials exists (not validated -- install boto3 to verify)"
    return False, "no AWS credentials found"


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
    aws_ok, aws_detail = aws_credentials_present()

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
        name: _package(name)
        for name in ("rdkit", "pandas", "numpy", "sklearn", "mcp", "boto3")
    }

    return Environment(
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        anthropic=Capability(
            "anthropic", anthropic_ok, anthropic_detail,
            "export ANTHROPIC_API_KEY=... or run: ant auth login",
        ),
        aws=Capability(
            "aws", aws_ok, aws_detail,
            "run: aws configure   (or aws sso login)",
        ),
        binaries=binaries,
        packages=packages,
    )
