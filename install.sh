#!/usr/bin/env bash
#
# chilecule installer.
#
# Two package managers, because no single one covers the ground:
#   uv          -- the Python layer. RDKit ships wheels, so pip suffices.
#   micromamba  -- smina, fpocket, and friends. These are compiled C++ binaries
#                  with no PyPI distribution; conda-forge is the only source.
#
# Nothing is installed globally. Python goes into ./.venv and binaries go into a
# micromamba environment under ./.micromamba, both removable with `rm -rf`.

set -euo pipefail

TIER="core"
YES=0
VENV_DIR="${VENV_DIR:-.venv}"
MAMBA_DIR="${MAMBA_DIR:-.micromamba}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[0;33m'; BOLD=$'\033[1m'; OFF=$'\033[0m'
info()  { printf '%s==>%s %s\n' "$GREEN" "$OFF" "$1"; }
warn()  { printf '%s!!%s  %s\n' "$YELLOW" "$OFF" "$1"; }
fail()  { printf '%sxx%s  %s\n' "$RED" "$OFF" "$1" >&2; exit 1; }

usage() {
    cat <<'USAGE'
Usage: ./install.sh [--tier core|dock|all] [--yes] [--help]

  --tier core   RDKit and the Python layer. Enough for dossier, sar, validate.
                (~2 minutes, ~400 MB)
  --tier dock   Adds smina and fpocket from conda-forge, enabling docking and
                pocket detection. (~4 minutes, ~1 GB)
  --tier all    Adds the MCP server, agent, and ML extras.

  --yes         Do not prompt before installing.

Environment overrides: VENV_DIR, MAMBA_DIR, PYTHON_VERSION.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier) TIER="${2:-}"; shift 2 ;;
        --tier=*) TIER="${1#*=}"; shift ;;
        --yes|-y) YES=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) fail "unknown option: $1  (try --help)" ;;
    esac
done

case "$TIER" in core|dock|all) ;; *) fail "unknown tier: $TIER" ;; esac

# ---------------------------------------------------------------- platform

OS="$(uname -s)"; ARCH="$(uname -m)"
case "${OS}-${ARCH}" in
    Darwin-arm64)  MAMBA_PLATFORM="osx-arm64" ;;
    Darwin-x86_64) MAMBA_PLATFORM="osx-64" ;;
    Linux-x86_64)  MAMBA_PLATFORM="linux-64" ;;
    Linux-aarch64) MAMBA_PLATFORM="linux-aarch64" ;;
    *) fail "unsupported platform ${OS}-${ARCH}. Install manually per docs/ARCHITECTURE.md." ;;
esac

printf '%s\n' "${BOLD}chilecule installer${OFF}"
printf '  platform : %s (%s)\n' "$MAMBA_PLATFORM" "${OS}-${ARCH}"
printf '  tier     : %s\n' "$TIER"
printf '  python   : %s -> %s/\n' "$PYTHON_VERSION" "$VENV_DIR"
[[ "$TIER" != "core" ]] && printf '  binaries : smina, fpocket -> %s/\n' "$MAMBA_DIR"
printf '\n'

if [[ "$YES" -eq 0 ]]; then
    read -r -p "Proceed? [y/N] " reply
    [[ "$reply" =~ ^[Yy] ]] || { echo "Aborted."; exit 0; }
fi

# ------------------------------------------------------------------- uv

if ! command -v uv >/dev/null 2>&1; then
    info "installing uv (Python package manager)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    command -v uv >/dev/null 2>&1 || fail "uv installed but not on PATH; open a new shell and re-run"
fi
info "uv $(uv --version | awk '{print $2}')"

# --------------------------------------------------------- Python layer

info "creating $VENV_DIR (Python $PYTHON_VERSION)"
# --seed installs pip into the venv. uv does not need it, but users reasonably
# expect `pip install something` to work after activating an environment, and
# a venv where pip is silently absent is a confusing thing to hand someone.
uv venv --seed --python "$PYTHON_VERSION" "$VENV_DIR" >/dev/null

EXTRAS=""
case "$TIER" in
    core) EXTRAS="" ;;
    dock) EXTRAS="" ;;
    all)  EXTRAS="[mcp,agent,dev]" ;;
esac

info "installing chilecule${EXTRAS} and dependencies"
VIRTUAL_ENV="$PWD/$VENV_DIR" uv pip install --quiet -e ".${EXTRAS}"

# ------------------------------------------------------------- binaries

if [[ "$TIER" != "core" ]]; then
    MAMBA_BIN="$MAMBA_DIR/bin/micromamba"
    if [[ ! -x "$MAMBA_BIN" ]]; then
        info "downloading micromamba for $MAMBA_PLATFORM"
        mkdir -p "$MAMBA_DIR"
        curl -Ls "https://micro.mamba.pm/api/micromamba/${MAMBA_PLATFORM}/latest" \
            | tar -xj -C "$MAMBA_DIR" bin/micromamba
        [[ -x "$MAMBA_BIN" ]] || fail "micromamba download failed"
    fi

    info "installing smina and fpocket from conda-forge"
    # smina and fpocket are Apache-2.0 and MIT respectively. The conda-forge
    # smina build links Open Babel (GPL-2.0); chilecule only ever invokes these
    # as separate processes, so that does not affect this codebase's license.
    # See docs/LICENSING.md.
    export MAMBA_ROOT_PREFIX="$PWD/$MAMBA_DIR"
    "$MAMBA_BIN" create -y -q -n chilecule-bins -c conda-forge smina fpocket >/dev/null

    BIN_PATH="$PWD/$MAMBA_DIR/envs/chilecule-bins/bin"
    ACTIVATE="$VENV_DIR/bin/activate"
    if ! grep -q "chilecule-bins" "$ACTIVATE" 2>/dev/null; then
        # Put the binaries on PATH whenever the venv is active, so that
        # `source .venv/bin/activate` is the only thing a user has to remember.
        #
        # APPENDED, not prepended. The conda-forge environment ships its own
        # python, and putting its bin directory first shadows the virtualenv's
        # interpreter -- `python -m pytest` and `pip install` would then run
        # against the wrong environment while `chilecule` kept working, because
        # console scripts hard-code their interpreter in the shebang. That is a
        # miserable bug to diagnose. Appending leaves the venv first for
        # anything it provides and falls through to conda for smina and
        # fpocket, which the venv does not provide.
        cat >> "$ACTIVATE" <<EOF

# Added by chilecule install.sh -- docking and pocket-detection binaries.
# Appended so the virtualenv's own python still wins.
export PATH="\$PATH:$BIN_PATH"
EOF
    fi
    info "binaries installed to $BIN_PATH"
fi

# ------------------------------------------------------------- verify

printf '\n'
info "verifying"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
chilecule doctor || warn "doctor reported problems -- see above"

cat <<EOF

${BOLD}Done.${OFF}

  source $VENV_DIR/bin/activate
  chilecule dossier EGFR
  chilecule validate EGFR
EOF
[[ "$TIER" != "core" ]] && cat <<EOF
  chilecule pockets 5CNN --chain A
EOF
printf '\nRemove everything with:  rm -rf %s %s\n' "$VENV_DIR" "$MAMBA_DIR"
