#!/usr/bin/env bash
# Copy this repository into a Hugging Face Space and push it.
#
# The Space needs Dockerfile and README.md at *its* root, while this repository
# keeps them in space/. Rather than restructure the repo to suit a deployment
# target, this assembles the Space layout in a temp directory.
#
#   usage:  bash space/push.sh <hf-username> [space-name]
set -euo pipefail

USER="${1:?usage: bash space/push.sh <hf-username> [space-name]}"
NAME="${2:-chilecule}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)/${NAME}"

echo "Cloning https://huggingface.co/spaces/${USER}/${NAME}"
git clone "https://huggingface.co/spaces/${USER}/${NAME}" "$WORK"
cd "$WORK"

# Everything the image needs. src/ and pyproject.toml because the Dockerfile
# pip-installs the package; space/app.py because the CMD runs it from there.
cp -R "$SRC/src" .
cp "$SRC/pyproject.toml" "$SRC/LICENSE" "$SRC/NOTICE" .
cp "$SRC/space/Dockerfile" "$SRC/space/README.md" .
mkdir -p space && cp "$SRC/space/app.py" space/app.py

# Nothing that could carry a secret or bloat the image.
cat > .gitignore <<'IGNORE'
.env
.env.*
__pycache__/
*.pyc
.venv/
data/
runs/
IGNORE

git add -A
git commit -m "chilecule: deterministic tool layer" || echo "nothing to commit"

echo
echo "Pushing. Username: ${USER}   Password: your Hugging Face WRITE token."
git push

echo
echo "Done. Build takes about ten minutes -- conda resolving smina and fpocket"
echo "is the slow part. Watch it at:"
echo "  https://huggingface.co/spaces/${USER}/${NAME}"
