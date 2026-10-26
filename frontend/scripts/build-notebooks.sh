#!/bin/sh
# Build the JupyterLite preview into public/notebooks/ so `npm run dev` can
# serve it. The Docker image builds its own copy (Dockerfile, stage "lite").
#
# Needs Python 3.10 or later. The pinned build tools are installed once into
# frontend/.jupyterlite-venv; rerun this script after editing a notebook.
set -eu

cd "$(dirname "$0")/.."
VENV=.jupyterlite-venv
if [ ! -x "$VENV/bin/jupyter" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --disable-pip-version-check -q -r jupyterlite/requirements.txt
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/contents" "$WORK/lite"
cp -R ../notebooks/. "$WORK/contents"
rm -rf "$WORK/contents/tools" "$WORK/contents/README.md"

rm -rf public/notebooks
"$VENV/bin/jupyter" lite build \
    --lite-dir "$WORK/lite" \
    --contents "$WORK/contents" \
    --output-dir "$PWD/public/notebooks" \
    --no-sourcemaps
echo "JupyterLite site written to public/notebooks/"
