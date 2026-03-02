#!/usr/bin/env bash
# Llama.cpp GPU Service Manager launcher
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

# Create venv and install deps if needed
if [[ ! -x "$VENV/bin/python" ]]; then
    echo "Setting up Python environment..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install textual requests --quiet
fi

exec "$VENV/bin/python" "$SCRIPT_DIR/app.py" "$@"
