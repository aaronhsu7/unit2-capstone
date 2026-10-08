#!/usr/bin/env bash
# Environment setup for the Unit 2 capstone (README Step 2).
#
# Usage:  bash setup.sh
set -euo pipefail

cd "$(dirname "$0")"

# --- Pick a Python interpreter ---------------------------------------------
# The checklist asks for Python 3.11; fall back to python3 if 3.11 isn't installed.
if command -v python3.11 >/dev/null 2>&1; then
    PYTHON=python3.11
else
    PYTHON=python3
    echo "WARNING: python3.11 not found, using $($PYTHON --version). The checklist asks for 3.11."
fi

# --- Virtual environment ---------------------------------------------------
if [ -x venv/bin/python ]; then
    echo "Reusing existing venv ($(venv/bin/python --version))"
else
    echo "Creating venv with $($PYTHON --version)..."
    "$PYTHON" -m venv venv
fi

# --- Dependencies ----------------------------------------------------------
# sentence-transformers pulls in PyTorch, so the first install can take a while.
echo "Installing dependencies..."
venv/bin/pip install --upgrade pip
venv/bin/pip install google-genai chromadb sentence-transformers python-dotenv
venv/bin/pip freeze > requirements.txt
echo "Wrote requirements.txt"

