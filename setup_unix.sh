#!/usr/bin/env bash
set -euo pipefail
python3.11 --version
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q
echo 'Setup complete. Activate with: source .venv/bin/activate'
