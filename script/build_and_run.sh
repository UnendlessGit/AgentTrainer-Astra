#!/usr/bin/env bash
set -euo pipefail
ASTRA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ASTRA_ROOT"
exec .venv/bin/python scripts/run_app.py "${1:-run}"
