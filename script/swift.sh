#!/bin/sh
set -eu
astra_project=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec "$astra_project/.venv/bin/python" "$astra_project/scripts/run_swift.py" "$@"
