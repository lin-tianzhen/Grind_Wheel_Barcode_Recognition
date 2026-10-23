#!/usr/bin/env bash
# Convenience launcher used in the supplied sandbox.  On a normal machine,
# activate .venv and call the scripts directly instead.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONHOME="${PYTHONHOME:-/Users/lintianzhen/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python}"
export PYTHONPATH="$ROOT/.venv/lib/python3.12/site-packages:$ROOT/src"
export MPLCONFIGDIR="$ROOT/.mplconfig"
exec "$ROOT/python-unsigned" "$@"
