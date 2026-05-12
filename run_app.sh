#!/usr/bin/env bash
# Convenience launcher for the FraudGuard Streamlit GUI.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# Pick the python interpreter from venv if present
if [[ -x ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
elif [[ -x "venv/bin/python" ]]; then
    PY="venv/bin/python"
else
    PY="python3"
fi

PYTHONPATH="$HERE" "$PY" -m streamlit run "$HERE/app/Home.py" "$@"
