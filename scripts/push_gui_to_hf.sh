#!/usr/bin/env bash
# Bundle the Streamlit GUI Space (Dockerfile + requirements + app/ + src/ + .streamlit/)
# and push to HF. Hosts the FraudGuard frontend at
# https://<HF_USERNAME>-fraudguard-gui.hf.space
#
# Usage:
#   ./scripts/push_gui_to_hf.sh            # private
#   ./scripts/push_gui_to_hf.sh public     # publicly accessible

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

: "${HF_TOKEN:?HF_TOKEN not set in environment or .env}"
: "${HF_USERNAME:?HF_USERNAME not set in environment or .env}"

VIS="${1:-private}"
SPACE_NAME="fraudguard-gui"
SPACE_ID="${HF_USERNAME}/${SPACE_NAME}"
STAGE="$(mktemp -d)"
trap "rm -rf '$STAGE'" EXIT

echo "[1/4] Staging GUI Space contents in $STAGE"
cp -R gui_space/* "$STAGE/"
mkdir -p "$STAGE/src" "$STAGE/app" "$STAGE/.streamlit"
cp -R src/*       "$STAGE/src/"
cp -R app/*       "$STAGE/app/"
cp -R .streamlit/* "$STAGE/.streamlit/"

find "$STAGE" -name '__pycache__'     -type d -prune -exec rm -rf '{}' \; 2>/dev/null || true
find "$STAGE" -name '.ipynb_checkpoints' -type d -prune -exec rm -rf '{}' \; 2>/dev/null || true

echo "[2/4] Creating Space $SPACE_ID (visibility: $VIS)"
PRIVATE_FLAG="True"; [[ "$VIS" == "public" ]] && PRIVATE_FLAG="False"
.venv/bin/python - <<PY
from huggingface_hub import create_repo
create_repo(repo_id="${SPACE_ID}", repo_type="space", space_sdk="docker",
            private=${PRIVATE_FLAG}, token="${HF_TOKEN}", exist_ok=True)
print("Space created/verified")
PY

echo "[3/4] Uploading bundle"
.venv/bin/python - <<PY
from huggingface_hub import upload_folder
upload_folder(folder_path="${STAGE}", repo_id="${SPACE_ID}", repo_type="space",
              token="${HF_TOKEN}", commit_message="Sync FraudGuard GUI")
print("Upload complete")
PY

echo "[4/4] Done."
echo "  Space:    https://huggingface.co/spaces/${SPACE_ID}"
echo "  Live URL: https://${HF_USERNAME}-${SPACE_NAME}.hf.space"
