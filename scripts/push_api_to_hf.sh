#!/usr/bin/env bash
# Bundle the FastAPI inference Space (Dockerfile, requirements, app/, src/)
# and push to a Hugging Face Space. After build, the Space exposes endpoints
# at https://<HF_USERNAME>-fraudguard-api.hf.space
#
# Usage:
#   ./scripts/push_api_to_hf.sh                    # private Space (recommended)
#   ./scripts/push_api_to_hf.sh public
#
# Prereqs: huggingface_hub installed; HF_TOKEN + HF_USERNAME in .env.

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

: "${HF_TOKEN:?HF_TOKEN not set in environment or .env}"
: "${HF_USERNAME:?HF_USERNAME not set in environment or .env}"

VIS="${1:-private}"
SPACE_NAME="fraudguard-api"
SPACE_ID="${HF_USERNAME}/${SPACE_NAME}"
STAGE="$(mktemp -d)"
trap "rm -rf '$STAGE'" EXIT

echo "[1/4] Staging API Space contents in $STAGE"
cp -R api_space/* "$STAGE/"
mkdir -p "$STAGE/src"
cp -R src/* "$STAGE/src/"

find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf '{}' \; 2>/dev/null || true
find "$STAGE" -name '.ipynb_checkpoints' -type d -prune -exec rm -rf '{}' \; 2>/dev/null || true

echo "[2/4] Creating Space $SPACE_ID (visibility: $VIS)"
PRIVATE_FLAG="True"
[[ "$VIS" == "public" ]] && PRIVATE_FLAG="False"
.venv/bin/python - <<PY
from huggingface_hub import create_repo
create_repo(
    repo_id="${SPACE_ID}",
    repo_type="space",
    space_sdk="docker",
    private=${PRIVATE_FLAG},
    token="${HF_TOKEN}",
    exist_ok=True,
)
print("Space created/verified")
PY

echo "[3/4] Uploading bundle"
.venv/bin/python - <<PY
from huggingface_hub import upload_folder
upload_folder(
    folder_path="${STAGE}",
    repo_id="${SPACE_ID}",
    repo_type="space",
    token="${HF_TOKEN}",
    commit_message="Sync FraudGuard inference API",
)
print("Upload complete")
PY

echo "[4/4] Done."
echo "  Space:     https://huggingface.co/spaces/${SPACE_ID}"
echo "  Live URL:  https://${HF_USERNAME}-${SPACE_NAME}.hf.space"
echo ""
echo "  Set HF_TOKEN + HF_USERNAME as Space secrets, then add this to your local .env:"
echo "    FRAUDGUARD_API_URL=https://${HF_USERNAME}-${SPACE_NAME}.hf.space"
