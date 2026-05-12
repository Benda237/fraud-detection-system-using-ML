#!/usr/bin/env bash
# Bundle the HF Space contents (Dockerfile, requirements, notebooks, src/) and
# push them to a HF Space repo. The Space then builds and serves JupyterLab,
# where you open the notebooks and run training — nothing runs on your local
# machine.
#
# Usage:
#   ./scripts/push_to_hf_space.sh                    # uses HF_USERNAME from .env, creates a private Space
#   ./scripts/push_to_hf_space.sh public             # makes the Space public
#
# Prereqs: huggingface_hub installed; HF_TOKEN in .env or environment.

set -euo pipefail

cd "$(dirname "$0")/.."

# Load .env
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

: "${HF_TOKEN:?HF_TOKEN not set in environment or .env}"
: "${HF_USERNAME:?HF_USERNAME not set in environment or .env}"

VIS="${1:-private}"
SPACE_NAME="fraudguard-training-lab"
SPACE_ID="${HF_USERNAME}/${SPACE_NAME}"
STAGE="$(mktemp -d)"
trap "rm -rf '$STAGE'" EXIT

echo "[1/4] Staging Space contents in $STAGE"
cp -R hf_space/* "$STAGE/"
mkdir -p "$STAGE/src"
cp -R src/* "$STAGE/src/"   # ship src/ so notebooks can import the same preprocessor classes

# Remove __pycache__ junk
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
    commit_message="Sync FraudGuard training lab",
)
print("Upload complete")
PY

echo "[4/4] Done."
echo "  Open: https://huggingface.co/spaces/${SPACE_ID}"
echo "  Set HF_TOKEN, KAGGLE_USERNAME, KAGGLE_KEY as Space secrets (Settings → Variables and secrets)."
echo "  Then open the Jupyter tab and run notebooks 01 → 05 in order."
