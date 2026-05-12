"""Central configuration for the local GUI.

Loads secrets from `.env`, defines local cache paths, hyperparameter defaults
exposed to the UI, and Hugging Face repo identifiers.

All training/notebook execution happens on the Hugging Face Space —
`hf_space/notebooks/`. This file is consumed only by the local Streamlit GUI.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# -------- Local cache paths (GUI uses these as scratch) --------
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
HF_CACHE_DIR = ROOT_DIR / ".hf_cache"

for _d in (RAW_DIR, PROCESSED_DIR, HF_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# -------- Hugging Face --------
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_USERNAME = os.getenv("HF_USERNAME", "mdhaggai")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "fraud-detection-models")
HF_REPO_ID = f"{HF_USERNAME}/{HF_MODEL_REPO}"

# -------- Decision threshold (probability above which we call fraud) --------
DEFAULT_FRAUD_THRESHOLD = 0.5

# -------- Rule-based baseline thresholds (used by GUI Single-Transaction page) --------
RULE_LARGE_AMOUNT = 200_000.0
RULE_BALANCE_DRAINED_TOLERANCE = 1.0

# -------- Artifact filenames in the HF model repo --------
ARTIFACTS = {
    "rf_model": "random_forest.joblib",
    "if_model": "isolation_forest.joblib",
    "paysim_preprocessor": "paysim_preprocessor.joblib",
    "custom_preprocessor": "custom_preprocessor.joblib",
    "metadata": "metadata.json",
    "model_card": "README.md",
}
