"""Central configuration for the fraud detection system.

Loads secrets from .env, defines paths, hyperparameters, and HF repo identifiers.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# -------- Paths --------
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_LOCAL_DIR = ROOT_DIR / "models_local"
HF_CACHE_DIR = ROOT_DIR / ".hf_cache"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_LOCAL_DIR, HF_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# -------- Dataset filenames --------
PAYSIM_CSV = RAW_DIR / "AIML Dataset.csv"
CUSTOM_CSV = RAW_DIR / "Fraud Detection Dataset.csv"

# -------- Kaggle dataset slugs --------
# PaySim mobile-money simulator dataset
PAYSIM_KAGGLE_SLUG = "ealaxi/paysim1"
# Generic fraud detection dataset matching the 51k mixed-feature schema
CUSTOM_KAGGLE_SLUG = "goyaladi/fraud-detection-dataset"

# -------- Hugging Face --------
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_USERNAME = os.getenv("HF_USERNAME", "mdhaggai")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "fraud-detection-models")
HF_REPO_ID = f"{HF_USERNAME}/{HF_MODEL_REPO}"

# -------- Kaggle credentials --------
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY = os.getenv("KAGGLE_KEY", "")

# -------- Model hyperparameters --------
RANDOM_STATE = 42
TEST_SIZE = 0.2

RF_PARAMS = {
    "n_estimators": 150,
    "max_depth": 12,
    "min_samples_split": 5,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

IF_PARAMS = {
    "n_estimators": 150,
    "max_samples": "auto",
    "contamination": "auto",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# -------- Rule-based baseline thresholds --------
RULE_LARGE_AMOUNT = 200_000.0
RULE_BALANCE_DRAINED_TOLERANCE = 1.0

# -------- Decision threshold (probability above which we call fraud) --------
DEFAULT_FRAUD_THRESHOLD = 0.5

# -------- Artifact filenames inside the HF model repo --------
ARTIFACTS = {
    "rf_model": "random_forest.joblib",
    "if_model": "isolation_forest.joblib",
    "paysim_preprocessor": "paysim_preprocessor.joblib",
    "custom_preprocessor": "custom_preprocessor.joblib",
    "metadata": "metadata.json",
    "model_card": "README.md",
}
