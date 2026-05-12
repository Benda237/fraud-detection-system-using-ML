"""Dataset loaders.

Datasets live on Hugging Face Hub (private dataset repo created by notebook 01
in the HF Space). The Streamlit GUI uses these helpers to pull samples for the
Data Explorer and Live Monitor pages — nothing is stored long-term on the
local disk, only HF's transparent download cache.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from src.config import HF_TOKEN, HF_USERNAME


def _dataset_repo() -> str:
    return f"{HF_USERNAME}/fraud-detection-datasets"


def _download(filename: str) -> Path:
    return Path(
        hf_hub_download(
            repo_id=_dataset_repo(),
            filename=filename,
            repo_type="dataset",
            token=HF_TOKEN or None,
        )
    )


@lru_cache(maxsize=2)
def load_paysim() -> pd.DataFrame:
    """Pull PaySim from the HF dataset repo (cached on disk and in memory)."""
    return pd.read_csv(_download("paysim.csv"))


@lru_cache(maxsize=2)
def load_custom() -> pd.DataFrame:
    return pd.read_csv(_download("custom.csv"))


def summarize(df: pd.DataFrame, name: str) -> dict:
    """Quick descriptive summary used in EDA and the GUI."""
    target_col = "isFraud" if "isFraud" in df.columns else (
        "Fraudulent" if "Fraudulent" in df.columns else None
    )
    fraud_rate = float(df[target_col].mean()) if target_col else None
    return {
        "name": name,
        "rows": len(df),
        "cols": df.shape[1],
        "fraud_rate": fraud_rate,
        "nulls": int(df.isna().sum().sum()),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }
