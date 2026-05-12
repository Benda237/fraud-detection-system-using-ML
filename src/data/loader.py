"""Dataset loaders. Both PaySim and the custom 51k mixed-feature CSV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import CUSTOM_CSV, PAYSIM_CSV


class DatasetNotFoundError(FileNotFoundError):
    """Raised with a friendly message when a CSV is missing from data/raw/."""


def _require(path: Path, hint: str) -> Path:
    if not path.exists():
        raise DatasetNotFoundError(
            f"Missing dataset: {path}\n"
            f"Hint: {hint}\n"
            "Run `python scripts/download_data.py` or place the file manually."
        )
    return path


def load_paysim(path: Path | None = None) -> pd.DataFrame:
    """Load the PaySim mobile-money simulator dataset (~6.3M rows)."""
    csv_path = _require(
        path or PAYSIM_CSV,
        "Download PaySim from Kaggle slug 'ealaxi/paysim1' "
        "and save as data/raw/AIML Dataset.csv",
    )
    df = pd.read_csv(csv_path)
    return df


def load_custom(path: Path | None = None) -> pd.DataFrame:
    """Load the 51k-row mixed-feature fraud dataset (device, location, etc.)."""
    csv_path = _require(
        path or CUSTOM_CSV,
        "Download the custom dataset from Kaggle slug 'goyaladi/fraud-detection-dataset' "
        "and save as data/raw/Fraud Detection Dataset.csv",
    )
    df = pd.read_csv(csv_path)
    return df


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
