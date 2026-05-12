"""Reusable preprocessors for both datasets.

Each preprocessor is a fit/transform object that bundles imputation, encoding,
and scaling so the exact same transformation can be applied at inference time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class PaySimPreprocessor:
    """PaySim schema: step, type, amount, oldbalanceOrg, newbalanceOrig,
    oldbalanceDest, newbalanceDest, isFlaggedFraud, (isFraud).

    nameOrig / nameDest are dropped — they're high-cardinality identifiers
    with near-zero predictive power (the original notebook confirmed this).
    """

    drop_cols: tuple[str, ...] = ("nameOrig", "nameDest")
    target_col: str = "isFraud"
    type_encoder: LabelEncoder = field(default_factory=LabelEncoder)
    scaler: StandardScaler = field(default_factory=StandardScaler)
    feature_cols: list[str] = field(default_factory=list)
    fitted: bool = False

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # Derived features that historically improve fraud detection on PaySim
        out["balance_delta_orig"] = out["oldbalanceOrg"] - out["newbalanceOrig"]
        out["balance_delta_dest"] = out["newbalanceDest"] - out["oldbalanceDest"]
        out["amount_equals_orig_balance"] = (
            np.isclose(out["amount"], out["oldbalanceOrg"], atol=1.0)
        ).astype(int)
        out["dest_was_empty"] = (out["oldbalanceDest"] == 0).astype(int)
        return out

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        df = self._engineer(df)
        y = df[self.target_col].to_numpy().astype(int)
        x = df.drop(columns=[c for c in (*self.drop_cols, self.target_col) if c in df.columns])

        x["type"] = self.type_encoder.fit_transform(x["type"].astype(str))
        self.feature_cols = x.columns.tolist()
        x_scaled = self.scaler.fit_transform(x.to_numpy())
        self.fitted = True
        return x_scaled, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("PaySimPreprocessor must be fit before transform.")
        df = self._engineer(df)
        x = df.drop(columns=[c for c in (*self.drop_cols, self.target_col) if c in df.columns])
        # Handle unseen labels gracefully
        known = set(self.type_encoder.classes_)
        x["type"] = x["type"].astype(str).map(
            lambda v: v if v in known else self.type_encoder.classes_[0]
        )
        x["type"] = self.type_encoder.transform(x["type"])
        x = x[self.feature_cols]
        return self.scaler.transform(x.to_numpy())

    def transform_single(self, payload: dict[str, Any]) -> np.ndarray:
        """Convenience for the GUI: single-transaction dict -> 2D array."""
        df = pd.DataFrame([payload])
        # Add target placeholder if absent (it's dropped anyway)
        if self.target_col not in df.columns:
            df[self.target_col] = 0
        return self.transform(df)


@dataclass
class CustomPreprocessor:
    """Custom schema: Transaction_ID, User_ID, Transaction_Amount, Transaction_Type,
    Time_of_Transaction, Device_Used, Location, Previous_Fraudulent_Transactions,
    Account_Age, Number_of_Transactions_Last_24H, Payment_Method, Fraudulent.
    """

    drop_cols: tuple[str, ...] = ("Transaction_ID",)
    target_col: str = "Fraudulent"
    encoders: dict[str, LabelEncoder] = field(default_factory=dict)
    scaler: StandardScaler = field(default_factory=StandardScaler)
    feature_cols: list[str] = field(default_factory=list)
    fitted: bool = False

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in out.columns:
            if out[col].dtype == "object":
                out[col] = out[col].fillna("Unknown")
            else:
                out[col] = out[col].fillna(out[col].median())
        return out

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        df = self._impute(df)
        y = df[self.target_col].to_numpy().astype(int)
        x = df.drop(columns=[c for c in (*self.drop_cols, self.target_col) if c in df.columns])

        for col in x.select_dtypes(include=["object"]).columns:
            enc = LabelEncoder()
            x[col] = enc.fit_transform(x[col].astype(str))
            self.encoders[col] = enc

        self.feature_cols = x.columns.tolist()
        x_scaled = self.scaler.fit_transform(x.to_numpy())
        self.fitted = True
        return x_scaled, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("CustomPreprocessor must be fit before transform.")
        df = self._impute(df)
        x = df.drop(columns=[c for c in (*self.drop_cols, self.target_col) if c in df.columns])
        for col, enc in self.encoders.items():
            if col not in x.columns:
                continue
            known = set(enc.classes_)
            x[col] = x[col].astype(str).map(lambda v: v if v in known else "Unknown")
            if "Unknown" not in known:
                enc_classes = np.append(enc.classes_, "Unknown")
                enc.classes_ = enc_classes
            x[col] = enc.transform(x[col])
        x = x[self.feature_cols]
        return self.scaler.transform(x.to_numpy())

    def transform_single(self, payload: dict[str, Any]) -> np.ndarray:
        df = pd.DataFrame([payload])
        if self.target_col not in df.columns:
            df[self.target_col] = 0
        return self.transform(df)
