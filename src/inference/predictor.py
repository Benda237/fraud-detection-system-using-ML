"""Unified prediction API consumed by the Streamlit GUI.

Wraps both models (RF for PaySim-schema, IF for the custom schema) plus a
simple ensemble vote. Models and preprocessors are pulled lazily from HF Hub.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from src.config import DEFAULT_FRAUD_THRESHOLD
from src.hf_hub.downloader import (
    get_custom_preprocessor,
    get_if_model,
    get_paysim_preprocessor,
    get_rf_model,
)
from src.models.isolation_forest import if_predict_fraud

Schema = Literal["paysim", "custom"]


def _to_dataframe(payload: dict | pd.DataFrame) -> pd.DataFrame:
    if isinstance(payload, dict):
        return pd.DataFrame([payload])
    return payload.copy()


def predict_paysim(payload: dict | pd.DataFrame, threshold: float = DEFAULT_FRAUD_THRESHOLD) -> dict:
    df = _to_dataframe(payload)
    preprocessor = get_paysim_preprocessor()
    model = get_rf_model()

    x = preprocessor.transform(df)
    prob = model.predict_proba(x)[:, 1]
    pred = (prob >= threshold).astype(int)

    return {
        "schema": "paysim",
        "model": "random_forest",
        "n": len(df),
        "fraud_probability": prob.tolist(),
        "prediction": pred.tolist(),
        "threshold": threshold,
    }


def predict_custom(payload: dict | pd.DataFrame) -> dict:
    df = _to_dataframe(payload)
    preprocessor = get_custom_preprocessor()
    model = get_if_model()

    x = preprocessor.transform(df)
    binary, score = if_predict_fraud(model, x)

    return {
        "schema": "custom",
        "model": "isolation_forest",
        "n": len(df),
        "anomaly_score": score.tolist(),
        "prediction": binary.tolist(),
    }


def predict(payload: dict | pd.DataFrame, schema: Schema, **kwargs) -> dict:
    """Top-level dispatch consumed by the Streamlit GUI."""
    if schema == "paysim":
        return predict_paysim(payload, **kwargs)
    if schema == "custom":
        return predict_custom(payload, **kwargs)
    raise ValueError(f"Unknown schema: {schema!r}")


def ensemble_predict(
    paysim_payload: dict | pd.DataFrame | None = None,
    custom_payload: dict | pd.DataFrame | None = None,
    threshold: float = DEFAULT_FRAUD_THRESHOLD,
) -> dict:
    """Run both models and combine via OR (high-recall) and majority vote.
    Pass whichever payloads you have — at least one must be provided."""
    if paysim_payload is None and custom_payload is None:
        raise ValueError("Provide at least one of paysim_payload or custom_payload.")

    out: dict[str, Any] = {}
    rf_pred = if_pred = None
    if paysim_payload is not None:
        rf_result = predict_paysim(paysim_payload, threshold=threshold)
        out["random_forest"] = rf_result
        rf_pred = np.array(rf_result["prediction"])
    if custom_payload is not None:
        if_result = predict_custom(custom_payload)
        out["isolation_forest"] = if_result
        if_pred = np.array(if_result["prediction"])

    if rf_pred is not None and if_pred is not None and len(rf_pred) == len(if_pred):
        out["ensemble_or"] = ((rf_pred | if_pred) > 0).astype(int).tolist()
        out["ensemble_and"] = ((rf_pred & if_pred) > 0).astype(int).tolist()
    return out
