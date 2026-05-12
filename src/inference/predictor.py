"""Unified prediction API consumed by the Streamlit GUI.

Two backends:
- **`http`** (default when `FRAUDGUARD_API_URL` is set in `.env`): calls the
  FraudGuard FastAPI Space hosted on Hugging Face. No sklearn inference runs
  locally — the GUI just sends JSON and renders responses.
- **`local`** (fallback): downloads model artifacts from HF Hub and runs sklearn
  inference in the GUI process. Useful for offline development.

The GUI never knows which backend is used — it calls `predict_paysim` /
`predict_custom` and gets the same response shape either way.
"""
from __future__ import annotations

import os
from typing import Any, Literal

import numpy as np
import pandas as pd
import requests

from src.config import DEFAULT_FRAUD_THRESHOLD, HF_TOKEN
from src.hf_hub.downloader import (
    get_custom_preprocessor,
    get_if_model,
    get_paysim_preprocessor,
    get_rf_model,
)
from src.models.isolation_forest import if_predict_fraud


API_URL = os.environ.get("FRAUDGUARD_API_URL", "").rstrip("/")
USE_HTTP = bool(API_URL)
TIMEOUT_S = float(os.environ.get("FRAUDGUARD_API_TIMEOUT", "30"))

Schema = Literal["paysim", "custom"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _to_records(payload: dict | pd.DataFrame) -> list[dict]:
    if isinstance(payload, pd.DataFrame):
        return payload.to_dict(orient="records")
    return [payload]


def _to_dataframe(payload: dict | pd.DataFrame) -> pd.DataFrame:
    if isinstance(payload, dict):
        return pd.DataFrame([payload])
    return payload.copy()


def _auth_headers() -> dict[str, str]:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN missing — required to authenticate with the FraudGuard API.")
    return {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }


class FraudGuardAPIError(RuntimeError):
    """Raised when the remote inference API returns a non-2xx response."""


def _post(path: str, body: dict) -> dict:
    url = f"{API_URL}{path}"
    try:
        resp = requests.post(url, json=body, headers=_auth_headers(), timeout=TIMEOUT_S)
    except requests.RequestException as exc:
        raise FraudGuardAPIError(f"Network error calling {url}: {exc}") from exc
    if not resp.ok:
        raise FraudGuardAPIError(
            f"{resp.status_code} from {url}: {resp.text[:300]}"
        )
    return resp.json()


# ---------------------------------------------------------------------------
# public API — same signature as before
# ---------------------------------------------------------------------------
def predict_paysim(
    payload: dict | pd.DataFrame,
    threshold: float = DEFAULT_FRAUD_THRESHOLD,
) -> dict:
    if USE_HTTP:
        body = {"transactions": _to_records(payload), "threshold": threshold}
        out = _post("/predict/paysim", body)
        return {
            "schema": out.get("dataset", "paysim"),
            "model": out.get("model", "random_forest"),
            "n": out["n"],
            "fraud_probability": out["fraud_probability"],
            "prediction": out["prediction"],
            "threshold": out.get("threshold", threshold),
            "backend": "http",
        }

    # Local fallback
    df = _to_dataframe(payload)
    pre = get_paysim_preprocessor()
    model = get_rf_model()
    x = pre.transform(df)
    prob = model.predict_proba(x)[:, 1]
    pred = (prob >= threshold).astype(int)
    return {
        "schema": "paysim",
        "model": "random_forest",
        "n": len(df),
        "fraud_probability": prob.tolist(),
        "prediction": pred.tolist(),
        "threshold": threshold,
        "backend": "local",
    }


def predict_custom(payload: dict | pd.DataFrame) -> dict:
    if USE_HTTP:
        body = {"transactions": _to_records(payload)}
        out = _post("/predict/custom", body)
        return {
            "schema": out.get("dataset", "custom"),
            "model": out.get("model", "isolation_forest"),
            "n": out["n"],
            "anomaly_score": out["anomaly_score"],
            "prediction": out["prediction"],
            "backend": "http",
        }

    # Local fallback
    df = _to_dataframe(payload)
    pre = get_custom_preprocessor()
    model = get_if_model()
    x = pre.transform(df)
    binary, score = if_predict_fraud(model, x)
    return {
        "schema": "custom",
        "model": "isolation_forest",
        "n": len(df),
        "anomaly_score": score.tolist(),
        "prediction": binary.tolist(),
        "backend": "local",
    }


def predict(payload: dict | pd.DataFrame, schema: Schema, **kwargs) -> dict:
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
    if paysim_payload is None and custom_payload is None:
        raise ValueError("Provide at least one of paysim_payload or custom_payload.")
    out: dict[str, Any] = {}
    rf_pred = if_pred = None
    if paysim_payload is not None:
        r = predict_paysim(paysim_payload, threshold=threshold)
        out["random_forest"] = r
        rf_pred = np.array(r["prediction"])
    if custom_payload is not None:
        r = predict_custom(custom_payload)
        out["isolation_forest"] = r
        if_pred = np.array(r["prediction"])
    if rf_pred is not None and if_pred is not None and len(rf_pred) == len(if_pred):
        out["ensemble_or"] = ((rf_pred | if_pred) > 0).astype(int).tolist()
        out["ensemble_and"] = ((rf_pred & if_pred) > 0).astype(int).tolist()
    return out


def health() -> dict:
    """Probe the remote API's /health endpoint, or report local backend status."""
    if not USE_HTTP:
        return {"backend": "local", "api_url": None}
    try:
        resp = requests.get(f"{API_URL}/health", timeout=TIMEOUT_S)
        resp.raise_for_status()
        return {"backend": "http", "api_url": API_URL, **resp.json()}
    except requests.RequestException as exc:
        return {"backend": "http", "api_url": API_URL, "error": str(exc)}


def get_metadata() -> dict:
    """Return metadata.json — either fetched from the API or from local HF Hub cache.

    Returns `{}` (not an exception) when nothing is available yet, so GUI pages
    can show a friendly "no training run yet" state."""
    if USE_HTTP:
        try:
            resp = requests.get(f"{API_URL}/metadata", headers=_auth_headers(), timeout=TIMEOUT_S)
            if resp.status_code == 503:
                return {}
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return {}
    from src.hf_hub.downloader import get_metadata as _local_meta
    return _local_meta()
