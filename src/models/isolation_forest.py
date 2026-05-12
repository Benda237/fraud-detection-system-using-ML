"""Isolation Forest wrapper for unsupervised anomaly detection.

Trained primarily on non-fraud data; the `contamination` hyperparameter is
inferred from the observed fraud rate so the decision boundary is calibrated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest

from src.config import IF_PARAMS, RANDOM_STATE


@dataclass
class IFTrainingResult:
    model: IsolationForest
    n_train: int
    contamination: float


def train_isolation_forest(
    x_train: np.ndarray,
    y_train: np.ndarray | None = None,
    contamination: float | None = None,
) -> IFTrainingResult:
    """Fit an IsolationForest. If y_train is supplied, contamination is
    inferred from its fraud rate (clamped to [0.001, 0.5])."""
    params = IF_PARAMS.copy()
    if contamination is None and y_train is not None:
        rate = float(np.mean(y_train))
        contamination = float(np.clip(rate, 0.001, 0.5)) if rate > 0 else "auto"
    if contamination is not None:
        params["contamination"] = contamination

    # Train on legitimate-only when labels available — classic IF protocol
    if y_train is not None:
        x_fit = x_train[y_train == 0]
    else:
        x_fit = x_train

    params["random_state"] = RANDOM_STATE
    model = IsolationForest(**params)
    model.fit(x_fit)

    return IFTrainingResult(
        model=model,
        n_train=len(x_fit),
        contamination=params["contamination"] if isinstance(params["contamination"], float) else -1.0,
    )


def if_predict_fraud(model: IsolationForest, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (binary_pred, anomaly_score). Binary: 1=fraud, 0=legit.
    Lower decision_function => more anomalous => more likely fraud."""
    raw = model.predict(x)            # -1 anomaly, +1 normal
    binary = (raw == -1).astype(int)
    score = -model.decision_function(x)  # higher means more fraud-like
    return binary, score
