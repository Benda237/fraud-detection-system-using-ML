"""IsolationForest inference helper. Training itself happens on the HF Space —
this module only exists locally so the Streamlit GUI can interpret a fitted
IsolationForest pulled from HF Hub.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


def if_predict_fraud(model: IsolationForest, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (binary_prediction, anomaly_score).

    - Binary: 1 = fraud, 0 = legit
    - Score: -decision_function — higher means more anomalous / more fraud-like.
    """
    raw = model.predict(x)                  # -1 anomaly, +1 normal
    binary = (raw == -1).astype(int)
    score = -model.decision_function(x)
    return binary, score
