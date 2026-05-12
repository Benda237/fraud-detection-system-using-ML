"""Standard classification metrics suite — accuracy, precision, recall, F1,
ROC-AUC, plus the full confusion matrix. Returns a serializable dict so it
can be cached as metadata.json on the HF model repo.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
    label: str = "model",
) -> dict[str, Any]:
    """Return a flat dict of metrics. `y_score` is the probability of fraud
    (RF: predict_proba[:,1]; IF: -decision_function). Omit if unavailable."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # cm is [[TN, FP],[FN, TP]]
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    out: dict[str, Any] = {
        "label": label,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "support_positive": int(np.sum(y_true == 1)),
        "support_negative": int(np.sum(y_true == 0)),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
            fpr, tpr, _ = roc_curve(y_true, y_score)
            out["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))}
        except ValueError:
            out["roc_auc"] = None
    out["classification_report"] = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    return out
