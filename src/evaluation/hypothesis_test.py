"""McNemar's test for comparing the ML model against the rule-based baseline.

Used to support the study's hypothesis test:
  H1 (null): ML-FDS does NOT significantly improve accuracy.
  HA: ML-FDS significantly improves accuracy.
"""
from __future__ import annotations

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar


def mcnemar_test(y_true: np.ndarray, y_pred_ml: np.ndarray, y_pred_baseline: np.ndarray) -> dict:
    """Compare two classifiers on the same test set with McNemar's test.

    The 2x2 contingency table counts disagreements:
        b = baseline_correct & ml_wrong
        c = baseline_wrong   & ml_correct
    If c >> b, the ML model is reliably better.
    """
    ml_correct = (y_pred_ml == y_true)
    base_correct = (y_pred_baseline == y_true)

    a = int(np.sum(ml_correct & base_correct))
    b = int(np.sum(base_correct & ~ml_correct))
    c = int(np.sum(~base_correct & ml_correct))
    d = int(np.sum(~ml_correct & ~base_correct))

    table = [[a, b], [c, d]]
    # exact=True is more reliable when b+c is small
    result = mcnemar(table, exact=(b + c) < 25, correction=True)

    return {
        "table": {"ml_right_base_right": a, "ml_wrong_base_right": b,
                  "ml_right_base_wrong": c, "ml_wrong_base_wrong": d},
        "statistic": float(result.statistic) if result.statistic is not None else None,
        "p_value": float(result.pvalue),
        "reject_null_at_0.05": bool(result.pvalue < 0.05),
        "interpretation": (
            "ML-FDS significantly improves accuracy (reject H1, accept HA)"
            if result.pvalue < 0.05 and c > b
            else "Insufficient evidence to reject H1"
        ),
    }
