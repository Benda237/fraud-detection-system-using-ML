"""End-to-end training orchestrator. Loads each dataset, fits the
preprocessor, trains the model, evaluates, and saves artifacts locally.
The HF uploader picks them up from MODELS_LOCAL_DIR afterwards.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ARTIFACTS,
    MODELS_LOCAL_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data.loader import load_custom, load_paysim
from src.data.preprocessor import CustomPreprocessor, PaySimPreprocessor
from src.evaluation.hypothesis_test import mcnemar_test
from src.evaluation.metrics import evaluate_classifier
from src.models.isolation_forest import if_predict_fraud, train_isolation_forest
from src.models.random_forest import train_random_forest
from src.models.rule_based_baseline import custom_rule_predict, paysim_rule_predict


def _save(obj, name: str) -> Path:
    path = MODELS_LOCAL_DIR / name
    joblib.dump(obj, path)
    return path


def train_paysim_pipeline(sample_n: int | None = None) -> dict:
    """Train RF on PaySim. Optionally subsample for fast iteration."""
    df = load_paysim()
    if sample_n is not None and sample_n < len(df):
        # Stratified subsample: keep ALL fraud rows, downsample legit
        fraud = df[df["isFraud"] == 1]
        legit_cap = max(sample_n - len(fraud), 0)
        legit = df[df["isFraud"] == 0].sample(
            n=min(legit_cap, len(df[df["isFraud"] == 0])),
            random_state=RANDOM_STATE,
        )
        df = pd.concat([fraud, legit]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    pre = PaySimPreprocessor()
    x, y = pre.fit_transform(df)
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    result = train_random_forest(x_tr, y_tr, feature_names=pre.feature_cols, use_smote=True)
    y_pred = result.model.predict(x_te)
    y_prob = result.model.predict_proba(x_te)[:, 1]

    rf_metrics = evaluate_classifier(y_te, y_pred, y_prob, label="random_forest")

    # Rule-based baseline on the same test rows
    df_with_idx = df.reset_index(drop=True)
    _, idx_test = train_test_split(
        df_with_idx.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    rule_pred = paysim_rule_predict(df_with_idx.loc[idx_test])
    rule_metrics = evaluate_classifier(y_te, rule_pred, None, label="rule_based_paysim")

    mcnemar_result = mcnemar_test(y_te, y_pred, rule_pred)

    _save(pre, ARTIFACTS["paysim_preprocessor"])
    _save(result.model, ARTIFACTS["rf_model"])

    return {
        "rf": rf_metrics,
        "rule_based": rule_metrics,
        "mcnemar": mcnemar_result,
        "feature_importances": result.feature_importances,
        "n_train_before_smote": result.n_train,
        "n_train_after_smote": result.n_train_after_smote,
        "fraud_rate_before_smote": result.fraud_rate_before,
        "fraud_rate_after_smote": result.fraud_rate_after,
    }


def train_custom_pipeline() -> dict:
    """Train Isolation Forest on the custom 51k-row mixed-feature dataset."""
    df = load_custom()

    pre = CustomPreprocessor()
    x, y = pre.fit_transform(df)
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y if y.sum() > 1 else None
    )

    result = train_isolation_forest(x_tr, y_tr)
    y_pred, y_score = if_predict_fraud(result.model, x_te)

    if_metrics = evaluate_classifier(y_te, y_pred, y_score, label="isolation_forest")

    df_with_idx = df.reset_index(drop=True)
    _, idx_test = train_test_split(
        df_with_idx.index, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=y if y.sum() > 1 else None,
    )
    rule_pred = custom_rule_predict(df_with_idx.loc[idx_test])
    rule_metrics = evaluate_classifier(y_te, rule_pred, None, label="rule_based_custom")

    mcnemar_result = mcnemar_test(y_te, y_pred, rule_pred)

    _save(pre, ARTIFACTS["custom_preprocessor"])
    _save(result.model, ARTIFACTS["if_model"])

    return {
        "isolation_forest": if_metrics,
        "rule_based": rule_metrics,
        "mcnemar": mcnemar_result,
        "contamination": result.contamination,
        "n_train_legit_only": result.n_train,
    }


def train_all(sample_paysim: int | None = None) -> dict:
    metadata: dict = {}
    metadata["paysim"] = train_paysim_pipeline(sample_n=sample_paysim)
    metadata["custom"] = train_custom_pipeline()

    # Persist metadata.json for the HF Hub upload
    (MODELS_LOCAL_DIR / ARTIFACTS["metadata"]).write_text(
        json.dumps(metadata, indent=2, default=str)
    )
    return metadata
