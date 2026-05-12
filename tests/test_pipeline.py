"""Smoke tests — verify the modules wire together correctly even without
the full datasets present. Anything dataset-dependent is skipped if the
CSV is missing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import CUSTOM_CSV, PAYSIM_CSV
from src.data.feature_engineering import custom_features, paysim_features
from src.data.preprocessor import CustomPreprocessor, PaySimPreprocessor
from src.evaluation.hypothesis_test import mcnemar_test
from src.evaluation.metrics import evaluate_classifier
from src.models.rule_based_baseline import (
    custom_rule_predict,
    paysim_rule_predict,
)


def _paysim_synth(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "step": rng.integers(1, 50, n),
        "type": rng.choice(["CASH_OUT", "PAYMENT", "TRANSFER", "DEBIT", "CASH_IN"], n),
        "amount": rng.exponential(2000, n),
        "nameOrig": [f"C{i}" for i in range(n)],
        "oldbalanceOrg": rng.exponential(5000, n),
        "newbalanceOrig": rng.exponential(4000, n),
        "nameDest": [f"M{i}" for i in range(n)],
        "oldbalanceDest": rng.exponential(2000, n),
        "newbalanceDest": rng.exponential(2500, n),
        "isFlaggedFraud": rng.integers(0, 2, n),
        "isFraud": rng.choice([0, 0, 0, 0, 1], n),
    })


def _custom_synth(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.DataFrame({
        "Transaction_ID": [f"T{i}" for i in range(n)],
        "User_ID": rng.integers(1, 9999, n),
        "Transaction_Amount": rng.exponential(2500, n),
        "Transaction_Type": rng.choice(["ATM Withdrawal", "Bill Payment", "POS Payment"], n),
        "Time_of_Transaction": rng.uniform(0, 23, n),
        "Device_Used": rng.choice(["Mobile", "Desktop", "Tablet"], n),
        "Location": rng.choice(["New York", "Chicago", "Miami"], n),
        "Previous_Fraudulent_Transactions": rng.integers(0, 5, n),
        "Account_Age": rng.integers(1, 1000, n),
        "Number_of_Transactions_Last_24H": rng.integers(0, 30, n),
        "Payment_Method": rng.choice(["Credit Card", "Debit Card", "UPI"], n),
        "Fraudulent": rng.choice([0, 0, 0, 0, 1], n),
    })


def test_paysim_preprocessor_roundtrip():
    df = _paysim_synth()
    pre = PaySimPreprocessor()
    x, y = pre.fit_transform(df)
    assert x.shape[0] == len(df)
    assert x.shape[1] == len(pre.feature_cols)
    x2 = pre.transform(df.head(5))
    assert x2.shape == (5, x.shape[1])


def test_custom_preprocessor_roundtrip():
    df = _custom_synth()
    pre = CustomPreprocessor()
    x, y = pre.fit_transform(df)
    assert x.shape[0] == len(df)
    x2 = pre.transform(df.head(5))
    assert x2.shape == (5, x.shape[1])


def test_paysim_features_pure():
    df = _paysim_synth()
    before = df.shape
    out = paysim_features(df)
    assert df.shape == before  # not mutated
    assert "balance_delta_orig" in out.columns


def test_custom_features_pure():
    df = _custom_synth()
    out = custom_features(df)
    assert "is_night" in out.columns


def test_rule_based_baselines_run():
    paysim_rule_predict(_paysim_synth())
    custom_rule_predict(_custom_synth())


def test_metrics_shape():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 1])
    metrics = evaluate_classifier(y_true, y_pred, np.array([.1, .8, .9, .3, .2, .7, .1, .95]))
    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["true_positives"] == 3


def test_mcnemar_runs():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    a = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
    b = np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0])
    res = mcnemar_test(y_true, a, b)
    assert "p_value" in res
    assert 0 <= res["p_value"] <= 1


@pytest.mark.skipif(not PAYSIM_CSV.exists(), reason="PaySim CSV not downloaded")
def test_real_paysim_load():
    from src.data.loader import load_paysim
    df = load_paysim()
    assert len(df) > 0


@pytest.mark.skipif(not CUSTOM_CSV.exists(), reason="Custom CSV not downloaded")
def test_real_custom_load():
    from src.data.loader import load_custom
    df = load_custom()
    assert len(df) > 0
