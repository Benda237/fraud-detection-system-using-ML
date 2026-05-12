"""Rule-based baseline fraud detector for comparative analysis.

Mirrors the kind of threshold heuristic many banks historically use, so the
ML models can be benchmarked against a realistic non-ML alternative
(research objective 4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RULE_BALANCE_DRAINED_TOLERANCE, RULE_LARGE_AMOUNT


def paysim_rule_predict(df: pd.DataFrame) -> np.ndarray:
    """Flag a PaySim transaction as fraud if ANY of these triggers fire:
    - amount > RULE_LARGE_AMOUNT
    - the originating balance is drained to zero (amount == oldbalanceOrg)
    - isFlaggedFraud == 1
    - TRANSFER followed by CASH_OUT pattern (approximated by type alone here)
    """
    large = df["amount"] > RULE_LARGE_AMOUNT
    drained = np.isclose(df["amount"], df["oldbalanceOrg"], atol=RULE_BALANCE_DRAINED_TOLERANCE)
    flagged = df.get("isFlaggedFraud", pd.Series(0, index=df.index)).astype(bool)
    risky_type = df["type"].isin(["TRANSFER", "CASH_OUT"])
    return (large & risky_type | drained & risky_type | flagged).astype(int).to_numpy()


def custom_rule_predict(df: pd.DataFrame) -> np.ndarray:
    """Rules for the custom dataset:
    - Previous_Fraudulent_Transactions >= 3
    - new account (<30 days) with large amount (>5000)
    - Payment_Method == 'Invalid Method'
    """
    prev = df.get("Previous_Fraudulent_Transactions", pd.Series(0, index=df.index)) >= 3
    new_big = (
        (df.get("Account_Age", pd.Series(999, index=df.index)) < 30)
        & (df.get("Transaction_Amount", pd.Series(0, index=df.index)) > 5000)
    )
    invalid = df.get("Payment_Method", pd.Series("", index=df.index)) == "Invalid Method"
    return (prev | new_big | invalid).astype(int).to_numpy()
