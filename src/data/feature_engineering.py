"""Standalone feature-engineering helpers reused by EDA notebooks and the GUI."""
from __future__ import annotations

import numpy as np
import pandas as pd


def paysim_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns to a PaySim DataFrame. Pure, non-mutating."""
    out = df.copy()
    out["balance_delta_orig"] = out["oldbalanceOrg"] - out["newbalanceOrig"]
    out["balance_delta_dest"] = out["newbalanceDest"] - out["oldbalanceDest"]
    out["amount_to_balance_ratio"] = np.where(
        out["oldbalanceOrg"] > 0,
        out["amount"] / out["oldbalanceOrg"],
        0.0,
    )
    out["amount_equals_orig_balance"] = (
        np.isclose(out["amount"], out["oldbalanceOrg"], atol=1.0)
    ).astype(int)
    out["dest_was_empty"] = (out["oldbalanceDest"] == 0).astype(int)
    return out


def custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns to the custom dataset."""
    out = df.copy()
    if "Time_of_Transaction" in out.columns:
        out["is_night"] = ((out["Time_of_Transaction"] < 6) | (out["Time_of_Transaction"] > 22)).astype(int)
    if "Account_Age" in out.columns:
        out["is_new_account"] = (out["Account_Age"] < 30).astype(int)
    if {"Transaction_Amount", "Number_of_Transactions_Last_24H"}.issubset(out.columns):
        out["avg_amount_per_recent_tx"] = np.where(
            out["Number_of_Transactions_Last_24H"] > 0,
            out["Transaction_Amount"] / out["Number_of_Transactions_Last_24H"],
            out["Transaction_Amount"],
        )
    return out
