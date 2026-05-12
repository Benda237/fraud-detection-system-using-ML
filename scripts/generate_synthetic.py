"""Generate synthetic stand-in datasets matching both schemas.

Useful when Kaggle download is unavailable or disk space is constrained.
Outputs are saved to data/raw/ under the same filenames the real pipeline
uses, so train_all.py works against them unchanged.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import CUSTOM_CSV, PAYSIM_CSV


def _make_paysim(n: int, fraud_rate: float = 0.0013, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    types = ["CASH_OUT", "PAYMENT", "TRANSFER", "CASH_IN", "DEBIT"]
    df = pd.DataFrame({
        "step": rng.integers(1, 744, n),
        "type": rng.choice(types, n, p=[0.30, 0.34, 0.06, 0.27, 0.03]),
        "amount": rng.exponential(scale=3000, size=n),
        "nameOrig": [f"C{rng.integers(1, 1_000_000)}" for _ in range(n)],
        "oldbalanceOrg": rng.exponential(scale=5000, size=n),
        "newbalanceOrig": rng.exponential(scale=4000, size=n),
        "nameDest": [f"M{rng.integers(1, 1_000_000)}" for _ in range(n)],
        "oldbalanceDest": rng.exponential(scale=2000, size=n),
        "newbalanceDest": rng.exponential(scale=2200, size=n),
        "isFlaggedFraud": 0,
    })

    fraud_mask = rng.random(n) < fraud_rate
    # Fraud bias — large amount, drained origin, TRANSFER/CASH_OUT types
    df.loc[fraud_mask, "type"] = rng.choice(["TRANSFER", "CASH_OUT"], fraud_mask.sum())
    df.loc[fraud_mask, "amount"] = rng.uniform(50_000, 500_000, fraud_mask.sum())
    df.loc[fraud_mask, "oldbalanceOrg"] = df.loc[fraud_mask, "amount"]
    df.loc[fraud_mask, "newbalanceOrig"] = 0.0
    df.loc[fraud_mask, "oldbalanceDest"] = 0.0
    df["isFraud"] = fraud_mask.astype(int)
    return df


def _make_custom(n: int, fraud_rate: float = 0.05, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Transaction_ID": [f"T{i+1}" for i in range(n)],
        "User_ID": rng.integers(1, 10_000, n),
        "Transaction_Amount": rng.exponential(scale=2500, size=n),
        "Transaction_Type": rng.choice(
            ["ATM Withdrawal", "Bill Payment", "Online Purchase",
             "POS Payment", "Bank Transfer"], n),
        "Time_of_Transaction": rng.uniform(0, 23, n).round(0),
        "Device_Used": rng.choice(
            ["Desktop", "Mobile", "Tablet", "Unknown Device"], n,
            p=[0.31, 0.31, 0.30, 0.08]),
        "Location": rng.choice(
            ["Boston", "Chicago", "Houston", "Los Angeles",
             "Miami", "New York", "San Francisco", "Seattle"], n),
        "Previous_Fraudulent_Transactions": rng.integers(0, 6, n),
        "Account_Age": rng.integers(1, 1500, n),
        "Number_of_Transactions_Last_24H": rng.integers(0, 25, n),
        "Payment_Method": rng.choice(
            ["Credit Card", "Debit Card", "Net Banking", "UPI", "Invalid Method"],
            n, p=[0.23, 0.24, 0.23, 0.23, 0.07]),
    })

    fraud_mask = rng.random(n) < fraud_rate
    # Add risk signals to the fraud subset
    df.loc[fraud_mask, "Previous_Fraudulent_Transactions"] = rng.integers(3, 8, fraud_mask.sum())
    df.loc[fraud_mask, "Account_Age"] = rng.integers(1, 30, fraud_mask.sum())
    df.loc[fraud_mask, "Transaction_Amount"] = rng.uniform(5_000, 20_000, fraud_mask.sum())
    df.loc[fraud_mask, "Payment_Method"] = rng.choice(
        ["Invalid Method", "Credit Card"], fraud_mask.sum(), p=[0.5, 0.5]
    )
    df["Fraudulent"] = fraud_mask.astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paysim-rows", type=int, default=200_000)
    parser.add_argument("--custom-rows", type=int, default=51_000)
    args = parser.parse_args()

    PAYSIM_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not PAYSIM_CSV.exists():
        print(f"Generating synthetic PaySim ({args.paysim_rows} rows)…")
        df = _make_paysim(args.paysim_rows)
        df.to_csv(PAYSIM_CSV, index=False)
        print(f"  saved -> {PAYSIM_CSV} ({PAYSIM_CSV.stat().st_size / 1e6:.1f} MB, "
              f"fraud rate {df['isFraud'].mean():.3%})")
    else:
        print(f"[skip] {PAYSIM_CSV.name} already exists")

    if not CUSTOM_CSV.exists():
        print(f"Generating synthetic custom dataset ({args.custom_rows} rows)…")
        df = _make_custom(args.custom_rows)
        df.to_csv(CUSTOM_CSV, index=False)
        print(f"  saved -> {CUSTOM_CSV} ({CUSTOM_CSV.stat().st_size / 1e6:.1f} MB, "
              f"fraud rate {df['Fraudulent'].mean():.3%})")
    else:
        print(f"[skip] {CUSTOM_CSV.name} already exists")


if __name__ == "__main__":
    main()
