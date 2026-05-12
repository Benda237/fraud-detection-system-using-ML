"""Smoke test: pull both models from HF Hub and run a single prediction
to verify the artifacts are loadable and the API works end-to-end."""
from __future__ import annotations

import sys

from src.inference.predictor import predict_custom, predict_paysim


def main() -> int:
    print("Pulling models from Hugging Face Hub…")

    paysim_sample = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 181.0,
        "nameOrig": "C1305486145",
        "oldbalanceOrg": 181.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C553264065",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "isFlaggedFraud": 0,
    }
    custom_sample = {
        "Transaction_ID": "T_TEST",
        "User_ID": 1234,
        "Transaction_Amount": 4500.0,
        "Transaction_Type": "ATM Withdrawal",
        "Time_of_Transaction": 23.0,
        "Device_Used": "Mobile",
        "Location": "New York",
        "Previous_Fraudulent_Transactions": 3,
        "Account_Age": 5,
        "Number_of_Transactions_Last_24H": 12,
        "Payment_Method": "Credit Card",
    }

    print("PaySim sample (known-fraud-like):")
    print(" ", predict_paysim(paysim_sample))

    print("Custom sample (high-risk-like):")
    print(" ", predict_custom(custom_sample))
    return 0


if __name__ == "__main__":
    sys.exit(main())
