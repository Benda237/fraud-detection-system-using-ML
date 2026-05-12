"""Input forms for the two supported transaction schemas."""
from __future__ import annotations

import streamlit as st


PAYSIM_TYPES = ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"]

CUSTOM_TX_TYPES = ["ATM Withdrawal", "Bill Payment", "Online Purchase",
                   "POS Payment", "Bank Transfer"]
CUSTOM_DEVICES = ["Desktop", "Mobile", "Tablet", "Unknown Device"]
CUSTOM_LOCATIONS = ["Boston", "Chicago", "Houston", "Los Angeles", "Miami",
                    "New York", "San Francisco", "Seattle"]
CUSTOM_PAYMENT_METHODS = ["Credit Card", "Debit Card", "Net Banking", "UPI",
                          "Invalid Method"]


def paysim_form(prefix: str = "ps") -> dict:
    """Renders the PaySim form and returns a transaction dict."""
    c1, c2 = st.columns(2)
    with c1:
        step = st.number_input("Step (hour of simulation)", 1, 744, 1, key=f"{prefix}_step")
        tx_type = st.selectbox("Transaction type", PAYSIM_TYPES, key=f"{prefix}_type")
        amount = st.number_input("Amount", 0.0, 1_000_000.0, 1000.0, step=100.0, key=f"{prefix}_amount")
        name_orig = st.text_input("Origin account ID", "C1234567", key=f"{prefix}_no")
        old_org = st.number_input("Origin balance (before)", 0.0, 1e9, 5000.0, key=f"{prefix}_oo")
        new_org = st.number_input("Origin balance (after)", 0.0, 1e9, 4000.0, key=f"{prefix}_no2")
    with c2:
        name_dest = st.text_input("Destination account ID", "C7654321", key=f"{prefix}_nd")
        old_dest = st.number_input("Dest balance (before)", 0.0, 1e9, 0.0, key=f"{prefix}_od")
        new_dest = st.number_input("Dest balance (after)", 0.0, 1e9, 1000.0, key=f"{prefix}_nd2")
        flagged = st.selectbox("Pre-flagged by bank?", [0, 1], key=f"{prefix}_fl")

    return {
        "step": int(step),
        "type": tx_type,
        "amount": float(amount),
        "nameOrig": name_orig,
        "oldbalanceOrg": float(old_org),
        "newbalanceOrig": float(new_org),
        "nameDest": name_dest,
        "oldbalanceDest": float(old_dest),
        "newbalanceDest": float(new_dest),
        "isFlaggedFraud": int(flagged),
    }


def custom_form(prefix: str = "cu") -> dict:
    """Renders the custom (device/location/method) form."""
    c1, c2 = st.columns(2)
    with c1:
        user_id = st.number_input("User ID", 1, 99_999, 1234, key=f"{prefix}_uid")
        tx_id = st.text_input("Transaction ID", "T_NEW_0001", key=f"{prefix}_tid")
        amount = st.number_input("Transaction amount", 0.0, 100_000.0, 2500.0,
                                 step=50.0, key=f"{prefix}_amt")
        tx_type = st.selectbox("Transaction type", CUSTOM_TX_TYPES, key=f"{prefix}_typ")
        tod = st.slider("Time of day (hour)", 0, 23, 14, key=f"{prefix}_tod")
        device = st.selectbox("Device used", CUSTOM_DEVICES, key=f"{prefix}_dev")
    with c2:
        location = st.selectbox("Location", CUSTOM_LOCATIONS, key=f"{prefix}_loc")
        prev_fraud = st.number_input("Previous fraudulent transactions",
                                     0, 50, 0, key=f"{prefix}_prev")
        account_age = st.number_input("Account age (days)",
                                      0, 5000, 120, key=f"{prefix}_age")
        n_24h = st.number_input("Transactions in last 24h",
                                0, 200, 4, key=f"{prefix}_24h")
        payment = st.selectbox("Payment method", CUSTOM_PAYMENT_METHODS,
                               key=f"{prefix}_pay")

    return {
        "Transaction_ID": tx_id,
        "User_ID": int(user_id),
        "Transaction_Amount": float(amount),
        "Transaction_Type": tx_type,
        "Time_of_Transaction": float(tod),
        "Device_Used": device,
        "Location": location,
        "Previous_Fraudulent_Transactions": int(prev_fraud),
        "Account_Age": int(account_age),
        "Number_of_Transactions_Last_24H": int(n_24h),
        "Payment_Method": payment,
    }
