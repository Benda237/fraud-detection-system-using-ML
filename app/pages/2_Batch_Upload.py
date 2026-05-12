"""Page 2 — upload a CSV, score every row, download results."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app.components import kpi, page_setup, section_title
from app.components.sidebar import render_sidebar
from src.inference.predictor import predict_custom, predict_paysim

page_setup("Batch Upload", icon="📦")
choices = render_sidebar()
threshold = choices["threshold"]

st.title("📦 Batch Transaction Scoring")
st.caption(
    "Upload a CSV with transactions matching either PaySim or the custom schema. "
    "Every row is scored and a fraud column is appended for download."
)

schema = st.radio(
    "Schema of the file you'll upload",
    ["PaySim (step, type, amount, balances)", "Custom (device, location, etc.)"],
    horizontal=True,
)

up = st.file_uploader("CSV file", type=["csv"], accept_multiple_files=False)

if up is None:
    st.info("⬆️ Drop a CSV above to start.")
    st.markdown(
        """
        **Expected columns:**
        - PaySim: `step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
          nameDest, oldbalanceDest, newbalanceDest, isFlaggedFraud`
        - Custom: `Transaction_ID, User_ID, Transaction_Amount, Transaction_Type,
          Time_of_Transaction, Device_Used, Location, Previous_Fraudulent_Transactions,
          Account_Age, Number_of_Transactions_Last_24H, Payment_Method`
        """
    )
    st.stop()

df = pd.read_csv(up)
st.success(f"Loaded {len(df):,} rows · {df.shape[1]} columns")

with st.expander("Preview (first 50 rows)"):
    st.dataframe(df.head(50), use_container_width=True)

if st.button("🚀 Score batch", type="primary", use_container_width=True):
    with st.spinner(f"Scoring {len(df):,} transactions…"):
        try:
            if schema.startswith("PaySim"):
                out = predict_paysim(df, threshold=threshold)
                df["fraud_probability"] = out["fraud_probability"]
                df["prediction"] = out["prediction"]
            else:
                out = predict_custom(df)
                df["anomaly_score"] = out["anomaly_score"]
                df["prediction"] = out["prediction"]
        except Exception as exc:  # noqa: BLE001
            st.error(f"Scoring failed: {exc}")
            st.stop()

    section_title("Results")
    flagged = int(df["prediction"].sum())
    rate = flagged / len(df) if len(df) else 0.0
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(kpi("Transactions scored", f"{len(df):,}"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("Flagged as fraud", f"{flagged:,}"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("Flag rate", f"{rate:.2%}"), unsafe_allow_html=True)

    st.dataframe(df.head(200), use_container_width=True, height=420)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download full results as CSV",
        data=csv_bytes,
        file_name=f"fraudguard_scored_{schema.split()[0].lower()}.csv",
        mime="text/csv",
        use_container_width=True,
    )
