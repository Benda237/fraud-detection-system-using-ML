"""Page 1 — score a single transaction with whichever schema is selected."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.components import page_setup, section_title, verdict_banner
from app.components.charts import gauge
from app.components.sidebar import render_sidebar
from app.components.transaction_form import custom_form, paysim_form
from src.inference.predictor import predict_custom, predict_paysim
from src.models.rule_based_baseline import (
    custom_rule_predict,
    paysim_rule_predict,
)

import pandas as pd

page_setup("Single Transaction", icon="📝")
choices = render_sidebar()
threshold = choices["threshold"]

st.title("📝 Single Transaction Scoring")
st.caption(
    "Fill in the form and click **Score transaction**. The system pulls the "
    "trained model from Hugging Face Hub (cached locally) and returns a "
    "verdict, probability, and rule-based comparison."
)

schema = st.radio(
    "Schema", ["PaySim (banking transfer)", "Custom (device + location)"],
    horizontal=True,
)

st.markdown("---")

if schema.startswith("PaySim"):
    section_title("Transaction details")
    payload = paysim_form()
    if st.button("🚀 Score transaction", type="primary", use_container_width=True):
        with st.spinner("Pulling model from Hugging Face Hub…"):
            try:
                rf_out = predict_paysim(payload, threshold=threshold)
                rule_pred = paysim_rule_predict(pd.DataFrame([payload]))[0]
            except Exception as exc:  # noqa: BLE001
                st.error(f"Prediction failed: {exc}")
                st.info(
                    "If this is the first run, train the models and push them "
                    "to HF Hub via `python scripts/train_all.py`."
                )
                st.stop()
        prob = rf_out["fraud_probability"][0]
        pred = rf_out["prediction"][0]
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(
                verdict_banner(bool(pred), prob, "Random Forest"),
                unsafe_allow_html=True,
            )
            rule_label = "🚨 FRAUD" if rule_pred else "✅ LEGIT"
            st.info(f"**Rule-based baseline verdict:** {rule_label}")
        with c2:
            st.plotly_chart(gauge(prob, "Fraud risk"), use_container_width=True)
        with st.expander("Raw prediction payload"):
            st.json(rf_out)
else:
    section_title("Transaction details")
    payload = custom_form()
    if st.button("🚀 Score transaction", type="primary", use_container_width=True):
        with st.spinner("Pulling Isolation Forest from Hugging Face Hub…"):
            try:
                if_out = predict_custom(payload)
                rule_pred = custom_rule_predict(pd.DataFrame([payload]))[0]
            except Exception as exc:  # noqa: BLE001
                st.error(f"Prediction failed: {exc}")
                st.stop()
        score = if_out["anomaly_score"][0]
        pred = if_out["prediction"][0]
        # Normalize score into [0,1] for the gauge — anomaly scores vary by training
        norm = float(max(0.0, min(1.0, (score + 0.5))))
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(
                verdict_banner(bool(pred), norm, "Isolation Forest"),
                unsafe_allow_html=True,
            )
            rule_label = "🚨 FRAUD" if rule_pred else "✅ LEGIT"
            st.info(f"**Rule-based baseline verdict:** {rule_label}")
        with c2:
            st.plotly_chart(gauge(norm, "Anomaly score"), use_container_width=True)
        with st.expander("Raw prediction payload"):
            st.json(if_out)
