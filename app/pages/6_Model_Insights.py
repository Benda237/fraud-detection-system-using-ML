"""Page 6 — feature importance & model insights."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.components import kpi, page_setup, section_title
from app.components.charts import feature_importance_bar
from app.components.sidebar import render_sidebar
from src.inference.predictor import get_metadata

page_setup("Model Insights", icon="🧠")
render_sidebar()

st.title("🧠 Model Insights")
st.caption("How the deployed models reach their verdicts.")

metadata = get_metadata()
if not metadata:
    st.warning("No metadata yet. Train and push first with `scripts/train_all.py`.")
    st.stop()

paysim = metadata.get("paysim", {})
custom = metadata.get("custom", {})

section_title("Class-imbalance handling — Random Forest (PaySim)")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(kpi("Fraud rate BEFORE SMOTE",
                    f"{paysim.get('fraud_rate_before_smote', 0):.3%}"),
                unsafe_allow_html=True)
with c2:
    st.markdown(kpi("Fraud rate AFTER SMOTE",
                    f"{paysim.get('fraud_rate_after_smote', 0):.3%}"),
                unsafe_allow_html=True)
with c3:
    st.markdown(kpi("Train rows after SMOTE",
                    f"{paysim.get('n_train_after_smote', 0):,}"),
                unsafe_allow_html=True)
st.markdown(
    """
    Fraud transactions in PaySim are ~0.13% of all transactions. We use
    **SMOTE** (Synthetic Minority Over-sampling Technique) on the training fold
    only, plus `class_weight='balanced'` inside the RandomForest. This
    addresses the imbalance specific objective in the research framework.
    """
)

section_title("Feature importance — Random Forest")
fi = paysim.get("feature_importances")
if fi:
    st.plotly_chart(feature_importance_bar(fi), use_container_width=True)
    st.markdown(
        "The most predictive features answer the study's specific research question: "
        "_What features have the strongest predictive power for fraud?_"
    )
else:
    st.info("Feature importances not present in metadata.")

section_title("Isolation Forest configuration (Custom dataset)")
c1, c2 = st.columns(2)
with c1:
    st.markdown(kpi("Contamination",
                    f"{custom.get('contamination', 0):.4f}"),
                unsafe_allow_html=True)
with c2:
    st.markdown(kpi("Trained on (legit-only)",
                    f"{custom.get('n_train_legit_only', 0):,}"),
                unsafe_allow_html=True)
st.markdown(
    """
    Isolation Forest is **unsupervised** — fit on the legitimate-only majority.
    Fraud is detected as anomalies. The `contamination` hyperparameter is
    inferred from the observed fraud rate so the decision threshold is
    calibrated to the data.
    """
)

with st.expander("Raw metadata.json"):
    st.json(metadata)
