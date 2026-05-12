"""FraudGuard — Streamlit entry point.

Run with: streamlit run app/Home.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when running via Streamlit
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.components import kpi, page_setup, section_title
from app.components.sidebar import render_sidebar

page_setup("Home", icon="🛡️")
render_sidebar()

# Hero
st.markdown(
    """
    <div style='padding:30px 36px;border-radius:18px;
                background:linear-gradient(135deg,#0c1428 0%,#1a2541 100%);
                border:1px solid #1f2937;margin-bottom:20px;'>
        <div style='color:#22d3ee;font-size:0.9rem;letter-spacing:0.18em;
                    text-transform:uppercase;margin-bottom:8px;'>
            ML-FDS · Banking & Financial Sector
        </div>
        <div style='color:#f8fafc;font-size:2.4rem;font-weight:700;line-height:1.1;'>
            🛡️ FraudGuard
        </div>
        <div style='color:#94a3b8;font-size:1.1rem;margin-top:10px;max-width:780px;'>
            A Machine Learning–based Fraud Detection System for commercial banks
            and credit unions. Random Forest + Isolation Forest models trained on
            real-world transaction datasets, hosted on Hugging Face Hub, with
            real-time scoring and a full evaluation dashboard.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# KPI row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(kpi("Models deployed", "2",
                    "Random Forest · Isolation Forest"), unsafe_allow_html=True)
with c2:
    st.markdown(kpi("Datasets", "2",
                    "PaySim 6.3M + Custom 51k"), unsafe_allow_html=True)
with c3:
    st.markdown(kpi("Imbalance handling", "SMOTE",
                    "Plus class_weight=balanced"), unsafe_allow_html=True)
with c4:
    st.markdown(kpi("Hosted on", "🤗 HF Hub",
                    "Pulled lazily, cached locally"), unsafe_allow_html=True)

st.write("")

# Pages overview
section_title("System overview")
st.markdown(
    """
**FraudGuard** addresses the problem outlined in the study:
> Commercial banks and credit unions in Cameroon face sophisticated, adaptive
> financial fraud. Rule-based systems produce too many false positives and miss
> novel patterns.

This system implements the full pipeline:
1. **Data preprocessing** — null handling, encoding, scaling, feature engineering
2. **Class-imbalance handling** — SMOTE oversampling + class-weighted RandomForest
3. **Comparative analysis** — RandomForest vs IsolationForest vs rule-based baseline
4. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC, McNemar's hypothesis test
5. **Deployment** — models published to Hugging Face Hub for secure, versioned access
    """
)

section_title("Use the system")
nav_c1, nav_c2 = st.columns(2)
with nav_c1:
    st.markdown(
        """
        - 📝 **Single Transaction** — Test a single transaction in real time
        - 📦 **Batch Upload** — Score a whole CSV file at once
        - 📊 **Model Comparison** — RF vs IF vs rule-based on every metric
        """
    )
with nav_c2:
    st.markdown(
        """
        - 🔍 **Data Explorer** — Interactive EDA of both datasets
        - 📡 **Live Monitor** — Simulated streaming fraud dashboard
        - 🧠 **Model Insights** — Feature importance, decision logic
        - ⚙️ **Settings** — Threshold, HF token, cache controls
        """
    )

st.write("")
section_title("Research framing")
with st.expander("Research questions, objectives, hypotheses", expanded=False):
    st.markdown(
        """
**Main research question.** How can a Machine Learning-based model be effectively
designed and implemented to significantly improve the accuracy and efficiency for
financial fraud detection in commercial banks and Financial Institutions?

**Specific objectives implemented in this system:**
- Comparative analysis of Random Forest and Isolation Forest
- Pre-processing and feature engineering to mitigate class imbalance
- Training on appropriate datasets (PaySim + custom mixed-feature)
- Evaluation against a rule-based baseline using Accuracy, Precision, Recall, F1
- Architectural framework: HF Hub for model artifacts, Streamlit for the UI

**Hypotheses (tested on the Model Comparison page):**
- **H1 (null):** ML-FDS does not significantly improve accuracy.
- **HA:** ML-FDS significantly improves accuracy.

McNemar's test on the test split decides between H1 and HA.
        """
    )

st.caption("⬅ Use the sidebar to switch pages.")
