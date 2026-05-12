"""Page 3 — model performance comparison + hypothesis test."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from app.components import kpi, page_setup, section_title
from app.components.charts import (
    confusion_matrix_heatmap,
    metric_comparison_bar,
    roc_curve_fig,
)
from app.components.sidebar import render_sidebar
from src.hf_hub.downloader import get_metadata

page_setup("Model Comparison", icon="📊")
render_sidebar()

st.title("📊 Model Comparison")
st.caption(
    "Random Forest vs Isolation Forest vs rule-based baseline. "
    "Metrics are loaded from the metadata.json artifact on Hugging Face Hub."
)

metadata = get_metadata()
if not metadata:
    st.warning(
        "No training metadata found on HF Hub yet. "
        "Run `python scripts/train_all.py` to train models and push artifacts."
    )
    st.stop()

paysim = metadata.get("paysim", {})
custom = metadata.get("custom", {})

rf = paysim.get("rf", {})
rf_baseline = paysim.get("rule_based", {})
iso = custom.get("isolation_forest", {})
iso_baseline = custom.get("rule_based", {})

section_title("Headline metrics")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(kpi("RF Accuracy", f"{rf.get('accuracy', 0):.3f}",
                    f"baseline {rf_baseline.get('accuracy', 0):.3f}"),
                unsafe_allow_html=True)
with c2:
    st.markdown(kpi("RF F1", f"{rf.get('f1', 0):.3f}",
                    f"baseline {rf_baseline.get('f1', 0):.3f}"),
                unsafe_allow_html=True)
with c3:
    st.markdown(kpi("IF Accuracy", f"{iso.get('accuracy', 0):.3f}",
                    f"baseline {iso_baseline.get('accuracy', 0):.3f}"),
                unsafe_allow_html=True)
with c4:
    st.markdown(kpi("IF Recall", f"{iso.get('recall', 0):.3f}",
                    f"baseline {iso_baseline.get('recall', 0):.3f}"),
                unsafe_allow_html=True)

# ===== Bar chart =====
rows = []
for label, m in [
    ("RandomForest (PaySim)", rf),
    ("Rule-based (PaySim)", rf_baseline),
    ("IsolationForest (Custom)", iso),
    ("Rule-based (Custom)", iso_baseline),
]:
    rows.append({
        "model": label,
        "accuracy": m.get("accuracy", 0),
        "precision": m.get("precision", 0),
        "recall": m.get("recall", 0),
        "f1": m.get("f1", 0),
    })
st.plotly_chart(metric_comparison_bar(rows), use_container_width=True)

# ===== ROC curves =====
section_title("ROC curves")
curves = {}
if "roc_curve" in rf:
    curves["RandomForest"] = rf["roc_curve"]
if "roc_curve" in iso:
    curves["IsolationForest"] = iso["roc_curve"]
if curves:
    st.plotly_chart(roc_curve_fig(curves), use_container_width=True)
else:
    st.info("No ROC curve data was stored in metadata.")

# ===== Confusion matrices =====
section_title("Confusion matrices")
cmc1, cmc2 = st.columns(2)
with cmc1:
    st.subheader("Random Forest")
    cm = np.array([
        [rf.get("true_negatives", 0), rf.get("false_positives", 0)],
        [rf.get("false_negatives", 0), rf.get("true_positives", 0)],
    ])
    st.plotly_chart(confusion_matrix_heatmap(cm), use_container_width=True)
with cmc2:
    st.subheader("Isolation Forest")
    cm2 = np.array([
        [iso.get("true_negatives", 0), iso.get("false_positives", 0)],
        [iso.get("false_negatives", 0), iso.get("true_positives", 0)],
    ])
    st.plotly_chart(confusion_matrix_heatmap(cm2), use_container_width=True)

# ===== Hypothesis test summary =====
section_title("Hypothesis test (H1 vs HA)")
st.markdown(
    """
- **H1 (null):** ML-FDS does not significantly improve accuracy.
- **HA:** ML-FDS significantly improves accuracy.

McNemar's test on the held-out test set compares each ML model against its
rule-based baseline.
"""
)
mc_rf = paysim.get("mcnemar")
mc_if = custom.get("mcnemar")
mcc1, mcc2 = st.columns(2)
with mcc1:
    st.markdown("**RandomForest vs Rule-based (PaySim)**")
    if mc_rf:
        st.json(mc_rf)
    else:
        st.info(
            "Hypothesis-test results are computed during training. "
            "Re-run `scripts/train_all.py` after the predictor is wired in."
        )
with mcc2:
    st.markdown("**IsolationForest vs Rule-based (Custom)**")
    if mc_if:
        st.json(mc_if)
    else:
        st.info("Same — re-run training to populate.")

with st.expander("Raw metadata.json"):
    st.json(metadata)
