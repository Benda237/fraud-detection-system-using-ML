"""Page 4 — interactive EDA of either dataset."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.components import kpi, page_setup, section_title
from app.components.charts import amount_distribution
from app.components.sidebar import render_sidebar
from src.config import CUSTOM_CSV, PAYSIM_CSV
from src.data.loader import load_custom, load_paysim

page_setup("Data Explorer", icon="🔍")
render_sidebar()

st.title("🔍 Data Explorer")
st.caption(
    "Interactive EDA of the raw datasets. "
    "Filter, slice, and visualize before training."
)

dataset = st.radio(
    "Dataset",
    ["PaySim (6.3M rows)", "Custom (51k rows)"],
    horizontal=True,
)


@st.cache_data(show_spinner="Loading dataset…")
def _load_paysim_cached() -> pd.DataFrame:
    return load_paysim().sample(n=200_000, random_state=42)  # cap for snappy UI


@st.cache_data(show_spinner="Loading dataset…")
def _load_custom_cached() -> pd.DataFrame:
    return load_custom()


if dataset.startswith("PaySim"):
    if not PAYSIM_CSV.exists():
        st.error(
            "PaySim CSV not found. Run `python scripts/download_data.py --paysim` "
            "or place the file at `data/raw/AIML Dataset.csv`."
        )
        st.stop()
    df = _load_paysim_cached()
    target = "isFraud"
    amount_col = "amount"
else:
    if not CUSTOM_CSV.exists():
        st.error(
            "Custom CSV not found. Run `python scripts/download_data.py --custom` "
            "or place the file at `data/raw/Fraud Detection Dataset.csv`."
        )
        st.stop()
    df = _load_custom_cached()
    target = "Fraudulent"
    amount_col = "Transaction_Amount"

section_title("Overview")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(kpi("Rows", f"{len(df):,}"), unsafe_allow_html=True)
with c2:
    st.markdown(kpi("Columns", f"{df.shape[1]}"), unsafe_allow_html=True)
with c3:
    fraud_rate = float(df[target].mean()) if target in df.columns else 0.0
    st.markdown(kpi("Fraud rate", f"{fraud_rate:.3%}"), unsafe_allow_html=True)
with c4:
    nulls = int(df.isna().sum().sum())
    st.markdown(kpi("Total nulls", f"{nulls:,}"), unsafe_allow_html=True)

section_title("Sample")
st.dataframe(df.head(100), use_container_width=True, height=320)

section_title("Distributions")
tab1, tab2, tab3 = st.tabs(["Amount", "Numeric features", "Categorical features"])

with tab1:
    if amount_col in df.columns and target in df.columns:
        st.plotly_chart(amount_distribution(df, amount_col, target), use_container_width=True)
    else:
        st.info("Required columns missing.")

with tab2:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    sel = st.selectbox("Choose numeric feature", num_cols)
    fig = px.histogram(
        df, x=sel, color=target if target in df.columns else None,
        nbins=50, log_y=True, template="plotly_dark", opacity=0.8,
        color_discrete_map={0: "#22d3ee", 1: "#f43f5e"},
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if not cat_cols:
        st.info("No categorical columns in this dataset.")
    else:
        sel = st.selectbox("Choose categorical feature", cat_cols)
        counts = df.groupby([sel, target]).size().reset_index(name="count")
        fig = px.bar(
            counts, x=sel, y="count", color=target, barmode="group",
            template="plotly_dark",
            color_discrete_map={0: "#22d3ee", 1: "#f43f5e"},
        )
        st.plotly_chart(fig, use_container_width=True)

section_title("Correlation matrix")
num_only = df.select_dtypes(include="number")
if len(num_only.columns) >= 2:
    corr = num_only.corr()
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        template="plotly_dark",
    )
    fig.update_layout(height=520, margin=dict(l=30, r=30, t=30, b=30))
    st.plotly_chart(fig, use_container_width=True)
