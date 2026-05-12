"""Page 5 — simulated streaming fraud monitor.

Sources rows from data/raw/AIML Dataset.csv at a user-controlled rate, scores
them with the deployed RF model, and updates a live dashboard.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.components import kpi, page_setup, section_title
from app.components.sidebar import render_sidebar
from src.config import PAYSIM_CSV
from src.inference.predictor import predict_paysim

page_setup("Live Monitor", icon="📡")
choices = render_sidebar()
threshold = choices["threshold"]

st.title("📡 Live Fraud Monitor")
st.caption(
    "Simulated stream of transactions through the deployed RF model. "
    "Adjust the rate, watch the alert log and KPIs update in real time."
)

if not PAYSIM_CSV.exists():
    st.error(
        f"PaySim CSV missing at {PAYSIM_CSV}. "
        "Run `python scripts/download_data.py --paysim`."
    )
    st.stop()


@st.cache_data(show_spinner="Sampling PaySim for stream…")
def _sample_stream(n: int = 500) -> pd.DataFrame:
    df = pd.read_csv(PAYSIM_CSV)
    # Bias the sample toward including some fraud so the UI is interesting
    fraud = df[df["isFraud"] == 1].sample(n=min(50, len(df[df["isFraud"] == 1])), random_state=1)
    legit = df[df["isFraud"] == 0].sample(n=n - len(fraud), random_state=1)
    out = pd.concat([fraud, legit]).sample(frac=1, random_state=1).reset_index(drop=True)
    return out


c1, c2, c3 = st.columns(3)
with c1:
    n_stream = st.number_input("Stream length (rows)", 50, 2000, 300, step=50)
with c2:
    rate = st.slider("Speed (rows / sec)", 1, 20, 5)
with c3:
    start = st.button("▶ Start stream", type="primary", use_container_width=True)

placeholder_kpis = st.empty()
placeholder_chart = st.empty()
placeholder_log = st.empty()

if start:
    stream = _sample_stream(int(n_stream))
    fraud_count = 0
    log_rows: list[dict] = []
    hist: list[dict] = []

    progress = st.progress(0.0, text="Streaming…")
    for i, row in stream.iterrows():
        payload = row.drop(labels=["isFraud"]).to_dict()
        try:
            res = predict_paysim(payload, threshold=threshold)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Live scoring failed at row {i}: {exc}")
            break
        prob = res["fraud_probability"][0]
        pred = res["prediction"][0]
        true = int(row["isFraud"])
        fraud_count += int(pred)
        hist.append({"row": i, "prob": prob, "pred": pred, "true": true})
        if pred:
            log_rows.append({
                "row": int(i),
                "type": row["type"],
                "amount": float(row["amount"]),
                "prob": round(float(prob), 3),
                "actual": "FRAUD" if true else "LEGIT",
            })

        # Update KPIs
        seen = i + 1
        with placeholder_kpis.container():
            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(kpi("Processed", f"{seen}"), unsafe_allow_html=True)
            k2.markdown(kpi("Flagged", f"{fraud_count}"), unsafe_allow_html=True)
            flag_rate = fraud_count / seen if seen else 0
            k3.markdown(kpi("Flag rate", f"{flag_rate:.1%}"), unsafe_allow_html=True)
            k4.markdown(kpi("Threshold", f"{threshold:.2f}"), unsafe_allow_html=True)

        # Update probability trace
        h = pd.DataFrame(hist)
        fig = px.line(
            h, x="row", y="prob", template="plotly_dark",
            title="Fraud probability per transaction",
        )
        fig.add_hline(y=threshold, line_dash="dash", line_color="#f59e0b",
                      annotation_text="threshold", annotation_position="top right")
        fig.update_layout(height=320, margin=dict(l=30, r=30, t=40, b=30))
        placeholder_chart.plotly_chart(fig, use_container_width=True, key=f"chart_{i}")

        # Update alert log
        if log_rows:
            placeholder_log.dataframe(
                pd.DataFrame(log_rows[-20:]).iloc[::-1],
                use_container_width=True, height=300,
            )
        progress.progress(seen / len(stream), text=f"{seen}/{len(stream)}")
        time.sleep(1.0 / rate)

    st.success(f"Stream finished — {fraud_count} alerts out of {len(stream)} transactions.")
else:
    section_title("How it works")
    st.markdown(
        """
        1. We sample real transactions from PaySim (with a slight fraud-bias so the
           dashboard is informative).
        2. Each row is sent through the deployed `predict_paysim` API one by one.
        3. KPIs, probability trace, and alert log update on every tick.
        """
    )
