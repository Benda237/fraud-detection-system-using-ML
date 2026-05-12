"""Plotly chart factories used across pages."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PLOTLY_TEMPLATE = "plotly_dark"


def confusion_matrix_heatmap(cm: np.ndarray, labels: list[str] = ("Legit", "Fraud")) -> go.Figure:
    z = cm
    text = [[str(v) for v in row] for row in z]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=list(labels),
            y=list(labels),
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        margin=dict(l=30, r=30, t=30, b=30),
        height=380,
    )
    return fig


def roc_curve_fig(curves: dict[str, dict]) -> go.Figure:
    """Plot one or more ROC curves overlaid.

    `curves` maps model name → {fpr: [...], tpr: [...], auc: float}.
    """
    fig = go.Figure()
    for name, data in curves.items():
        fig.add_trace(
            go.Scatter(
                x=data.get("fpr", []),
                y=data.get("tpr", []),
                mode="lines",
                name=f"{name} (AUC={data.get('auc', 0):.3f})",
                line=dict(width=2.5),
            )
        )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                   line=dict(dash="dash", color="gray"))
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        title="ROC Curves",
        margin=dict(l=30, r=30, t=40, b=30),
        height=420,
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
    )
    return fig


def feature_importance_bar(importances: dict[str, float], top_n: int = 12) -> go.Figure:
    items = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    feats, vals = zip(*items)
    fig = go.Figure(
        go.Bar(x=list(vals), y=list(feats), orientation="h",
               marker=dict(color="#22d3ee"))
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Feature Importance",
        xaxis_title="Importance",
        yaxis=dict(autorange="reversed"),
        height=420,
        margin=dict(l=30, r=30, t=40, b=30),
    )
    return fig


def metric_comparison_bar(rows: list[dict]) -> go.Figure:
    """rows = [{"model": "RF", "accuracy": .., "precision": .., ...}, ...]"""
    df = pd.DataFrame(rows)
    long = df.melt(id_vars=["model"], var_name="metric", value_name="score")
    fig = px.bar(
        long, x="metric", y="score", color="model", barmode="group",
        template=PLOTLY_TEMPLATE, height=420,
    )
    fig.update_layout(
        title="Model Comparison",
        margin=dict(l=30, r=30, t=40, b=30),
        yaxis=dict(range=[0, 1.0]),
    )
    return fig


def amount_distribution(df: pd.DataFrame, amount_col: str, fraud_col: str) -> go.Figure:
    fig = px.histogram(
        df, x=amount_col, color=fraud_col, log_y=True, nbins=60,
        template=PLOTLY_TEMPLATE, opacity=0.75,
        title=f"Distribution of {amount_col} by {fraud_col}",
        color_discrete_map={0: "#22d3ee", 1: "#f43f5e"},
    )
    fig.update_layout(height=420, margin=dict(l=30, r=30, t=40, b=30))
    return fig


def gauge(score: float, title: str = "Risk score") -> go.Figure:
    score = float(max(0.0, min(1.0, score)))
    if score < 0.33:
        color = "#10b981"
    elif score < 0.66:
        color = "#f59e0b"
    else:
        color = "#ef4444"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={"suffix": "%"},
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title, "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 33], "color": "#1f3a2e"},
                    {"range": [33, 66], "color": "#3a2e1f"},
                    {"range": [66, 100], "color": "#3a1f1f"},
                ],
            },
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=300, margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig
