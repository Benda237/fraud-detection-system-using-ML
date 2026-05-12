"""Reusable UI components for the Streamlit pages."""
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
CSS_PATH = ROOT / "assets" / "style.css"


def inject_css() -> None:
    """Inject the project stylesheet — call once at the top of each page."""
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


def page_setup(title: str, icon: str = "🛡️") -> None:
    """Common page header — sets title, icon, and injects CSS."""
    st.set_page_config(page_title=f"FraudGuard · {title}", page_icon=icon, layout="wide")
    inject_css()


def kpi(label: str, value: str, delta: str | None = None, delta_positive: bool = True) -> str:
    delta_html = ""
    if delta:
        cls = "kpi-delta-pos" if delta_positive else "kpi-delta-neg"
        delta_html = f"<div class='{cls}'>{delta}</div>"
    return (
        "<div class='kpi-card'>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{value}</div>"
        f"{delta_html}"
        "</div>"
    )


def verdict_banner(is_fraud: bool, prob: float | None = None, model: str = "") -> str:
    if is_fraud:
        title = "FRAUD DETECTED"
        sub = f"{model} · probability {prob:.1%}" if prob is not None else model
        return (
            f"<div class='verdict-fraud'>"
            f"<div class='verdict-title'>🚨 {title}</div>"
            f"<div class='verdict-sub'>{sub}</div></div>"
        )
    title = "LEGITIMATE"
    sub = f"{model} · probability {prob:.1%}" if prob is not None else model
    return (
        f"<div class='verdict-legit'>"
        f"<div class='verdict-title'>✅ {title}</div>"
        f"<div class='verdict-sub'>{sub}</div></div>"
    )


def section_title(text: str) -> None:
    st.markdown(f"<div class='section-title'>{text}</div>", unsafe_allow_html=True)
