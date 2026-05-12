"""Shared sidebar — model status, HF repo link, threshold control."""
from __future__ import annotations

import streamlit as st

from src.config import HF_REPO_ID, HF_TOKEN


def render_sidebar() -> dict:
    """Render the sidebar and return a dict of user choices."""
    with st.sidebar:
        st.markdown("## 🛡️ FraudGuard")
        st.caption("ML-based Fraud Detection System")
        st.markdown("---")

        st.markdown("### Configuration")
        threshold = st.slider(
            "Fraud probability threshold",
            min_value=0.05, max_value=0.95, value=0.50, step=0.05,
            help="Transactions with predicted fraud probability ≥ this value are flagged.",
        )

        st.markdown("---")
        st.markdown("### Hugging Face Hub")
        token_ok = bool(HF_TOKEN and HF_TOKEN.startswith("hf_"))
        st.markdown(f"**Repo:** `{HF_REPO_ID}`")
        st.markdown(f"**Token:** {'✅ loaded' if token_ok else '❌ missing'}")
        st.link_button("Open on HF Hub", f"https://huggingface.co/{HF_REPO_ID}")

        st.markdown("---")
        st.caption(
            "Models pulled lazily from HF and cached locally — "
            "first prediction may take a few seconds."
        )
    return {"threshold": threshold}
