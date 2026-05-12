"""Page 7 — settings, cache controls, HF status, API health."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.components import kpi, page_setup, section_title
from app.components.sidebar import render_sidebar
from src.config import ARTIFACTS, HF_CACHE_DIR, HF_REPO_ID, HF_TOKEN
from src.hf_hub.downloader import clear_cache
from src.inference.predictor import API_URL, USE_HTTP, health

page_setup("Settings", icon="⚙️")
render_sidebar()

st.title("⚙️ Settings")

section_title("Environment status")
c1, c2, c3, c4 = st.columns(4)
with c1:
    token_status = "✅ loaded" if (HF_TOKEN and HF_TOKEN.startswith("hf_")) else "❌ missing"
    st.markdown(kpi("HF token", token_status), unsafe_allow_html=True)
with c2:
    st.markdown(kpi("HF model repo", HF_REPO_ID), unsafe_allow_html=True)
with c3:
    cached = sum(p.stat().st_size for p in HF_CACHE_DIR.rglob("*") if p.is_file())
    st.markdown(kpi("Local HF cache", f"{cached / 1e6:.1f} MB"), unsafe_allow_html=True)
with c4:
    backend = "🌐 HTTP API" if USE_HTTP else "🖥️ Local sklearn"
    st.markdown(kpi("Inference backend", backend), unsafe_allow_html=True)

section_title("Inference API health")
if USE_HTTP:
    st.markdown(f"**Configured URL:** `{API_URL}`")
    if st.button("🔍 Probe /health", use_container_width=False):
        with st.spinner("Calling API…"):
            h = health()
        if "error" in h:
            st.error(f"API unreachable: {h['error']}")
        else:
            st.success("API is reachable")
            st.json(h)
    st.caption(
        "If models_loaded shows all `false`, run the training notebooks (03 + 04) "
        "in the training Space first, then POST `/reload` on this Space (or wait "
        "for the next startup)."
    )
else:
    st.warning(
        "No `FRAUDGUARD_API_URL` configured — falling back to local sklearn inference. "
        "Add `FRAUDGUARD_API_URL=https://<HF_USERNAME>-fraudguard-api.hf.space` to `.env` "
        "after deploying the API Space."
    )

section_title("Artifacts on Hugging Face")
rows: list[dict] = [
    {"Artifact": human, "Filename": fn, "Repo": HF_REPO_ID}
    for human, fn in ARTIFACTS.items()
]
st.dataframe(rows, use_container_width=True)

section_title("Maintenance")
mc1, mc2 = st.columns(2)
with mc1:
    if st.button(
        "🧹 Clear in-memory model cache",
        help="Local-fallback only: forces next prediction to re-pull models from disk cache.",
        use_container_width=True,
    ):
        clear_cache()
        st.success("In-memory cache cleared.")

with mc2:
    if st.button(
        "🗑️ Wipe local HF disk cache",
        help="Deletes downloaded model files. They re-download on next use.",
        use_container_width=True,
    ):
        if HF_CACHE_DIR.exists():
            shutil.rmtree(HF_CACHE_DIR, ignore_errors=True)
            HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        clear_cache()
        st.success("Disk cache wiped.")

section_title("How to retrain (everything happens on Hugging Face)")
st.code(
    """# Open your HF Space → Jupyter tab → run notebooks in order:
#   01_setup_datasets.ipynb        Kaggle  ->  HF dataset repo
#   02_eda.ipynb                   exploratory analysis
#   03_train_random_forest.ipynb   trains RF, pushes to HF model repo
#   04_train_isolation_forest.ipynb  trains IF, pushes to HF model repo
#   05_compare_and_test.ipynb      McNemar test, model card

# After retraining, force the API Space to reload artifacts:
curl -X POST 'https://<HF_USERNAME>-fraudguard-api.hf.space/reload' \\
     -H "Authorization: Bearer $HF_TOKEN"
""",
    language="bash",
)

st.caption(
    "All credentials live in `.env` (gitignored). "
    "Rotate them at https://huggingface.co/settings/tokens and https://www.kaggle.com/settings."
)
