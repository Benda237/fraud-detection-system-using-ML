"""Page 7 — settings, cache controls, HF status."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import shutil

import streamlit as st

from app.components import kpi, page_setup, section_title
from app.components.sidebar import render_sidebar
from src.config import (
    ARTIFACTS,
    HF_CACHE_DIR,
    HF_REPO_ID,
    HF_TOKEN,
    MODELS_LOCAL_DIR,
)
from src.hf_hub.downloader import clear_cache

page_setup("Settings", icon="⚙️")
render_sidebar()

st.title("⚙️ Settings")

section_title("Environment status")
c1, c2, c3 = st.columns(3)
with c1:
    token_status = "✅ loaded" if (HF_TOKEN and HF_TOKEN.startswith("hf_")) else "❌ missing"
    st.markdown(kpi("HF token", token_status), unsafe_allow_html=True)
with c2:
    st.markdown(kpi("HF repo", HF_REPO_ID), unsafe_allow_html=True)
with c3:
    cached = sum(p.stat().st_size for p in HF_CACHE_DIR.rglob("*") if p.is_file())
    st.markdown(kpi("Cache on disk", f"{cached / 1e6:.1f} MB"), unsafe_allow_html=True)

section_title("Artifacts inventory")
rows: list[dict] = []
for human, fn in ARTIFACTS.items():
    local = MODELS_LOCAL_DIR / fn
    rows.append({
        "Artifact": human,
        "Filename": fn,
        "Present locally": "✅" if local.exists() else "—",
        "Size (KB)": f"{local.stat().st_size / 1024:.1f}" if local.exists() else "",
    })
st.dataframe(rows, use_container_width=True)

section_title("Maintenance")
mc1, mc2 = st.columns(2)
with mc1:
    if st.button("🧹 Clear in-memory model cache",
                 help="Forces next prediction to re-load models from disk cache",
                 use_container_width=True):
        clear_cache()
        st.success("In-memory cache cleared.")

with mc2:
    if st.button("🗑️ Wipe HF disk cache",
                 help="Deletes downloaded model files. They re-download on next use.",
                 use_container_width=True):
        if HF_CACHE_DIR.exists():
            shutil.rmtree(HF_CACHE_DIR, ignore_errors=True)
            HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        clear_cache()
        st.success("Disk cache wiped.")

section_title("How to retrain")
st.code(
    """# Download data (only needed once)
python scripts/download_data.py

# Train both models + push to HF Hub
python scripts/train_all.py

# Faster iteration with PaySim subsample
python scripts/train_all.py --sample 500000
""",
    language="bash",
)

st.caption(
    "All credentials live in `.env` (gitignored). "
    "Rotate them at https://huggingface.co/settings/tokens "
    "and https://www.kaggle.com/settings."
)
