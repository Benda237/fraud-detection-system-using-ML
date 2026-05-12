"""Pull model artifacts from the HF Hub. Cached on disk via huggingface_hub's
built-in cache so repeated calls are free."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
from huggingface_hub import hf_hub_download

from src.config import ARTIFACTS, HF_CACHE_DIR, HF_REPO_ID, HF_TOKEN


def _download(filename: str) -> Path:
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        token=HF_TOKEN or None,
        cache_dir=str(HF_CACHE_DIR),
    )
    return Path(path)


@lru_cache(maxsize=8)
def get_rf_model():
    return joblib.load(_download(ARTIFACTS["rf_model"]))


@lru_cache(maxsize=8)
def get_if_model():
    return joblib.load(_download(ARTIFACTS["if_model"]))


@lru_cache(maxsize=8)
def get_paysim_preprocessor():
    return joblib.load(_download(ARTIFACTS["paysim_preprocessor"]))


@lru_cache(maxsize=8)
def get_custom_preprocessor():
    return joblib.load(_download(ARTIFACTS["custom_preprocessor"]))


@lru_cache(maxsize=2)
def get_metadata() -> dict:
    try:
        return json.loads(_download(ARTIFACTS["metadata"]).read_text())
    except Exception:  # noqa: BLE001
        return {}


def clear_cache() -> None:
    """Invalidate the in-memory cache. Disk cache stays — call this after the
    HF repo has been re-pushed to force a fresh download next call."""
    for fn in (get_rf_model, get_if_model, get_paysim_preprocessor,
               get_custom_preprocessor, get_metadata):
        fn.cache_clear()
