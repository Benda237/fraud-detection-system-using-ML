"""Push trained model artifacts to a Hugging Face Hub model repo.

The repo is created on first run (private by default). Subsequent runs
overwrite the artifacts (HF Hub keeps full version history).
"""
from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file

from src.config import ARTIFACTS, HF_REPO_ID, HF_TOKEN, MODELS_LOCAL_DIR


def _ensure_token() -> str:
    if not HF_TOKEN or not HF_TOKEN.startswith("hf_"):
        raise RuntimeError(
            "HF_TOKEN missing or malformed. Set it in .env "
            "(get one at https://huggingface.co/settings/tokens)."
        )
    return HF_TOKEN


def ensure_repo(private: bool = True) -> str:
    """Create the model repo on the Hub if it does not already exist."""
    token = _ensure_token()
    try:
        create_repo(repo_id=HF_REPO_ID, token=token, private=private, repo_type="model", exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to create HF repo {HF_REPO_ID}: {exc}") from exc
    return HF_REPO_ID


def _write_model_card(metadata: dict) -> Path:
    """Generate a model card README using the metrics from training."""
    card = MODELS_LOCAL_DIR / ARTIFACTS["model_card"]
    paysim = metadata.get("paysim", {})
    custom = metadata.get("custom", {})
    rf = paysim.get("rf", {})
    iso = custom.get("isolation_forest", {})

    body = f"""---
license: mit
tags:
  - fraud-detection
  - random-forest
  - isolation-forest
  - tabular-classification
  - banking
---

# Fraud Detection Models — Random Forest + Isolation Forest

Trained as part of an ML-FDS (Machine Learning Fraud Detection System) study for
commercial banks and credit unions in Cameroon. Targets the research question:

> *How can a Machine Learning-based model be effectively designed and implemented
> to significantly improve the accuracy and efficiency for financial fraud
> detection in commercial banks and Financial Institutions?*

## Artifacts in this repo

| File | Purpose |
|---|---|
| `{ARTIFACTS['rf_model']}` | RandomForestClassifier trained on PaySim with SMOTE |
| `{ARTIFACTS['if_model']}` | IsolationForest trained on the custom mixed-feature dataset |
| `{ARTIFACTS['paysim_preprocessor']}` | Fitted PaySimPreprocessor (encoders + scaler) |
| `{ARTIFACTS['custom_preprocessor']}` | Fitted CustomPreprocessor (encoders + scaler) |
| `{ARTIFACTS['metadata']}` | Full training metrics, feature importances, hypothesis-test results |

## Headline metrics

### Random Forest on PaySim (supervised)
- Accuracy: **{rf.get('accuracy', 'n/a')}**
- Precision: **{rf.get('precision', 'n/a')}**
- Recall: **{rf.get('recall', 'n/a')}**
- F1: **{rf.get('f1', 'n/a')}**
- ROC-AUC: **{rf.get('roc_auc', 'n/a')}**

### Isolation Forest on custom dataset (unsupervised)
- Accuracy: **{iso.get('accuracy', 'n/a')}**
- Precision: **{iso.get('precision', 'n/a')}**
- Recall: **{iso.get('recall', 'n/a')}**
- F1: **{iso.get('f1', 'n/a')}**

## Usage

```python
from huggingface_hub import hf_hub_download
import joblib

rf_path = hf_hub_download(repo_id="{HF_REPO_ID}", filename="{ARTIFACTS['rf_model']}")
pre_path = hf_hub_download(repo_id="{HF_REPO_ID}", filename="{ARTIFACTS['paysim_preprocessor']}")
rf = joblib.load(rf_path)
preprocessor = joblib.load(pre_path)
```

## Class imbalance handling
PaySim fraud rate is ~0.13%. We apply SMOTE on the training fold only.

## Source code
https://github.com/mdhaggai/fraud-detection-system-using-ML
"""
    card.write_text(body)
    return card


def push_all() -> dict:
    """Push every artifact in MODELS_LOCAL_DIR to the configured HF repo."""
    token = _ensure_token()
    ensure_repo()

    metadata_file = MODELS_LOCAL_DIR / ARTIFACTS["metadata"]
    metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
    _write_model_card(metadata)

    api = HfApi()
    uploaded: list[str] = []
    skipped: list[str] = []
    for name in ARTIFACTS.values():
        path = MODELS_LOCAL_DIR / name
        if not path.exists():
            skipped.append(name)
            continue
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=HF_REPO_ID,
            repo_type="model",
            token=token,
            commit_message=f"Update {name}",
        )
        uploaded.append(name)
    return {"repo": HF_REPO_ID, "uploaded": uploaded, "skipped": skipped}
