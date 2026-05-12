---
title: FraudGuard Training Lab
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# FraudGuard — Training Lab (Hugging Face Space)

This Space is the **training infrastructure** for the FraudGuard ML-FDS.
All notebooks run here on Hugging Face's CPU; nothing trains locally.

## What lives in this Space

| Notebook | Purpose |
|---|---|
| `notebooks/01_setup_datasets.ipynb` | Pull both datasets from Kaggle, push them as a private HF dataset repo |
| `notebooks/02_eda.ipynb` | Interactive EDA on the HF-hosted datasets |
| `notebooks/03_train_random_forest.ipynb` | Train RF on PaySim, push to HF model repo |
| `notebooks/04_train_isolation_forest.ipynb` | Train IF on custom dataset, push to HF model repo |
| `notebooks/05_compare_and_test.ipynb` | Cross-model comparison + McNemar hypothesis test, push metadata |

## How to use this Space

1. **Duplicate** this Space into your account (Files → Duplicate).
2. Open **Settings → Variables and secrets** and add:
   - `HF_TOKEN` — your HF write token
   - `KAGGLE_USERNAME` and `KAGGLE_KEY` — from <https://www.kaggle.com/settings>
3. Open the **Jupyter tab** of the Space.
4. Run the notebooks in numeric order. Each one is self-contained:
   - Installs its own pip dependencies in the first cell.
   - Reads tokens from environment.
   - Writes its outputs back to your HF account (datasets or models).

## What gets produced

Two HF repos in your account:

- **Dataset repo** `{HF_USERNAME}/fraud-detection-datasets` (private)
  - `paysim.csv` (or sampled subset)
  - `custom.csv`
- **Model repo** `{HF_USERNAME}/fraud-detection-models`
  - `random_forest.joblib`
  - `isolation_forest.joblib`
  - `paysim_preprocessor.joblib`
  - `custom_preprocessor.joblib`
  - `metadata.json`
  - `README.md` (auto-generated model card)

Once these exist, the local **Streamlit GUI** in the main repo can consume the
models over HF Hub. The GUI itself does no training.
