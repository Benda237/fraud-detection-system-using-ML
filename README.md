# 🛡️ FraudGuard — ML-based Fraud Detection System

A Machine Learning Fraud Detection System (ML-FDS) for commercial banks and
credit unions. Trains **Random Forest** + **Isolation Forest** models on real
transaction data, publishes them to **Hugging Face Hub**, and serves them
through a rich **Streamlit** GUI with multiple pages.

> Tailored to the research framework on fraud detection in the Cameroonian
> banking sector — see *Research framing* below.

---

## ⚡ Quick start

```bash
# 1. clone and enter
git clone https://github.com/mdhaggai/fraud-detection-system-using-ML
cd fraud-detection-system-using-ML

# 2. install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. configure secrets — copy .env.example to .env, fill in HF + Kaggle creds
cp .env.example .env

# 4. download both datasets from Kaggle
python scripts/download_data.py

# 5. train + push models to HF Hub
python scripts/train_all.py            # full PaySim (slow)
python scripts/train_all.py --sample 500000   # faster iteration

# 6. launch the GUI
./run_app.sh
#   or:  PYTHONPATH=. streamlit run app/Home.py
```

The app opens at <http://localhost:8501>.

---

## 🗺️ Folder structure

```
fraud-detection-system-using-ML/
├── data/raw/                    drop CSVs here (or auto-downloaded)
├── notebooks/                   cleaned EDA + training notebooks
├── src/
│   ├── config.py                paths, HF/Kaggle creds, hyperparameters
│   ├── data/                    loaders, preprocessors, feature engineering
│   ├── models/                  RF, IF, rule-based baseline, trainer
│   ├── evaluation/              metrics + McNemar hypothesis test
│   ├── hf_hub/                  upload & download model artifacts
│   └── inference/               unified predict() API
├── app/                         Streamlit multi-page GUI
│   ├── Home.py
│   ├── pages/                   7 pages (single, batch, comparison, …)
│   └── components/              forms, charts, sidebar, CSS
├── scripts/                     download_data, train_all, verify_hf_models
├── tests/                       pytest smoke tests
└── models_local/                trained artifacts staged for HF upload
```

---

## 🧠 The models

| Model | Dataset | Algorithm | Class imbalance |
|---|---|---|---|
| RandomForest | PaySim (~6.3M rows) | Supervised | SMOTE + `class_weight='balanced'` |
| IsolationForest | Custom (51k rows) | Unsupervised | Fit on legit-only; `contamination` inferred |
| Rule-based baseline | Both | Threshold heuristics | n/a — used for comparison |

All three are evaluated on the same held-out test split using **Accuracy,
Precision, Recall, F1, ROC-AUC**, and confusion matrices. **McNemar's test**
decides H1 vs HA on each dataset.

---

## ☁️ Hugging Face Hub layout

After `train_all.py`, the following are pushed to `HF_USERNAME/HF_MODEL_REPO`:

| File | Purpose |
|---|---|
| `random_forest.joblib` | Fitted RandomForestClassifier |
| `isolation_forest.joblib` | Fitted IsolationForest |
| `paysim_preprocessor.joblib` | PaySim encoders + scaler |
| `custom_preprocessor.joblib` | Custom-dataset encoders + scaler |
| `metadata.json` | Full metrics + McNemar results + feature importances |
| `README.md` | Auto-generated model card |

The Streamlit app pulls these lazily on first prediction and caches them on disk.

---

## 🖥️ The GUI pages

| Page | What it does |
|---|---|
| **Home** | Hero, system overview, research framing |
| **Single Transaction** | Form → real-time prediction with gauge + verdict |
| **Batch Upload** | Upload CSV → score every row → download results |
| **Model Comparison** | Metrics table, ROC curves, confusion matrices, McNemar |
| **Data Explorer** | Interactive EDA — distributions, correlation, filters |
| **Live Monitor** | Simulated streaming dashboard with alert log |
| **Model Insights** | Feature importance, SMOTE before/after, IF config |
| **Settings** | HF status, cache controls, retraining cheat sheet |

---

## 🧪 Tests

```bash
PYTHONPATH=. pytest -q
```

Tests are designed to pass even without the CSVs — dataset-dependent tests are
skipped automatically when raw files are missing.

---

## 🔐 Secrets

Never commit `.env`. Both providers let you rotate tokens at any time:
- HF: <https://huggingface.co/settings/tokens>
- Kaggle: <https://www.kaggle.com/settings>

---

## 🎓 Research framing

**Problem.** Cameroonian commercial banks face sophisticated fraud and rely on
outdated rule-based systems that produce too many false positives and miss
novel patterns.

**Main objective.** Design, develop, and evaluate an ML-FDS that improves the
accuracy and real-time detection rate of financial fraud.

**Specific objectives** addressed by this codebase:
- Comparative analysis of RandomForest vs IsolationForest (`src/models/`)
- Pre-processing + feature engineering to mitigate class imbalance (`src/data/`)
- Training using appropriate datasets (PaySim 6.3M + custom 51k)
- Evaluation against a rule-based baseline (`src/evaluation/`)
- Architectural framework — HF Hub for versioned, secure model artifacts;
  Streamlit GUI for human-in-the-loop review

**Hypotheses (tested on the Model Comparison page):**
- H1 (null): ML-FDS does not significantly improve accuracy.
- HA: ML-FDS significantly improves accuracy.

McNemar's test on the held-out test set provides the statistical evidence
needed to accept or reject H1.

---

## License

MIT — for academic and research use.
