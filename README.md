# 🛡️ FraudGuard — ML-based Fraud Detection System

A Machine Learning Fraud Detection System (ML-FDS) for commercial banks and
credit unions. **All training and notebook execution runs on Hugging Face Spaces** —
nothing trains locally. The local machine only runs the Streamlit GUI, which
consumes trained models + datasets from Hugging Face Hub.

> Built around the research framework on fraud detection in the Cameroonian
> banking sector — see *Research framing* below.

---

## 🗺️ Architecture in one picture

```
┌─────────────────────────────────────────┐        ┌──────────────────────────┐
│  HF Space  fraudguard-training-lab      │        │  Local machine           │
│  (Docker + JupyterLab on HF infra)      │        │  Streamlit GUI only      │
│                                         │        │                          │
│  notebooks/                             │        │  app/                    │
│  ├── 01 setup_datasets    ──┐           │        │  ├── Home                │
│  ├── 02 EDA                 │           │        │  └── pages/ ×7           │
│  ├── 03 train RF            │  pushes   │        │                          │
│  ├── 04 train IF            │           │        │  Pulls models + data via │
│  └── 05 compare + test      │           │        │  huggingface_hub         │
│                             ▼           │        │  (cached on disk)        │
│                  ┌────────────────────┐ │        │                          │
│                  │  HF Hub            │ │ ◀──────│                          │
│                  │  ├─ datasets repo  │ │        │                          │
│                  │  └─ models repo    │ │        │                          │
│                  └────────────────────┘ │        │                          │
└─────────────────────────────────────────┘        └──────────────────────────┘
```

---

## ⚡ Quick start

### Step 1 — local clone + minimal install (for the GUI only)

```bash
git clone https://github.com/mdhaggai/fraud-detection-system-using-ML
cd fraud-detection-system-using-ML
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in HF_TOKEN, HF_USERNAME
```

### Step 2 — create your HF training Space (one command)

```bash
./scripts/push_to_hf_space.sh                    # private Space
# or:
./scripts/push_to_hf_space.sh public
```

This bundles `hf_space/` + `src/` and pushes them to
`https://huggingface.co/spaces/<HF_USERNAME>/fraudguard-training-lab`.

### Step 3 — configure secrets in your HF Space

In the Space → **Settings → Variables and secrets** → add:

| Name | Value |
|---|---|
| `HF_TOKEN` | your write token from huggingface.co/settings/tokens |
| `KAGGLE_USERNAME` | your Kaggle username |
| `KAGGLE_KEY` | your Kaggle API key |

### Step 4 — run notebooks on HF, in order

Open the Jupyter tab of the Space and run:

1. **01_setup_datasets.ipynb** — pulls PaySim from Kaggle, generates the custom dataset, pushes both as a private HF dataset repo.
2. **02_eda.ipynb** — exploratory analysis from the HF dataset.
3. **03_train_random_forest.ipynb** — trains RF, pushes to your model repo.
4. **04_train_isolation_forest.ipynb** — trains IF, pushes to the same repo.
5. **05_compare_and_test.ipynb** — comparison table, ROC curves, McNemar hypothesis test, model card.

### Step 5 — launch the local GUI

```bash
./run_app.sh
```

The Streamlit app opens at `http://localhost:8501`. It pulls trained models
from your HF model repo on first prediction and caches them on disk.

---

## 🗺️ Repository layout

```
fraud-detection-system-using-ML/
├── hf_space/                       ← deployed to HF Space via push_to_hf_space.sh
│   ├── README.md                   ← Space README (HF Spaces metadata header)
│   ├── Dockerfile                  ← JupyterLab on port 7860
│   ├── requirements.txt            ← training-environment deps
│   └── notebooks/
│       ├── 01_setup_datasets.ipynb
│       ├── 02_eda.ipynb
│       ├── 03_train_random_forest.ipynb
│       ├── 04_train_isolation_forest.ipynb
│       └── 05_compare_and_test.ipynb
│
├── src/                            ← shared package — used by BOTH the Space (training) and the GUI (inference)
│   ├── config.py                   ← paths + HF/Kaggle creds from .env
│   ├── data/
│   │   ├── loader.py               ← HF-dataset-repo loaders (load_paysim, load_custom)
│   │   ├── preprocessor.py         ← PaySimPreprocessor + CustomPreprocessor (joblib-friendly)
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── isolation_forest.py     ← if_predict_fraud helper
│   │   └── rule_based_baseline.py  ← paysim_rule_predict + custom_rule_predict
│   ├── evaluation/
│   │   └── metrics.py
│   ├── hf_hub/
│   │   └── downloader.py           ← cached HF Hub artifact downloader
│   └── inference/
│       └── predictor.py            ← unified predict_paysim / predict_custom API
│
├── app/                            ← Streamlit multi-page GUI (local)
│   ├── Home.py
│   ├── pages/
│   │   ├── 1_Single_Transaction.py
│   │   ├── 2_Batch_Upload.py
│   │   ├── 3_Model_Comparison.py
│   │   ├── 4_Data_Explorer.py
│   │   ├── 5_Live_Monitor.py
│   │   ├── 6_Model_Insights.py
│   │   └── 7_Settings.py
│   ├── components/                 ← forms, charts, sidebar, CSS
│   └── assets/
│
├── scripts/
│   └── push_to_hf_space.sh         ← bundles + uploads hf_space/ + src/ to your HF Space
│
├── tests/
│   └── test_pipeline.py            ← smoke tests for preprocessor/baseline/metrics
│
├── requirements.txt                ← local GUI deps
├── .env.example
└── run_app.sh
```

---

## 🤗 What gets created on Hugging Face

After the notebooks finish:

| Repo | Type | Purpose |
|---|---|---|
| `<HF_USERNAME>/fraudguard-training-lab` | Space | Where you run training notebooks |
| `<HF_USERNAME>/fraud-detection-datasets` | Dataset (private) | `paysim.csv`, `custom.csv` |
| `<HF_USERNAME>/fraud-detection-models` | Model (private) | `random_forest.joblib`, `isolation_forest.joblib`, `*_preprocessor.joblib`, `metadata.json`, auto-generated `README.md` |

---

## 🧠 Models

| Model | Dataset | Algorithm | Class imbalance |
|---|---|---|---|
| RandomForest | PaySim (~6.3M rows) | Supervised | SMOTE + `class_weight='balanced'` |
| IsolationForest | Custom (51k rows) | Unsupervised | Fit on legit-only; `contamination` inferred |
| Rule-based baseline | Both | Threshold heuristics | n/a — used for comparison |

All three are evaluated on the same held-out test split using Accuracy,
Precision, Recall, F1, ROC-AUC, and confusion matrices. **McNemar's test**
provides the statistical evidence for H1 vs HA.

---

## 🖥️ The GUI — pages overview

| Page | What it does |
|---|---|
| Home | Hero, system overview, research framing |
| Single Transaction | Form → real-time prediction with gauge + verdict + rule-based comparison |
| Batch Upload | Upload CSV → score every row → download annotated results |
| Model Comparison | Metrics table, ROC curves, confusion matrices, McNemar |
| Data Explorer | Interactive EDA — distributions, correlation, filters (pulls samples from HF dataset) |
| Live Monitor | Simulated streaming dashboard with live alert log |
| Model Insights | Feature importance, SMOTE before/after, IF config |
| Settings | HF status, cache controls, retraining instructions |

---

## 🔐 Secrets

Never commit `.env`. Rotate tokens at:
- HF: <https://huggingface.co/settings/tokens>
- Kaggle: <https://www.kaggle.com/settings>

In **HF Space → Settings → Variables and secrets**, the Space picks up
`HF_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY` automatically — notebooks read
them via `os.environ`.

---

## 🎓 Research framing

**Problem.** Cameroonian commercial banks face sophisticated, adaptive fraud
and rely on outdated rule-based systems that produce too many false positives
and miss novel patterns.

**Main objective.** Design, develop, and evaluate an ML-FDS that improves the
accuracy and real-time detection rate of financial fraud transactions.

**Specific objectives** addressed by this codebase:

1. Comparative analysis of RandomForest vs IsolationForest — notebooks 03 + 04
2. Pre-processing + feature engineering to mitigate class imbalance — `src/data/preprocessor.py`
3. Training using appropriate datasets (PaySim + custom) — notebook 01 ingests, 03/04 train
4. Evaluation against a rule-based baseline — `src/models/rule_based_baseline.py` + notebook 05
5. Architectural framework — HF Hub for versioned, secure model artifacts; Streamlit GUI for human review

**Hypotheses (tested on the Model Comparison page):**

- H1 (null): ML-FDS does not significantly improve accuracy.
- HA: ML-FDS significantly improves accuracy.

McNemar's test on the held-out test set provides the statistical evidence
needed to accept or reject H1.

---

## License

MIT — for academic and research use.
