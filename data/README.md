# Data directory

Datasets are not committed to git (too large and many require attribution).

## Layout
- `raw/` — original CSVs from Kaggle (downloaded by `scripts/download_data.py`)
- `processed/` — cached intermediate artifacts (currently unused; reserved)

## Files expected in `raw/`

| Filename | Source | Approx rows | Used by |
|---|---|---|---|
| `AIML Dataset.csv` | Kaggle `ealaxi/paysim1` | 6,362,620 | RandomForest pipeline |
| `Fraud Detection Dataset.csv` | Kaggle `goyaladi/fraud-detection-dataset` | 51,000 | IsolationForest pipeline |

## How to populate
```bash
# Set KAGGLE_USERNAME and KAGGLE_KEY in .env, then:
python scripts/download_data.py
```

## Schemas

### PaySim
- `step` (int) — hour of simulation
- `type` (categorical) — CASH_OUT, PAYMENT, TRANSFER, CASH_IN, DEBIT
- `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest` (float)
- `nameOrig`, `nameDest` (string IDs — dropped during preprocessing)
- `isFlaggedFraud` (binary) — rule already applied by simulator
- `isFraud` (binary target)

### Custom
- `Transaction_ID`, `User_ID`
- `Transaction_Amount`, `Time_of_Transaction`, `Account_Age`
- `Transaction_Type`, `Device_Used`, `Location`, `Payment_Method` (categorical)
- `Previous_Fraudulent_Transactions`, `Number_of_Transactions_Last_24H` (int)
- `Fraudulent` (binary target)
