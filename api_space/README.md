---
title: FraudGuard Inference API
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# FraudGuard — Inference API (Hugging Face Space)

FastAPI service that loads the trained Random Forest + Isolation Forest from
the model repo `<HF_USERNAME>/fraud-detection-models` and exposes them over
HTTPS. The local Streamlit GUI calls these endpoints — no sklearn inference
happens locally.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service banner + docs link |
| GET | `/health` | Reports model load status + repo + cache stats |
| GET | `/metadata` | Returns `metadata.json` (metrics, McNemar, feature importances) |
| POST | `/predict/paysim` | Score a PaySim-schema transaction (or batch) with the RF |
| POST | `/predict/custom` | Score a custom-schema transaction (or batch) with the IF |
| POST | `/reload` | Force re-pull of model artifacts from the model repo |

## Auth

The Space is private. Pass the HF token as a bearer header on every request:

```
Authorization: Bearer <HF_TOKEN>
```

Inside the Space the same token is set as the `HF_TOKEN` secret and is used to
pull artifacts from the model repo at startup.

## Required secrets (Space Settings → Variables and secrets)

| Name | Value |
|---|---|
| `HF_TOKEN` | your HF write token |
| `HF_USERNAME` | your HF username (e.g. `Benda237`) |
| `HF_MODEL_REPO` | optional, defaults to `fraud-detection-models` |

## Request examples

### Single PaySim transaction
```bash
curl -X POST 'https://<HF_USERNAME>-fraudguard-api.hf.space/predict/paysim' \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "transactions": [{
      "step": 1, "type": "TRANSFER", "amount": 181.0,
      "nameOrig": "C1", "oldbalanceOrg": 181.0, "newbalanceOrig": 0.0,
      "nameDest": "C2", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
      "isFlaggedFraud": 0
    }],
    "threshold": 0.5
  }'
```

### Single custom transaction
```bash
curl -X POST 'https://<HF_USERNAME>-fraudguard-api.hf.space/predict/custom' \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "transactions": [{
      "Transaction_ID": "T1", "User_ID": 1234,
      "Transaction_Amount": 4500.0, "Transaction_Type": "ATM Withdrawal",
      "Time_of_Transaction": 23, "Device_Used": "Mobile",
      "Location": "New York", "Previous_Fraudulent_Transactions": 3,
      "Account_Age": 5, "Number_of_Transactions_Last_24H": 12,
      "Payment_Method": "Credit Card"
    }]
  }'
```
