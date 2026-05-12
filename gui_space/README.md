---
title: FraudGuard GUI
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
license: mit
---

# FraudGuard — Streamlit GUI (Hugging Face Space)

The user-facing fraud detection app. Pulls predictions from the FraudGuard
**Inference API** Space (`Benda237/fraudguard-api`) over HTTPS. No model files
or sklearn inference run here — this Space is a pure frontend.

## Pages

| Page | What it does |
|---|---|
| Home | Hero + research framing |
| Single Transaction | Form → real-time fraud verdict |
| Batch Upload | CSV upload → batch scoring → download results |
| Model Comparison | Metrics table, ROC curves, McNemar |
| Data Explorer | Interactive EDA from the HF dataset repo |
| Live Monitor | Simulated streaming dashboard |
| Model Insights | Feature importance, SMOTE before/after |
| Settings | API health, cache controls, retraining cheat sheet |

## Required secrets (Space → Settings → Variables and secrets)

| Name | Value |
|---|---|
| `HF_TOKEN` | HF token, used as bearer when calling the API Space |
| `HF_USERNAME` | your HF username (e.g. `Benda237`) |
| `FRAUDGUARD_API_URL` | `https://<HF_USERNAME>-fraudguard-api.hf.space` |
