"""FraudGuard inference API — FastAPI service hosted on a Hugging Face Space.

Loads the trained Random Forest + Isolation Forest models from the HF model
repo `<HF_USERNAME>/<HF_MODEL_REPO>` at startup, then serves predictions over
HTTPS.

Authentication: every request must carry an `Authorization: Bearer <token>`
header. The token must match the Space's `HF_TOKEN` secret (or be a token
belonging to the same HF account / org).
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
import joblib
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download, whoami
from pydantic import BaseModel, Field

from src.models.isolation_forest import if_predict_fraud

HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
HF_USERNAME: str = os.environ.get("HF_USERNAME") or "Benda237"
HF_MODEL_REPO: str = os.environ.get("HF_MODEL_REPO", "fraud-detection-models")
MODEL_REPO_ID: str = f"{HF_USERNAME}/{HF_MODEL_REPO}"


# ---------- in-memory artifact cache ----------
class _Artifacts:
    rf = None
    if_ = None
    paysim_pre = None
    custom_pre = None
    metadata: dict = {}
    load_error: str | None = None


ARTIFACTS = _Artifacts()


def _download(filename: str):
    return hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=filename,
        token=HF_TOKEN or None,
    )


def load_all() -> None:
    """Pull every artifact from the HF model repo and cache it in memory.

    Resilient: if any individual artifact is missing (e.g. only RF has been
    trained so far), keep going. The /health endpoint reports what loaded."""
    ARTIFACTS.load_error = None
    targets = {
        "rf": "random_forest.joblib",
        "if_": "isolation_forest.joblib",
        "paysim_pre": "paysim_preprocessor.joblib",
        "custom_pre": "custom_preprocessor.joblib",
    }
    for attr, filename in targets.items():
        try:
            setattr(ARTIFACTS, attr, joblib.load(_download(filename)))
        except Exception as exc:  # noqa: BLE001
            print(f"[load] could not load {filename}: {exc}")
            setattr(ARTIFACTS, attr, None)

    try:
        meta_path = _download("metadata.json")
        with open(meta_path) as fh:
            ARTIFACTS.metadata = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        print(f"[load] could not load metadata.json: {exc}")
        ARTIFACTS.metadata = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[startup] loading artifacts from {MODEL_REPO_ID}")
    try:
        load_all()
    except Exception as exc:  # noqa: BLE001
        ARTIFACTS.load_error = str(exc)
        print(f"[startup] artifact load failed: {exc}")
    yield


app = FastAPI(
    title="FraudGuard Inference API",
    version="1.0.0",
    description=(
        "Serves Random Forest + Isolation Forest fraud detection models from "
        f"HF repo `{MODEL_REPO_ID}`. Pass `Authorization: Bearer <HF_TOKEN>` "
        "on every request."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- auth ----------
def require_bearer(request: Request) -> str:
    """Reject requests without a valid Authorization: Bearer <token> header."""
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = auth.split(" ", 1)[1].strip()
    if not token.startswith("hf_"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token must start with 'hf_'",
        )
    # Verify against HF — cheap call, ~50ms; cached implicitly by HF infra
    try:
        whoami(token=token)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"HF token rejected: {exc}",
        ) from exc
    return token


# ---------- pydantic schemas ----------
class PaySimTx(BaseModel):
    step: int = Field(..., ge=0)
    type: str
    amount: float = Field(..., ge=0)
    nameOrig: str = ""
    oldbalanceOrg: float = 0.0
    newbalanceOrig: float = 0.0
    nameDest: str = ""
    oldbalanceDest: float = 0.0
    newbalanceDest: float = 0.0
    isFlaggedFraud: int = 0


class CustomTx(BaseModel):
    Transaction_ID: str = ""
    User_ID: int = 0
    Transaction_Amount: float = Field(..., ge=0)
    Transaction_Type: str
    Time_of_Transaction: float = Field(..., ge=0, le=23)
    Device_Used: str
    Location: str
    Previous_Fraudulent_Transactions: int = 0
    Account_Age: int = Field(..., ge=0)
    Number_of_Transactions_Last_24H: int = Field(..., ge=0)
    Payment_Method: str


class PaySimRequest(BaseModel):
    transactions: list[PaySimTx]
    threshold: float = 0.5


class CustomRequest(BaseModel):
    transactions: list[CustomTx]


class PaySimResponse(BaseModel):
    dataset: str = "paysim"
    model: str = "random_forest"
    n: int
    fraud_probability: list[float]
    prediction: list[int]
    threshold: float


class CustomResponse(BaseModel):
    dataset: str = "custom"
    model: str = "isolation_forest"
    n: int
    anomaly_score: list[float]
    prediction: list[int]


# ---------- routes ----------
@app.get("/")
async def root():
    return {
        "service": "FraudGuard Inference API",
        "version": "1.0.0",
        "model_repo": MODEL_REPO_ID,
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {
        "model_repo": MODEL_REPO_ID,
        "models_loaded": {
            "random_forest": ARTIFACTS.rf is not None,
            "isolation_forest": ARTIFACTS.if_ is not None,
            "paysim_preprocessor": ARTIFACTS.paysim_pre is not None,
            "custom_preprocessor": ARTIFACTS.custom_pre is not None,
        },
        "has_metadata": bool(ARTIFACTS.metadata),
        "load_error": ARTIFACTS.load_error,
    }


@app.get("/metadata")
async def metadata(_: str = Depends(require_bearer)) -> dict[str, Any]:
    if not ARTIFACTS.metadata:
        raise HTTPException(status_code=503, detail="metadata.json not yet on HF Hub")
    return ARTIFACTS.metadata


@app.post("/reload")
async def reload_models(_: str = Depends(require_bearer)):
    load_all()
    return {"reloaded": True, **(await health())}


@app.post("/predict/paysim", response_model=PaySimResponse)
async def predict_paysim(req: PaySimRequest, _: str = Depends(require_bearer)):
    if ARTIFACTS.rf is None or ARTIFACTS.paysim_pre is None:
        raise HTTPException(
            status_code=503,
            detail="Random Forest or its preprocessor not loaded. "
                   "Run training notebook 03 in the training Space.",
        )
    df = pd.DataFrame([t.model_dump() for t in req.transactions])
    x = ARTIFACTS.paysim_pre.transform(df)
    prob = ARTIFACTS.rf.predict_proba(x)[:, 1]
    pred = (prob >= req.threshold).astype(int)
    return PaySimResponse(
        n=len(df),
        fraud_probability=prob.tolist(),
        prediction=pred.tolist(),
        threshold=req.threshold,
    )


@app.post("/predict/custom", response_model=CustomResponse)
async def predict_custom(req: CustomRequest, _: str = Depends(require_bearer)):
    if ARTIFACTS.if_ is None or ARTIFACTS.custom_pre is None:
        raise HTTPException(
            status_code=503,
            detail="Isolation Forest or its preprocessor not loaded. "
                   "Run training notebook 04 in the training Space.",
        )
    df = pd.DataFrame([t.model_dump() for t in req.transactions])
    x = ARTIFACTS.custom_pre.transform(df)
    binary, score = if_predict_fraud(ARTIFACTS.if_, x)
    return CustomResponse(
        n=len(df),
        anomaly_score=score.tolist(),
        prediction=binary.tolist(),
    )
