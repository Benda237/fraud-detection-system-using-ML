# FraudGuard inference API — FastAPI on port 7860 (HF Spaces convention).
# Bundles the project's `src/` package so joblib can unpickle the preprocessor
# classes saved during training.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/home/user/app \
    HOME=/home/user

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app
ENV PATH="/home/user/.local/bin:${PATH}"

COPY --chown=user:user requirements.txt ./
RUN pip install --user -r requirements.txt

COPY --chown=user:user src ./src
COPY --chown=user:user app ./app

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
