# Dockerfile for the FraudGuard Streamlit GUI Space.
# Streamlit on port 7860 (HF Spaces convention). Calls the FraudGuard API Space
# for predictions — no models live here.

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

COPY --chown=user:user .streamlit /home/user/.streamlit
COPY --chown=user:user src ./src
COPY --chown=user:user app ./app

EXPOSE 7860
CMD ["streamlit", "run", "app/Home.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
