# Dockerfile for the FraudGuard Training Space.
# Spins up JupyterLab on port 7860 (HF Spaces convention) with training deps preinstalled.
# Ships the project's `src/` package so notebooks can import the same preprocessor
# classes the local Streamlit GUI uses — keeps joblib unpickling identical on both sides.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/home/user/app \
    HOME=/home/user

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app
ENV PATH="/home/user/.local/bin:${PATH}"

COPY --chown=user:user requirements.txt ./
RUN pip install --user -r requirements.txt

COPY --chown=user:user jupyter_server_config.py /home/user/.jupyter/jupyter_server_config.py
COPY --chown=user:user src ./src
COPY --chown=user:user notebooks ./notebooks

EXPOSE 7860
CMD ["jupyter", "lab", "--config=/home/user/.jupyter/jupyter_server_config.py"]
