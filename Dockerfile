# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install -e ".[dev]" --no-build-isolation

# Pre-download HuggingFace models into the image layer (cache them)
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
[AutoTokenizer.from_pretrained(m) for m in [\
    'ProsusAI/finbert', \
    'yiyanghkust/finbert-tone', \
    'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis', \
    'dslim/bert-base-NER' \
]]"

COPY . .

# ── API target ─────────────────────────────────────────────────────────────────
FROM base AS api
EXPOSE 8080
HEALTHCHECK --interval=20s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8080/v1/health || exit 1
CMD ["python", "-m", "src.serving.api"]

# ── Consumer target ────────────────────────────────────────────────────────────
FROM base AS consumer
CMD ["python", "-m", "src.processing.kafka_consumer"]

# ── Dashboard target ───────────────────────────────────────────────────────────
FROM base AS dashboard
EXPOSE 8501
HEALTHCHECK --interval=20s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "src/dashboard/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.fileWatcherType=none"]
