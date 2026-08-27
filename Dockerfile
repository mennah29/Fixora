FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ANONYMIZED_TELEMETRY=False \
    SENTENCE_TRANSFORMERS_HOME=/opt/model-cache \
    USE_TF=0 \
    USE_TORCH=1

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --system fixora && useradd --system --gid fixora --create-home fixora
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Fetch the embedding model while building, avoiding an unpredictable download at startup.
ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

COPY app ./app
COPY scripts ./scripts
RUN mkdir -p /app/data && chown -R fixora:fixora /app /opt/model-cache
USER fixora

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
