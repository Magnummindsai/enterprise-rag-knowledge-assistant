FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app
RUN useradd --create-home --uid 1000 appuser

# CPU-only PyTorch keeps the image ~1.5 GB smaller than the default CUDA build.
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY app ./app
COPY data ./data
COPY scripts ./scripts

# Optionally bake the BGE reranker into the image: docker compose build --build-arg PRELOAD_RERANKER=true
ARG PRELOAD_RERANKER=false
RUN if [ "$PRELOAD_RERANKER" = "true" ]; then python scripts/download_models.py; fi

RUN mkdir -p storage .cache && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
