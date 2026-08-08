FROM python:3.11-slim

WORKDIR /app

# System deps needed by lightgbm/catboost/arch wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
# The registered/promoted model artifact must already exist — this image
# does NOT train anything, it only serves. Build the model with
# `python -m src.train && python -m src.fit_final` before building, or bind
# mount models/ at runtime.
COPY models/ ./models/
COPY data/processed/daily_features.parquet ./data/processed/daily_features.parquet

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
