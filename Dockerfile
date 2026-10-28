FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/masmur71/insurance-fraud-detection-ml-pipeline" \
      org.opencontainers.image.description="Synthetic insurance claim review API"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    MODEL_PATH=/app/models/model.joblib PYTHONPATH=/app/src
WORKDIR /app
COPY requirements-serving.txt ./
RUN pip install --no-cache-dir -r requirements-serving.txt \
    && groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app
COPY src/ ./src/
COPY models/phase-3/model.joblib ./models/model.joblib
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "claim_fraud.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
