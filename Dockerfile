# Serving image only - CPU, no CUDA, no training dependencies.
#
# Training happens on Kaggle (see docs/06_technical_playbook.md); shipping a CUDA base
# image here would add ~6 GB to deploy an inference service that does not need it. The
# behaviour, verification and reporting layers are pure Python + numpy by design, which
# is what makes this image small enough to be worth building.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Dependency layer first so source edits do not invalidate the pip cache.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/run_demo.py ./scripts/

# Non-root: the service reads structured JSON and writes SQLite, nothing more.
RUN useradd --create-home --shell /bin/bash app \
    && mkdir -p /app/data \
    && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u; \
        exit(0 if u.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "behaviorsense.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
