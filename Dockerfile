FROM python:3.11-slim

# libgomp is required by the XGBoost wheel
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so code changes do not invalidate this layer
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pipeline/ pipeline/
COPY api/ api/
COPY simulator/ simulator/
COPY dashboard/ dashboard/

# artifacts/ and data/ are bind-mounted at runtime (see docker-compose.yml)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
