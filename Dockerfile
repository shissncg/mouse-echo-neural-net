ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MENN_HOST=0.0.0.0 \
    MENN_PORT=8000 \
    MENN_DATA_ROOT=/data

WORKDIR /app

# System deps for manylinux wheels (opencv/tf) and basic runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libglib2.0-0 \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir uv && uv sync --frozen

EXPOSE 8000

CMD ["uv", "run", "menn-web", "--host", "0.0.0.0", "--port", "8000", "--data-root", "/data"]
