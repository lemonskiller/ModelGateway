FROM docker.m.daocloud.io/library/python:3.12-slim

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/local/nvidia/lib64 \
    && touch /usr/local/nvidia/lib64/libcuda.so \
    && touch /usr/local/nvidia/lib64/libcuda.so.1 \
    && touch /usr/local/nvidia/lib64/libnvidia-ml.so \
    && touch /usr/local/nvidia/lib64/libnvidia-ml.so.1

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir \
    "fastapi>=0.115,<1" \
    "httpx>=0.28,<1" \
    "pydantic>=2.9,<3" \
    "PyYAML>=6,<7" \
    "uvicorn[standard]>=0.34,<1"

COPY app ./app
COPY config ./config

ENV MODEL_GATEWAY_CONFIG=/app/config/models.yaml
EXPOSE 18001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "18001"]
