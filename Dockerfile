FROM docker.m.daocloud.io/library/python:3.12-slim

WORKDIR /app

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
