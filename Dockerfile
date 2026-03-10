# Multi-AI-Agent Document System
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

COPY src/ ./src/
COPY README.md LICENSE ./

# Data persisted via volume
ENV DATA_DIR=/data
VOLUME /data

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
