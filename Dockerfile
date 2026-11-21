# One container = website + API. Works on Hugging Face Spaces (Docker SDK), Render, Fly.io, Railway, any VM.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-api.txt
COPY api ./api
COPY src ./src
COPY models ./models
COPY web ./web
COPY data/reference ./data/reference
COPY data/demo ./data/demo
ENV PORT=7860 PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["sh", "-c", "uvicorn api.server:app --host 0.0.0.0 --port ${PORT}"]
