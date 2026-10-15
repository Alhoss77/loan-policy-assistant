FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/model_cache

WORKDIR /app

# System library used by numerical computing packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Our embedding model runs on CPU, so avoid GPU dependencies.
RUN python -m pip install --no-cache-dir \
    torch==2.14.1 \
    --index-url https://download.pytorch.org/whl/cpu

RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy only the application files needed by the server.
COPY ingest.py rag_chat.py calculator.py agent.py api.py ./
COPY data/ ./data/

# Download the embedding model and build the demo document index.
# This step does not call Gemini or require an API key.
RUN python ingest.py

# Use the cached embedding model at runtime.
ENV HF_HUB_OFFLINE=1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]