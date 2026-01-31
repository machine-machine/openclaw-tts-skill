# Minimal Qwen3-TTS Server
# Model downloaded at runtime, not build time
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

# Minimal deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python packages (no model download here)
RUN pip3 install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu121 \
    transformers accelerate soundfile scipy \
    fastapi uvicorn[standard] huggingface_hub

COPY server.py /app/server.py

# Model will be in /models volume, downloaded at runtime
ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models

EXPOSE 8000

CMD ["python3", "server.py"]
