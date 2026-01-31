# Minimal Qwen3-TTS Server
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch with CUDA
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Other Python deps
RUN pip3 install --no-cache-dir \
    transformers \
    accelerate \
    soundfile \
    scipy \
    fastapi \
    "uvicorn[standard]" \
    huggingface_hub

COPY server.py /app/server.py

ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models

EXPOSE 8000

CMD ["python3", "server.py"]
