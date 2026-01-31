# Qwen3-TTS Server with GPU support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    transformers>=4.45.0 \
    accelerate \
    soundfile \
    scipy \
    fastapi \
    uvicorn[standard] \
    huggingface_hub

# Pre-download model (will use HF token from env)
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
RUN python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')" 2>/dev/null || true

# Copy server
COPY server.py /app/server.py

EXPOSE 8000

CMD ["python", "server.py"]
