# Qwen3-TTS Server
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git curl libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch with CUDA
RUN pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Python deps
RUN pip3 install --no-cache-dir \
    transformers>=4.45.0 \
    accelerate \
    soundfile \
    scipy \
    numpy \
    fastapi \
    "uvicorn[standard]" \
    huggingface_hub \
    python-multipart

# Clone and install qwen_tts from HuggingFace space
RUN git clone --depth 1 https://huggingface.co/spaces/Qwen/Qwen3-TTS /tmp/qwen3-tts && \
    pip3 install --no-cache-dir /tmp/qwen3-tts && \
    rm -rf /tmp/qwen3-tts

COPY server.py /app/server.py

ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models

EXPOSE 8000

CMD ["python3", "server.py"]
