# Qwen3-TTS Server
# GPU required - uses CUDA
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Clone Qwen3-TTS
RUN git clone --depth 1 https://github.com/QwenLM/Qwen3-TTS.git /app/qwen3-tts

WORKDIR /app/qwen3-tts

# Install requirements
RUN pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir transformers accelerate soundfile fastapi uvicorn

# Create API wrapper
COPY server.py /app/server.py

WORKDIR /app

EXPOSE 8000

CMD ["python3", "server.py"]
