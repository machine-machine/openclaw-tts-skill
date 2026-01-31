# Qwen3-TTS Server - using pre-built PyTorch image
# v2 - added ffmpeg for opus/mp3 conversion
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install ffmpeg with full codec support
# libavcodec-extra includes non-free codecs like libopus, libmp3lame, libvorbis
RUN apt-get update && \
    apt-get install -y ffmpeg libavcodec-extra58 && \
    rm -rf /var/lib/apt/lists/* && \
    echo "=== FFmpeg encoders ===" && \
    ffmpeg -encoders 2>/dev/null | grep -E "libopus|libmp3lame|libvorbis" || echo "Warning: Some codecs missing"

# Install qwen-tts and deps (torch already included)
# flash-attn is optional but improves performance - may fail on some systems
RUN pip install --no-cache-dir \
    qwen-tts \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    aiofiles && \
    pip install --no-cache-dir flash-attn --no-build-isolation || echo "flash-attn install failed (optional)"

COPY server.py /app/server.py

ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models

EXPOSE 8000

CMD ["python", "server.py"]
