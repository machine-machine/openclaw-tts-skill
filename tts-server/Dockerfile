# Qwen3-TTS Server - using pre-built PyTorch image
# v2 - added ffmpeg for opus/mp3 conversion
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install ffmpeg with opus/mp3/vorbis codecs
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libopus0 libmp3lame0 libvorbis0a && \
    rm -rf /var/lib/apt/lists/* && \
    echo "=== FFmpeg codecs ===" && \
    ffmpeg -codecs 2>/dev/null | grep -E "opus|mp3|vorbis" || true

# Only install qwen-tts and its deps (torch already included)
RUN pip install --no-cache-dir \
    qwen-tts \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    aiofiles

COPY server.py /app/server.py

ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models

EXPOSE 8000

CMD ["python", "server.py"]
