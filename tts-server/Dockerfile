# Qwen3-TTS Server - using pre-built PyTorch image
# v2 - added ffmpeg for opus/mp3 conversion
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Download static ffmpeg with all codecs (includes libopus, libmp3lame, libvorbis)
RUN apt-get update && apt-get install -y wget xz-utils && \
    wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && \
    tar xf ffmpeg-release-amd64-static.tar.xz && \
    mv ffmpeg-*-static/ffmpeg /usr/local/bin/ && \
    mv ffmpeg-*-static/ffprobe /usr/local/bin/ && \
    rm -rf ffmpeg-* && \
    apt-get remove -y wget xz-utils && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    echo "=== FFmpeg encoders ===" && \
    ffmpeg -encoders 2>/dev/null | grep -E "libopus|libmp3lame|libvorbis"

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
# Ensure our static ffmpeg is used (before conda's)
ENV PATH="/usr/local/bin:${PATH}"

EXPOSE 8000

CMD ["python", "server.py"]
