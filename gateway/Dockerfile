FROM python:3.11-slim

WORKDIR /app

# Download static ffmpeg with all codecs (libopus, libmp3lame, libvorbis)
# Using static build instead of apt - more reliable across base images
RUN apt-get update && apt-get install -y --no-install-recommends curl wget xz-utils \
    && wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
    && tar xf ffmpeg-release-amd64-static.tar.xz \
    && mv ffmpeg-*-static/ffmpeg /usr/local/bin/ \
    && mv ffmpeg-*-static/ffprobe /usr/local/bin/ \
    && rm -rf ffmpeg-* \
    && apt-get remove -y wget xz-utils && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && echo "=== FFmpeg version ===" && ffmpeg -version | head -1 \
    && echo "=== Opus/MP3 codecs ===" \
    && ffmpeg -encoders 2>/dev/null | grep -E "libopus|libmp3lame|libvorbis"

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    httpx \
    aiofiles \
    pydantic

COPY main.py .

ENV PATH="/usr/local/bin:${PATH}"

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
