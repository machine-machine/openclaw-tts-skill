# Qwen3-TTS Server - using pre-built PyTorch image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Only install qwen-tts and its deps (torch already included)
RUN pip install --no-cache-dir \
    qwen-tts \
    fastapi \
    "uvicorn[standard]" \
    python-multipart

COPY server.py /app/server.py

ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models

EXPOSE 8000

CMD ["python", "server.py"]
