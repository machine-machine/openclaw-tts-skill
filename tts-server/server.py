"""
Qwen3-TTS Server - Full featured API with logging
Generated with Cerebras assistance ⚡
"""
import os
import io
import time
import logging
import base64
import tempfile
from typing import Optional, Literal, List
from contextlib import asynccontextmanager

import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException, Response, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("qwen_tts")

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("HUGGING_FACE_HUB_TOKEN"))
MODELS_DIR = os.getenv("HF_HOME", "/models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

SPEAKERS = ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"]
LANGUAGES = ["auto", "chinese", "english", "japanese", "korean", "french", "german", "spanish", "portuguese", "russian"]
DEFAULT_SPEAKER = "vivian"

# Model state
custom_voice_model = None
model_load_time = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global custom_voice_model, model_load_time
    
    start = time.time()
    logger.info(f"Starting server, loading model on {DEVICE}...")
    
    if HF_TOKEN:
        from huggingface_hub import login
        login(token=HF_TOKEN)
        logger.info("Logged into HuggingFace")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    try:
        from qwen_tts import Qwen3TTSModel
        
        # Check for local model path first
        local_model_path = os.getenv("LOCAL_MODEL_PATH")
        if local_model_path and os.path.isdir(local_model_path):
            logger.info(f"Loading Qwen3-TTS from local path: {local_model_path}")
            custom_voice_model = Qwen3TTSModel.from_pretrained(
                local_model_path,
                device_map=DEVICE,
                dtype=DTYPE,
            )
        else:
            logger.info("Loading Qwen3-TTS CustomVoice 1.7B from HuggingFace...")
            custom_voice_model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                device_map=DEVICE,
                dtype=DTYPE,
                token=HF_TOKEN,
                cache_dir=MODELS_DIR,
            )
        model_load_time = time.time() - start
        logger.info(f"Model loaded in {model_load_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    
    yield
    
    logger.info("Shutting down...")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

app = FastAPI(
    title="m2 Voice - Qwen3-TTS Server",
    description="Text-to-Speech API with custom voices and style instructions",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    logger.info(f"→ {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        duration = (time.time() - start) * 1000
        response.headers["X-Process-Time"] = f"{duration:.2f}ms"
        logger.info(f"← {request.method} {request.url.path} {response.status_code} ({duration:.2f}ms)")
        return response
    except Exception as e:
        duration = (time.time() - start) * 1000
        logger.error(f"✗ {request.method} {request.url.path} ERROR: {e} ({duration:.2f}ms)")
        raise

# Request models
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str = Field(default=DEFAULT_SPEAKER, description="Speaker name")
    language: str = Field(default="auto")
    instruct: Optional[str] = Field(default=None, description="Style instruction")
    format: Literal["wav", "mp3", "ogg"] = Field(default="wav")

def audio_to_bytes(wav: np.ndarray, sr: int, format: str = "wav") -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, wav, sr, format=format.upper() if format != "mp3" else "WAV")
    buffer.seek(0)
    return buffer.read()

# Endpoints
@app.get("/", tags=["General"])
async def root():
    return {
        "service": "m2 Voice - Qwen3-TTS",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "speakers": "/speakers", 
            "tts": "/v1/audio/speech",
            "test": "/test"
        }
    }

@app.get("/health", tags=["System"])
async def health():
    return JSONResponse({
        "status": "ok" if custom_voice_model else "loading",
        "model_loaded": custom_voice_model is not None,
        "model_load_time_s": model_load_time,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "speakers_count": len(SPEAKERS),
        "version": "1.0.0"
    })

@app.get("/speakers", tags=["Info"])
async def get_speakers():
    return {
        "speakers": SPEAKERS,
        "languages": LANGUAGES,
        "default_speaker": DEFAULT_SPEAKER
    }

@app.get("/test", tags=["Testing"])
async def test_tts():
    """Generate a short test audio"""
    if custom_voice_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        with torch.no_grad():
            wavs, sr = custom_voice_model.generate_custom_voice(
                text="Hello! I am m2, your AI assistant with a voice.",
                language="english",
                speaker=DEFAULT_SPEAKER,
                instruct="Speak in a friendly and warm tone",
                non_streaming_mode=True,
                max_new_tokens=512,
            )
        
        audio_bytes = audio_to_bytes(wavs[0], sr, "wav")
        return StreamingResponse(
            iter([audio_bytes]),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=test.wav"}
        )
    except Exception as e:
        logger.error(f"Test TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/audio/speech", tags=["TTS"])
async def openai_speech(request: TTSRequest):
    """OpenAI-compatible TTS with style instructions"""
    if custom_voice_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    speaker = request.voice.lower().replace(" ", "_")
    if speaker not in SPEAKERS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid speaker '{speaker}'. Available: {', '.join(SPEAKERS)}"
        )
    
    try:
        start = time.time()
        with torch.no_grad():
            wavs, sr = custom_voice_model.generate_custom_voice(
                text=request.text.strip(),
                language=request.language.lower(),
                speaker=speaker,
                instruct=request.instruct.strip() if request.instruct else None,
                non_streaming_mode=True,
                max_new_tokens=2048,
            )
        
        gen_time = time.time() - start
        logger.info(f"Generated {len(request.text)} chars in {gen_time:.2f}s")
        
        audio_bytes = audio_to_bytes(wavs[0], sr, request.format)
        return Response(
            content=audio_bytes,
            media_type=f"audio/{request.format}",
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.format}",
                "X-Generation-Time": f"{gen_time:.2f}s"
            }
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=507, detail="GPU out of memory")
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tts", tags=["TTS"])
async def simple_tts(
    text: str = Query(..., min_length=1),
    voice: str = Query(default=DEFAULT_SPEAKER),
    language: str = Query(default="auto"),
    instruct: Optional[str] = Query(default=None),
):
    """Simple GET endpoint for TTS"""
    request = TTSRequest(text=text, voice=voice, language=language, instruct=instruct)
    return await openai_speech(request)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
