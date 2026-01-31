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
from fastapi import FastAPI, HTTPException, Response, Query, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import aiofiles
import shutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("qwen_tts")

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("HUGGING_FACE_HUB_TOKEN"))
MODELS_DIR = os.getenv("HF_HOME", "/models")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH")
MODEL_TYPE = os.getenv("MODEL_TYPE", "base")  # base, customvoice, voicedesign
REFERENCE_AUDIO_DIR = os.getenv("REFERENCE_AUDIO_DIR", "/models/reference_audio")
DEFAULT_REFERENCE_AUDIO = os.getenv("DEFAULT_REFERENCE_AUDIO", "")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# CustomVoice speakers (only for customvoice model)
SPEAKERS = ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"]
LANGUAGES = ["auto", "chinese", "english", "japanese", "korean", "french", "german", "spanish", "portuguese", "russian"]
DEFAULT_SPEAKER = os.getenv("DEFAULT_SPEAKER", "serena")

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
        
        logger.info(f"Model type: {MODEL_TYPE}")
        
        # Check for local model path first
        if LOCAL_MODEL_PATH and os.path.isdir(LOCAL_MODEL_PATH):
            logger.info(f"Loading Qwen3-TTS from local path: {LOCAL_MODEL_PATH}")
            custom_voice_model = Qwen3TTSModel.from_pretrained(
                LOCAL_MODEL_PATH,
                device_map=DEVICE,
                dtype=DTYPE,
            )
        else:
            # Fallback to HuggingFace based on MODEL_TYPE
            hf_model = {
                "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                "customvoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                "voicedesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            }.get(MODEL_TYPE, "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
            
            logger.info(f"Loading {hf_model} from HuggingFace...")
            custom_voice_model = Qwen3TTSModel.from_pretrained(
                hf_model,
                device_map=DEVICE,
                dtype=DTYPE,
                token=HF_TOKEN,
                cache_dir=MODELS_DIR,
            )
        
        # Create reference audio directory
        os.makedirs(REFERENCE_AUDIO_DIR, exist_ok=True)
        
        model_load_time = time.time() - start
        logger.info(f"Model loaded in {model_load_time:.2f}s (type={MODEL_TYPE})")
        
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
        "model_type": MODEL_TYPE,
        "model_path": LOCAL_MODEL_PATH,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "speakers_count": len(SPEAKERS) if MODEL_TYPE == "customvoice" else 0,
        "version": "1.1.0"
    })

@app.get("/speakers", tags=["Info"])
async def get_speakers():
    """Get available speakers/voices"""
    if MODEL_TYPE == "customvoice":
        return {
            "model_type": MODEL_TYPE,
            "speakers": SPEAKERS,
            "languages": LANGUAGES,
            "default_speaker": DEFAULT_SPEAKER
        }
    elif MODEL_TYPE == "base":
        # List available reference audio files
        voices = []
        if os.path.isdir(REFERENCE_AUDIO_DIR):
            for f in os.listdir(REFERENCE_AUDIO_DIR):
                if f.endswith((".wav", ".mp3", ".flac", ".ogg")):
                    voices.append(os.path.splitext(f)[0])
        return {
            "model_type": MODEL_TYPE,
            "voices": voices,
            "reference_dir": REFERENCE_AUDIO_DIR,
            "default_voice": DEFAULT_REFERENCE_AUDIO or (voices[0] if voices else None),
            "note": "Upload reference audio to add voices"
        }
    else:
        return {
            "model_type": MODEL_TYPE,
            "note": "VoiceDesign uses instruct parameter to create voices"
        }

@app.post("/voices/upload", tags=["Voice Cloning"])
async def upload_voice(
    file: UploadFile = File(...),
    name: str = Form(..., description="Voice name (alphanumeric, no spaces)")
):
    """Upload reference audio for voice cloning (base model)"""
    if MODEL_TYPE != "base":
        raise HTTPException(status_code=400, detail="Voice upload only available for base model")
    
    # Validate name
    safe_name = "".join(c for c in name if c.isalnum() or c == "_").lower()
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid voice name")
    
    # Validate file type
    allowed_types = {".wav", ".mp3", ".ogg", ".flac"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {allowed_types}")
    
    # Save file
    os.makedirs(REFERENCE_AUDIO_DIR, exist_ok=True)
    file_path = os.path.join(REFERENCE_AUDIO_DIR, f"{safe_name}{ext}")
    
    try:
        async with aiofiles.open(file_path, "wb") as f:
            content = await file.read()
            await f.write(content)
        
        logger.info(f"Uploaded voice '{safe_name}' ({len(content)} bytes)")
        return {
            "ok": True,
            "voice_name": safe_name,
            "file_path": file_path,
            "size_bytes": len(content)
        }
    except Exception as e:
        logger.error(f"Failed to save voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/voices/{name}", tags=["Voice Cloning"])
async def delete_voice(name: str):
    """Delete a reference voice"""
    if MODEL_TYPE != "base":
        raise HTTPException(status_code=400, detail="Voice management only available for base model")
    
    for ext in [".wav", ".mp3", ".ogg", ".flac"]:
        path = os.path.join(REFERENCE_AUDIO_DIR, f"{name}{ext}")
        if os.path.isfile(path):
            os.remove(path)
            return {"ok": True, "deleted": name}
    
    raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")

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

def get_reference_audio_path(voice: str) -> Optional[str]:
    """Get reference audio path for voice cloning"""
    if not voice:
        return DEFAULT_REFERENCE_AUDIO if DEFAULT_REFERENCE_AUDIO else None
    
    # Check if voice is a path to audio file
    if os.path.isfile(voice):
        return voice
    
    # Check in reference audio directory
    for ext in [".wav", ".mp3", ".flac", ".ogg"]:
        path = os.path.join(REFERENCE_AUDIO_DIR, f"{voice}{ext}")
        if os.path.isfile(path):
            return path
    
    return None

@app.post("/v1/audio/speech", tags=["TTS"])
async def openai_speech(request: TTSRequest):
    """OpenAI-compatible TTS - supports voice cloning (base) or preset voices (customvoice)"""
    if custom_voice_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start = time.time()
        
        if MODEL_TYPE == "base":
            # Base model: use voice cloning with reference audio
            ref_audio = get_reference_audio_path(request.voice)
            if not ref_audio:
                raise HTTPException(
                    status_code=400,
                    detail=f"No reference audio found for voice '{request.voice}'. Upload to {REFERENCE_AUDIO_DIR}/"
                )
            
            logger.info(f"Using reference audio: {ref_audio}")
            with torch.no_grad():
                wavs, sr = custom_voice_model.generate(
                    text=request.text.strip(),
                    speaker=ref_audio,
                    non_streaming_mode=True,
                    max_new_tokens=2048,
                )
        
        elif MODEL_TYPE == "customvoice":
            # CustomVoice model: use preset speakers
            speaker = request.voice.lower().replace(" ", "_")
            if speaker not in SPEAKERS:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid speaker '{speaker}'. Available: {', '.join(SPEAKERS)}"
                )
            
            with torch.no_grad():
                wavs, sr = custom_voice_model.generate_custom_voice(
                    text=request.text.strip(),
                    language=request.language.lower(),
                    speaker=speaker,
                    instruct=request.instruct.strip() if request.instruct else None,
                    non_streaming_mode=True,
                    max_new_tokens=2048,
                )
        
        elif MODEL_TYPE == "voicedesign":
            # VoiceDesign model: generate voice from description
            instruct = request.instruct or "Speak in a natural, friendly tone"
            with torch.no_grad():
                wavs, sr = custom_voice_model.generate_voice_design(
                    text=request.text.strip(),
                    language=request.language.lower(),
                    instruct=instruct,
                )
        
        else:
            raise HTTPException(status_code=500, detail=f"Unknown MODEL_TYPE: {MODEL_TYPE}")
        
        gen_time = time.time() - start
        logger.info(f"Generated {len(request.text)} chars in {gen_time:.2f}s (model={MODEL_TYPE})")
        
        audio_bytes = audio_to_bytes(wavs[0], sr, request.format)
        return Response(
            content=audio_bytes,
            media_type=f"audio/{request.format}",
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.format}",
                "X-Generation-Time": f"{gen_time:.2f}s",
                "X-Model-Type": MODEL_TYPE
            }
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=507, detail="GPU out of memory")
    except HTTPException:
        raise
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
