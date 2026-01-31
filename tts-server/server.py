"""
Qwen3-TTS Server - Full featured API
Supports: VoiceDesign, VoiceClone, CustomVoice with Style Instructions
"""
import os
import io
import logging
import base64
import tempfile
from typing import Optional, Literal, List
from contextlib import asynccontextmanager

import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException, Response, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("HUGGING_FACE_HUB_TOKEN"))
MODELS_DIR = os.getenv("HF_HOME", "/models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Available speakers for CustomVoice
SPEAKERS = ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"]
LANGUAGES = ["auto", "chinese", "english", "japanese", "korean", "french", "german", "spanish", "portuguese", "russian"]

# Models
voice_design_model = None
base_model = None
custom_voice_model = None

def normalize_audio(wav, clip=True):
    """Normalize audio to float32 [-1, 1]"""
    x = np.asarray(wav)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        y = x.astype(np.float32) / max(abs(info.min), info.max)
    else:
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0:
            y = y / m
    if clip:
        y = np.clip(y, -1.0, 1.0)
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)
    return y

@asynccontextmanager
async def lifespan(app: FastAPI):
    global voice_design_model, base_model, custom_voice_model
    
    if HF_TOKEN:
        from huggingface_hub import login
        login(token=HF_TOKEN)
        logger.info("Logged into HuggingFace")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    try:
        from qwen_tts import Qwen3TTSModel
        
        # Load CustomVoice model (most commonly used)
        logger.info("Loading Qwen3-TTS CustomVoice 1.7B...")
        custom_voice_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            device_map=DEVICE,
            dtype=DTYPE,
            token=HF_TOKEN,
            cache_dir=MODELS_DIR,
        )
        logger.info("CustomVoice model loaded!")
        
        # Optionally load other models (can be lazy-loaded later)
        # voice_design_model = ...
        # base_model = ...
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Continue - will error on requests
    
    yield
    
    logger.info("Shutting down...")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

app = FastAPI(title="Qwen3-TTS Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str = Field(default="vivian", description="Speaker name")
    language: str = Field(default="auto")
    instruct: Optional[str] = Field(default=None, description="Style instruction")
    format: Literal["wav", "mp3", "ogg"] = Field(default="wav")

class VoiceDesignRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice_description: str = Field(..., description="Describe the voice you want")
    language: str = Field(default="auto")
    format: Literal["wav", "mp3", "ogg"] = Field(default="wav")

class VoiceCloneRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    ref_audio_base64: str = Field(..., description="Base64 encoded reference audio")
    ref_text: Optional[str] = Field(default=None, description="Text spoken in reference audio")
    language: str = Field(default="auto")
    x_vector_only: bool = Field(default=False)
    format: Literal["wav", "mp3", "ogg"] = Field(default="wav")

def audio_to_bytes(wav: np.ndarray, sr: int, format: str = "wav") -> bytes:
    """Convert numpy audio to bytes"""
    buffer = io.BytesIO()
    sf.write(buffer, wav, sr, format=format.upper() if format != "mp3" else "WAV")
    buffer.seek(0)
    return buffer.read()

@app.get("/health")
async def health():
    return {
        "status": "ok" if custom_voice_model else "loading",
        "device": DEVICE,
        "cuda": torch.cuda.is_available(),
        "models": {
            "custom_voice": custom_voice_model is not None,
            "voice_design": voice_design_model is not None,
            "base": base_model is not None,
        },
        "speakers": SPEAKERS,
        "languages": LANGUAGES,
    }

@app.get("/")
async def root():
    return {"service": "Qwen3-TTS", "version": "1.0", "status": "running"}

@app.get("/speakers")
async def get_speakers():
    return {"speakers": SPEAKERS, "languages": LANGUAGES}

@app.post("/v1/audio/speech")
async def openai_speech(request: TTSRequest):
    """OpenAI-compatible TTS with Qwen3-TTS CustomVoice"""
    if custom_voice_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    speaker = request.voice.lower().replace(" ", "_")
    if speaker not in SPEAKERS:
        speaker = "vivian"  # Default
    
    try:
        with torch.no_grad():
            wavs, sr = custom_voice_model.generate_custom_voice(
                text=request.text.strip(),
                language=request.language.lower(),
                speaker=speaker,
                instruct=request.instruct.strip() if request.instruct else None,
                non_streaming_mode=True,
                max_new_tokens=2048,
            )
        
        audio_bytes = audio_to_bytes(wavs[0], sr, request.format)
        return Response(
            content=audio_bytes,
            media_type=f"audio/{request.format}",
            headers={"Content-Disposition": f"attachment; filename=speech.{request.format}"}
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=507, detail="GPU OOM")
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tts")
async def simple_tts(
    text: str = Query(..., min_length=1),
    voice: str = Query(default="vivian"),
    language: str = Query(default="auto"),
    instruct: Optional[str] = Query(default=None),
):
    """Simple GET endpoint for TTS"""
    request = TTSRequest(text=text, voice=voice, language=language, instruct=instruct)
    return await openai_speech(request)

@app.post("/voice-design")
async def voice_design(request: VoiceDesignRequest):
    """Generate speech with custom voice from description"""
    global voice_design_model
    
    # Lazy load voice design model
    if voice_design_model is None:
        try:
            from qwen_tts import Qwen3TTSModel
            logger.info("Loading VoiceDesign model...")
            voice_design_model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                device_map=DEVICE,
                dtype=DTYPE,
                token=HF_TOKEN,
                cache_dir=MODELS_DIR,
            )
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Failed to load model: {e}")
    
    try:
        with torch.no_grad():
            wavs, sr = voice_design_model.generate_voice_design(
                text=request.text.strip(),
                language=request.language.lower(),
                instruct=request.voice_description.strip(),
                non_streaming_mode=True,
                max_new_tokens=2048,
            )
        
        audio_bytes = audio_to_bytes(wavs[0], sr, request.format)
        return Response(content=audio_bytes, media_type=f"audio/{request.format}")
    except Exception as e:
        logger.error(f"Voice design error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice-clone")
async def voice_clone(request: VoiceCloneRequest):
    """Clone voice from reference audio"""
    global base_model
    
    # Lazy load base model
    if base_model is None:
        try:
            from qwen_tts import Qwen3TTSModel
            logger.info("Loading Base model for voice cloning...")
            base_model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                device_map=DEVICE,
                dtype=DTYPE,
                token=HF_TOKEN,
                cache_dir=MODELS_DIR,
            )
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Failed to load model: {e}")
    
    # Decode reference audio
    try:
        audio_data = base64.b64decode(request.ref_audio_base64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name
        
        ref_wav, ref_sr = sf.read(temp_path)
        ref_wav = normalize_audio(ref_wav)
        ref_audio = (ref_wav, ref_sr)
        os.unlink(temp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio: {e}")
    
    try:
        with torch.no_grad():
            wavs, sr = base_model.generate_voice_clone(
                text=request.text.strip(),
                language=request.language.lower(),
                ref_audio=ref_audio,
                ref_text=request.ref_text.strip() if request.ref_text else None,
                x_vector_only_mode=request.x_vector_only,
                max_new_tokens=2048,
            )
        
        audio_bytes = audio_to_bytes(wavs[0], sr, request.format)
        return Response(content=audio_bytes, media_type=f"audio/{request.format}")
    except Exception as e:
        logger.error(f"Voice clone error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
