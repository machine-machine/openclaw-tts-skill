import os
import io
import torch
import numpy as np
import uvicorn
from typing import Optional, Literal, Generator
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
import soundfile as sf

# Configuration
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-Coder-0.5B")  # Placeholder - will use actual TTS model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
SAMPLE_RATE = 24000

# Global model variables
model = None
processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor
    print(f"Loading TTS model on {DEVICE}...")
    
    # For now, we'll create a simple placeholder
    # Real Qwen3-TTS integration would go here
    print("TTS Server ready (placeholder mode)")
    
    yield
    
    print("Shutting down TTS server...")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

app = FastAPI(title="Qwen TTS Server", lifespan=lifespan)

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str = Field(default="default")
    format: Literal["mp3", "wav", "ogg"] = Field(default="wav")
    stream: bool = Field(default=False)

def generate_silence(duration_ms: int, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate silence as placeholder"""
    samples = int(sample_rate * duration_ms / 1000)
    return np.zeros(samples, dtype=np.float32)

def text_to_audio_placeholder(text: str) -> np.ndarray:
    """Placeholder - generates silence proportional to text length"""
    # ~100ms per character as rough approximation
    duration_ms = len(text) * 100
    return generate_silence(duration_ms)

def convert_audio(audio_array: np.ndarray, format: str, rate: int) -> bytes:
    """Convert numpy audio to bytes in specified format"""
    buffer = io.BytesIO()
    
    if format == "wav":
        sf.write(buffer, audio_array, rate, format='WAV')
    elif format == "ogg":
        sf.write(buffer, audio_array, rate, format='OGG')
    elif format == "mp3":
        # Fallback to wav if no mp3 encoder
        sf.write(buffer, audio_array, rate, format='WAV')
    
    buffer.seek(0)
    return buffer.read()

def chunk_generator(data: bytes, chunk_size: int = 4096):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "device": DEVICE,
        "mode": "placeholder",
        "note": "Replace with Qwen3-TTS when GPU available"
    }

@app.post("/v1/audio/speech")
async def speech(payload: TTSRequest):
    try:
        # Generate audio (placeholder)
        audio = text_to_audio_placeholder(payload.text)
        audio_bytes = convert_audio(audio, payload.format, SAMPLE_RATE)
        
        media_type = f"audio/{payload.format}"
        
        if payload.stream:
            return StreamingResponse(
                chunk_generator(audio_bytes),
                media_type=media_type
            )
        else:
            return Response(
                content=audio_bytes,
                media_type=media_type,
                headers={"Content-Disposition": f"attachment; filename=speech.{payload.format}"}
            )
            
    except Exception as e:
        print(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
