"""
Qwen3-TTS Server - FastAPI wrapper for Qwen/Qwen3-TTS
"""
import os
import io
import logging
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException, Response, Query
from pydantic import BaseModel, Field
from typing import Optional, Literal
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-TTS")
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("HUGGING_FACE_HUB_TOKEN"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
SAMPLE_RATE = 24000

model = None
processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor
    
    logger.info(f"Loading {MODEL_ID} on {DEVICE} with {DTYPE}...")
    
    try:
        # Login to HF if token provided
        if HF_TOKEN:
            from huggingface_hub import login
            login(token=HF_TOKEN)
            logger.info("Logged into Hugging Face")
        
        # Try loading Qwen3-TTS
        # The model architecture may vary - try different loaders
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
        
        try:
            # Try as speech model
            processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                MODEL_ID,
                torch_dtype=DTYPE,
                device_map="auto" if DEVICE == "cuda" else None,
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"AutoModel failed: {e}, trying alternative...")
            # Fallback: try with specific class if available
            from transformers import pipeline
            model = pipeline("text-to-speech", model=MODEL_ID, device=0 if DEVICE == "cuda" else -1)
            processor = None
        
        if hasattr(model, 'eval'):
            model.eval()
        
        logger.info(f"Model loaded successfully on {DEVICE}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    logger.info("Shutting down...")
    del model
    del processor
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

app = FastAPI(title="Qwen3-TTS Server", lifespan=lifespan)

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str = Field(default="default")
    format: Literal["wav", "mp3", "ogg"] = Field(default="wav")

def synthesize(text: str, voice: str = "default") -> bytes:
    """Generate speech from text"""
    global model, processor
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        with torch.no_grad():
            if processor is not None:
                # Using processor + model
                inputs = processor(text=text, return_tensors="pt")
                if DEVICE == "cuda":
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                
                # Generate audio
                if hasattr(model, 'generate'):
                    output = model.generate(**inputs)
                else:
                    output = model(**inputs)
                
                # Extract audio tensor
                if hasattr(output, 'audio'):
                    audio = output.audio
                elif hasattr(output, 'waveform'):
                    audio = output.waveform
                elif isinstance(output, tuple):
                    audio = output[0]
                else:
                    audio = output
                
                audio_np = audio.cpu().numpy().squeeze()
            else:
                # Using pipeline
                result = model(text)
                audio_np = result["audio"]
                global SAMPLE_RATE
                SAMPLE_RATE = result.get("sampling_rate", SAMPLE_RATE)
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, SAMPLE_RATE, format='WAV')
        buffer.seek(0)
        return buffer.read()
        
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=507, detail="GPU out of memory")
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": model is not None
    }

@app.get("/")
async def root():
    return {"message": "Qwen3-TTS Server", "status": "running"}

@app.post("/v1/audio/speech")
async def openai_speech(request: TTSRequest):
    """OpenAI-compatible TTS endpoint"""
    audio = synthesize(request.text, request.voice)
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"Content-Disposition": f"attachment; filename=speech.wav"}
    )

@app.get("/api/tts")
async def coqui_compatible(
    text: str = Query(..., min_length=1, max_length=5000),
    speaker_id: Optional[str] = None
):
    """Coqui-compatible TTS endpoint"""
    audio = synthesize(text, speaker_id or "default")
    return Response(content=audio, media_type="audio/wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
