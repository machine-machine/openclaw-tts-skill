"""
Qwen3-TTS Server - Model downloaded at runtime to /models volume
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
MODELS_DIR = os.getenv("HF_HOME", "/models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
SAMPLE_RATE = 24000

model = None
processor = None

def download_model_if_needed():
    """Download model to /models on first run"""
    logger.info(f"Checking for model in {MODELS_DIR}...")
    
    if HF_TOKEN:
        from huggingface_hub import login
        login(token=HF_TOKEN)
        logger.info("Logged into Hugging Face")
    
    # Model will be cached in HF_HOME=/models
    os.makedirs(MODELS_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor
    
    download_model_if_needed()
    
    logger.info(f"Loading {MODEL_ID} on {DEVICE}...")
    
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        # Try loading - will download if not in cache
        processor = AutoProcessor.from_pretrained(
            MODEL_ID, 
            trust_remote_code=True,
            cache_dir=MODELS_DIR
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            device_map="auto" if DEVICE == "cuda" else None,
            trust_remote_code=True,
            cache_dir=MODELS_DIR
        )
        model.eval()
        logger.info(f"Model loaded on {DEVICE}")
        
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        logger.info("Trying pipeline fallback...")
        try:
            from transformers import pipeline
            model = pipeline(
                "text-to-speech", 
                model=MODEL_ID, 
                device=0 if DEVICE == "cuda" else -1,
                model_kwargs={"cache_dir": MODELS_DIR}
            )
            processor = None
            logger.info("Pipeline loaded")
        except Exception as e2:
            logger.error(f"Pipeline also failed: {e2}")
            # Continue anyway - will fail on synthesis
    
    yield
    
    logger.info("Shutting down...")
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
                inputs = processor(text=text, return_tensors="pt")
                if DEVICE == "cuda":
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                
                output = model.generate(**inputs) if hasattr(model, 'generate') else model(**inputs)
                
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
                result = model(text)
                audio_np = result["audio"]
                global SAMPLE_RATE
                SAMPLE_RATE = result.get("sampling_rate", SAMPLE_RATE)
        
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, SAMPLE_RATE, format='WAV')
        buffer.seek(0)
        return buffer.read()
        
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=507, detail="GPU OOM")
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "ok" if model else "loading",
        "model": MODEL_ID,
        "device": DEVICE,
        "cuda": torch.cuda.is_available(),
        "models_dir": MODELS_DIR
    }

@app.get("/")
async def root():
    return {"service": "Qwen3-TTS", "status": "running"}

@app.post("/v1/audio/speech")
async def openai_speech(request: TTSRequest):
    audio = synthesize(request.text, request.voice)
    return Response(content=audio, media_type="audio/wav")

@app.get("/api/tts")
async def coqui_compat(text: str = Query(...), speaker_id: Optional[str] = None):
    audio = synthesize(text, speaker_id or "default")
    return Response(content=audio, media_type="audio/wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
