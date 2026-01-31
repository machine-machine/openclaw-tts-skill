import os
import uuid
import logging
import io
import aiofiles
import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://coqui-tts:5002")
TTS_TYPE = os.getenv("TTS_TYPE", "coqui")  # coqui, openai, piper
PUBLIC_URL = os.getenv("PUBLIC_URL", "")
STORAGE_DIR = os.getenv("STORAGE_DIR", "/app/output")
os.makedirs(STORAGE_DIR, exist_ok=True)

app = FastAPI(title="m2 Voice Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SpeechRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str = Field(default="default")
    format: Literal["mp3", "wav", "ogg"] = Field(default="wav")
    stream: bool = Field(default=False)

class SpeakFileRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str = Field(default="default")
    format: Literal["mp3", "wav", "ogg"] = Field(default="wav")

client = httpx.AsyncClient(timeout=120.0)

@app.on_event("shutdown")
async def shutdown():
    await client.aclose()

async def synthesize_coqui(text: str, speaker_id: str = None) -> bytes:
    """Call Coqui TTS API"""
    params = {"text": text}
    if speaker_id:
        params["speaker_id"] = speaker_id
    
    resp = await client.get(f"{TTS_BASE_URL}/api/tts", params=params)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"TTS error: {resp.text}")
    return resp.content

async def synthesize_openai(text: str, voice: str, format: str) -> bytes:
    """Call OpenAI-compatible TTS API"""
    resp = await client.post(
        f"{TTS_BASE_URL}/v1/audio/speech",
        json={"text": text, "voice": voice, "format": format}
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"TTS error: {resp.text}")
    return resp.content

async def synthesize(text: str, voice: str = "default", format: str = "wav") -> bytes:
    """Route to appropriate TTS backend"""
    if TTS_TYPE == "coqui":
        return await synthesize_coqui(text, voice if voice != "default" else None)
    elif TTS_TYPE == "openai":
        return await synthesize_openai(text, voice, format)
    else:
        raise HTTPException(status_code=500, detail=f"Unknown TTS type: {TTS_TYPE}")

@app.get("/health")
async def health():
    try:
        if TTS_TYPE == "coqui":
            resp = await client.get(f"{TTS_BASE_URL}/", timeout=5.0)
            backend_ok = resp.status_code == 200
        else:
            resp = await client.get(f"{TTS_BASE_URL}/health", timeout=5.0)
            backend_ok = resp.status_code == 200
    except:
        backend_ok = False
    
    return {
        "status": "healthy",
        "backend": TTS_BASE_URL,
        "backend_type": TTS_TYPE,
        "backend_status": "ok" if backend_ok else "unavailable"
    }

@app.post("/v1/audio/speech")
async def openai_speech(request: SpeechRequest):
    """OpenAI-compatible TTS endpoint"""
    try:
        audio = await synthesize(request.text, request.voice, request.format)
        return Response(
            content=audio,
            media_type=f"audio/{request.format}",
            headers={"Content-Disposition": f"attachment; filename=speech.{request.format}"}
        )
    except httpx.RequestError as e:
        logger.error(f"Backend error: {e}")
        raise HTTPException(status_code=503, detail="TTS backend unavailable")

@app.post("/speak/file")
async def speak_file(request: SpeakFileRequest):
    """Generate audio file and return URL"""
    try:
        audio = await synthesize(request.text, request.voice, request.format)
        
        file_id = f"{uuid.uuid4()}.{request.format}"
        file_path = os.path.join(STORAGE_DIR, file_id)
        
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(audio)
        
        base_url = PUBLIC_URL.rstrip('/') if PUBLIC_URL else ""
        return {
            "url": f"{base_url}/files/{file_id}",
            "format": request.format,
            "id": file_id
        }
    except httpx.RequestError as e:
        logger.error(f"Backend error: {e}")
        raise HTTPException(status_code=503, detail="TTS backend unavailable")

@app.get("/files/{filename}")
async def get_file(filename: str):
    """Serve generated audio files"""
    if not filename.replace(".", "").replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    file_path = os.path.join(STORAGE_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)

@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    """WebSocket for real-time TTS"""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            text = data.get("text")
            voice = data.get("voice", "default")
            format = data.get("format", "wav")
            
            if not text:
                await websocket.send_json({"error": "Missing 'text' field"})
                continue
            
            try:
                audio = await synthesize(text, voice, format)
                await websocket.send_bytes(audio)
                await websocket.send_json({"status": "done", "bytes": len(audio)})
            except Exception as e:
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
