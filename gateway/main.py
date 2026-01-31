import os
import uuid
import logging
import aiofiles
import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Literal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://qwen-tts:8000")
PUBLIC_URL = os.getenv("PUBLIC_URL", "")
STORAGE_DIR = os.getenv("STORAGE_DIR", "/app/output")
os.makedirs(STORAGE_DIR, exist_ok=True)

app = FastAPI(title="m2 Voice Gateway", version="1.0.0")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class SpeechRequest(BaseModel):
    text: str
    voice: str = "default"
    format: Literal["mp3", "wav", "ogg"] = "mp3"
    stream: bool = False

class SpeakFileRequest(BaseModel):
    text: str
    voice: str = "default"
    format: Literal["mp3", "wav", "ogg"] = "mp3"

# HTTP Client for Backend Proxy
client = httpx.AsyncClient(timeout=120.0)

@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        resp = await client.get(f"{TTS_BASE_URL}/health", timeout=5.0)
        backend_status = "healthy" if resp.status_code == 200 else "degraded"
    except:
        backend_status = "unavailable"
    
    return {
        "status": "healthy",
        "backend": TTS_BASE_URL,
        "backend_status": backend_status
    }

@app.post("/v1/audio/speech")
async def openai_speech(request: SpeechRequest):
    """OpenAI compatible TTS endpoint"""
    backend_url = f"{TTS_BASE_URL}/v1/audio/speech"
    
    try:
        payload = request.dict()
        
        if request.stream:
            async def generate():
                async with client.stream("POST", backend_url, json=payload) as resp:
                    if resp.status_code != 200:
                        error_detail = await resp.aread()
                        raise HTTPException(status_code=resp.status_code, detail=error_detail.decode())
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            
            return StreamingResponse(generate(), media_type=f"audio/{request.format}")
        else:
            resp = await client.post(backend_url, json=payload)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            
            return StreamingResponse(
                iter([resp.content]),
                media_type=f"audio/{request.format}",
                headers={"Content-Disposition": f"attachment; filename=speech.{request.format}"}
            )

    except httpx.RequestError as e:
        logger.error(f"Backend connection error: {e}")
        raise HTTPException(status_code=503, detail="Backend service unavailable")

@app.post("/speak/file")
async def speak_file(request: SpeakFileRequest):
    """Generate audio file, save locally, return URL"""
    backend_url = f"{TTS_BASE_URL}/v1/audio/speech"
    
    try:
        payload = {"text": request.text, "voice": request.voice, "format": request.format, "stream": False}
        
        resp = await client.post(backend_url, json=payload)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        file_id = f"{uuid.uuid4()}.{request.format}"
        file_path = os.path.join(STORAGE_DIR, file_id)
        
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(resp.content)

        base_url = PUBLIC_URL.rstrip('/') if PUBLIC_URL else ""
        file_url = f"{base_url}/files/{file_id}"
        
        return {"url": file_url, "format": request.format, "id": file_id}

    except httpx.RequestError as e:
        logger.error(f"Backend connection error: {e}")
        raise HTTPException(status_code=503, detail="Backend service unavailable")

@app.get("/files/{filename}")
async def get_file(filename: str):
    """Serve generated audio files"""
    # Security: only allow alphanumeric + dots
    if not filename.replace(".", "").replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    file_path = os.path.join(STORAGE_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)

@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    """WebSocket for streaming audio - send JSON, receive binary chunks"""
    await websocket.accept()
    backend_url = f"{TTS_BASE_URL}/v1/audio/speech"
    
    try:
        while True:
            data = await websocket.receive_json()
            text = data.get("text")
            voice = data.get("voice", "default")
            format = data.get("format", "mp3")
            
            if not text:
                await websocket.send_json({"error": "Missing 'text' field"})
                continue

            payload = {"text": text, "voice": voice, "format": format, "stream": True}
            
            try:
                async with client.stream("POST", backend_url, json=payload) as resp:
                    if resp.status_code != 200:
                        error_msg = await resp.aread()
                        await websocket.send_json({"error": f"Backend error: {error_msg.decode()}"})
                        continue
                    
                    async for chunk in resp.aiter_bytes():
                        await websocket.send_bytes(chunk)
                        
                    await websocket.send_json({"status": "done"})
                    
            except httpx.RequestError as e:
                logger.error(f"WS Backend error: {e}")
                await websocket.send_json({"error": "Backend unavailable"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
