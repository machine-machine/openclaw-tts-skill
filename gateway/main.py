import os
import re
import uuid
import logging
import io
import asyncio
import subprocess
import tempfile
import aiofiles
import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verify ffmpeg is available at startup
def _check_ffmpeg() -> str:
    """Check ffmpeg availability and return version string."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        if result.returncode == 0:
            version_line = result.stdout.decode().split("\n")[0]
            logger.info(f"ffmpeg available: {version_line}")
            return version_line
        else:
            logger.error("ffmpeg found but returned non-zero exit code")
            return "error"
    except FileNotFoundError:
        logger.error("ffmpeg NOT FOUND - audio concatenation will fail for chunked requests!")
        return "missing"
    except Exception as e:
        logger.error(f"ffmpeg check failed: {e}")
        return f"error: {e}"

FFMPEG_VERSION = _check_ffmpeg()

# Chunking config
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "200"))  # Max chars per chunk
CHUNK_ENABLED = os.getenv("CHUNK_ENABLED", "true").lower() in ("true", "1", "yes")

# Configuration
TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://coqui-tts:5002")
TTS_TYPE = os.getenv("TTS_TYPE", "coqui")  # coqui, openai, piper
PUBLIC_URL = os.getenv("PUBLIC_URL", "")
STORAGE_DIR = os.getenv("STORAGE_DIR", "/app/output")
# Force all requests to use this voice (for voice cloning). Empty = use requested voice.
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "vivian")
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
    # Accept both 'text' (our format) and 'input' (OpenAI format)
    text: str = Field(default=None, min_length=1, max_length=5000)
    input: str = Field(default=None, min_length=1, max_length=5000)  # OpenAI compat
    voice: str = Field(default="v")
    format: Literal["mp3", "wav", "ogg", "opus"] = Field(default="wav")
    response_format: str = Field(default=None)  # OpenAI compat (maps to format)
    model: str = Field(default=None)  # OpenAI compat (ignored)
    stream: bool = Field(default=False)
    # Style/instruction support
    language: str = Field(default="auto")
    instruct: str = Field(default=None, description="Voice style instruction")
    style: str = Field(default=None, description="Alias for instruct")
    
    def get_text(self) -> str:
        """Get text from either field"""
        return self.text or self.input or ""
    
    def get_format(self) -> str:
        """Get format, preferring response_format if set"""
        if self.response_format:
            return self.response_format.replace("audio/", "")
        return self.format
    
    def get_instruct(self) -> str:
        """Get style instruction from either field"""
        return self.instruct or self.style

class SpeakFileRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str = Field(default="default")
    format: Literal["mp3", "wav", "ogg", "opus"] = Field(default="wav")

client = httpx.AsyncClient(timeout=120.0)


def split_text_into_chunks(text: str, max_chars: int = CHUNK_MAX_CHARS) -> List[str]:
    """Split text into chunks at sentence boundaries, respecting max_chars.
    
    Strategy:
    1. Split on sentence boundaries (. ! ? followed by space or end)
    2. If a sentence is still too long, split on clause boundaries (, ; : —)
    3. Last resort: split on word boundaries
    """
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    remaining = text.strip()
    
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break
        
        # Try to find a sentence boundary within max_chars
        best_split = -1
        
        # Priority 1: Sentence endings (. ! ? followed by space)
        for m in re.finditer(r'[.!?]+[\s]', remaining[:max_chars + 10]):
            pos = m.end()
            if pos <= max_chars:
                best_split = pos
        
        # Priority 2: Clause boundaries (, ; : — followed by space)
        if best_split == -1:
            for m in re.finditer(r'[,;:\u2014]\s', remaining[:max_chars + 5]):
                pos = m.end()
                if pos <= max_chars:
                    best_split = pos
        
        # Priority 3: Last space before max_chars
        if best_split == -1:
            space_pos = remaining.rfind(' ', 0, max_chars)
            if space_pos > 0:
                best_split = space_pos + 1
        
        # Last resort: hard split at max_chars
        if best_split == -1:
            best_split = max_chars
        
        chunk = remaining[:best_split].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[best_split:].strip()
    
    return [c for c in chunks if c]


def concatenate_audio_ffmpeg(audio_chunks: List[bytes], output_format: str) -> bytes:
    """Concatenate audio chunks using ffmpeg concat demuxer."""
    with tempfile.TemporaryDirectory(prefix="tts_concat_") as tmpdir:
        # Write each chunk to a file
        input_files = []
        for i, chunk in enumerate(audio_chunks):
            fpath = os.path.join(tmpdir, f"chunk_{i:04d}.{output_format}")
            with open(fpath, "wb") as f:
                f.write(chunk)
            input_files.append(fpath)
        
        # Create concat list file
        list_path = os.path.join(tmpdir, "concat.txt")
        with open(list_path, "w") as f:
            for fpath in input_files:
                f.write(f"file '{fpath}'\n")
        
        # Output file
        out_path = os.path.join(tmpdir, f"output.{output_format}")
        
        # Run ffmpeg concat
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",  # No re-encoding, just concatenate
            out_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        
        if result.returncode != 0:
            # If copy fails (incompatible streams), try with re-encoding
            logger.warning(f"ffmpeg concat copy failed, re-encoding: {result.stderr.decode()[-200:]}")
            if output_format == "opus":
                cmd = [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", list_path,
                    "-c:a", "libopus", "-b:a", "64k", "-ar", "24000",
                    out_path
                ]
            elif output_format == "mp3":
                cmd = [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", list_path,
                    "-c:a", "libmp3lame", "-b:a", "128k",
                    out_path
                ]
            else:
                cmd = [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", list_path,
                    out_path
                ]
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg concatenation failed: {result.stderr.decode()[-300:]}")
        
        with open(out_path, "rb") as f:
            return f.read()

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

async def synthesize_openai_single(text: str, voice: str, format: str, language: str = "auto", instruct: str = None) -> bytes:
    """Call OpenAI-compatible TTS API for a single chunk."""
    payload = {"text": text, "voice": voice, "format": format, "language": language}
    if instruct:
        payload["instruct"] = instruct
    
    resp = await client.post(
        f"{TTS_BASE_URL}/v1/audio/speech",
        json=payload
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"TTS error: {resp.text}")
    return resp.content


async def synthesize_openai(text: str, voice: str, format: str, language: str = "auto", instruct: str = None) -> bytes:
    """Call OpenAI-compatible TTS API with automatic chunking for long texts."""
    
    # Short texts: no chunking needed
    if not CHUNK_ENABLED or len(text) <= CHUNK_MAX_CHARS:
        logger.info(f"TTS single: format={format}, voice={voice}, text_len={len(text)}")
        return await synthesize_openai_single(text, voice, format, language, instruct)
    
    # Long texts: split into chunks and generate each
    chunks = split_text_into_chunks(text, CHUNK_MAX_CHARS)
    logger.info(f"TTS chunked: format={format}, voice={voice}, text_len={len(text)}, chunks={len(chunks)}")
    
    if len(chunks) == 1:
        return await synthesize_openai_single(chunks[0], voice, format, language, instruct)
    
    # Generate all chunks (sequentially to avoid GPU contention)
    audio_chunks = []
    for i, chunk in enumerate(chunks):
        logger.info(f"  Chunk {i+1}/{len(chunks)}: {len(chunk)} chars")
        try:
            audio = await synthesize_openai_single(chunk, voice, format, language, instruct)
            audio_chunks.append(audio)
        except Exception as e:
            logger.error(f"  Chunk {i+1} failed: {e}")
            raise
    
    # Concatenate audio chunks
    logger.info(f"Concatenating {len(audio_chunks)} audio chunks...")
    try:
        result = concatenate_audio_ffmpeg(audio_chunks, format)
        logger.info(f"Concatenation done: {len(result)} bytes")
        return result
    except Exception as e:
        logger.error(f"Concatenation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Audio concatenation failed: {e}")

async def synthesize(text: str, voice: str = "default", format: str = "wav", language: str = "auto", instruct: str = None) -> bytes:
    """Route to appropriate TTS backend"""
    # Override voice with DEFAULT_VOICE if set (for voice cloning)
    if DEFAULT_VOICE:
        voice = DEFAULT_VOICE
        logger.info(f"Using default voice override: {voice}")
    
    if TTS_TYPE == "coqui":
        return await synthesize_coqui(text, voice if voice != "default" else None)
    elif TTS_TYPE == "openai":
        return await synthesize_openai(text, voice, format, language, instruct)
    else:
        raise HTTPException(status_code=500, detail=f"Unknown TTS type: {TTS_TYPE}")

@app.get("/health")
async def health():
    backend_health = None
    try:
        if TTS_TYPE == "coqui":
            resp = await client.get(f"{TTS_BASE_URL}/", timeout=5.0)
            backend_ok = resp.status_code == 200
        else:
            resp = await client.get(f"{TTS_BASE_URL}/health", timeout=5.0)
            backend_ok = resp.status_code == 200
            if backend_ok:
                backend_health = resp.json()
    except:
        backend_ok = False
    
    return {
        "status": "healthy",
        "backend": TTS_BASE_URL,
        "backend_type": TTS_TYPE,
        "backend_status": "ok" if backend_ok else "unavailable",
        "backend_health": backend_health,
        "ffmpeg": FFMPEG_VERSION,
        "chunking": {"enabled": CHUNK_ENABLED, "max_chars": CHUNK_MAX_CHARS},
        "gateway_version": "1.2.0"
    }

@app.post("/v1/audio/speech")
async def openai_speech(request: SpeechRequest):
    """OpenAI-compatible TTS endpoint with style instructions"""
    text = request.get_text()
    if not text:
        raise HTTPException(status_code=422, detail="Either 'text' or 'input' field required")
    
    fmt = request.get_format()
    instruct = request.get_instruct()
    logger.info(f"TTS request: voice={request.voice}, format={fmt}, lang={request.language}, instruct={bool(instruct)}, text_len={len(text)}")
    
    try:
        audio = await synthesize(text, request.voice, fmt, request.language, instruct)
        return Response(
            content=audio,
            media_type=f"audio/{fmt}",
            headers={"Content-Disposition": f"attachment; filename=speech.{fmt}"}
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

@app.post("/voices/upload")
async def upload_voice(file: bytes = None, name: str = None):
    """Proxy voice upload to backend - use multipart form"""
    from fastapi import UploadFile, File, Form, Request
    raise HTTPException(status_code=400, detail="Use form-data with 'file' and 'name' fields")

@app.post("/voices")
async def upload_voice_form(request: Request):
    """Proxy voice upload to backend via form-data"""
    # Forward the raw request body to backend
    body = await request.body()
    headers = {"content-type": request.headers.get("content-type", "")}
    
    async with httpx.AsyncClient() as c:
        resp = await c.post(
            f"{TTS_BASE_URL}/voices/upload",
            content=body,
            headers=headers,
            timeout=60.0
        )
        return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")

@app.delete("/voices/{name}")
async def delete_voice(name: str):
    """Proxy voice deletion to backend"""
    async with httpx.AsyncClient() as c:
        resp = await c.delete(f"{TTS_BASE_URL}/voices/{name}", timeout=30.0)
        return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")

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
    port = int(os.getenv("PORT", "80"))
    uvicorn.run(app, host="0.0.0.0", port=port)
