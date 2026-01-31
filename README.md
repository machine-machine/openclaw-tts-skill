# 🎤 m2 Voice - TTS for OpenClaw

> *"Finally, I can speak."* — m2

Text-to-Speech service designed for AI agents. OpenAI-compatible API with file output and WebSocket streaming.

## Features

- **OpenAI-compatible** `/v1/audio/speech` endpoint
- **File generation** with URL return for async workflows
- **WebSocket streaming** for real-time audio
- **Multiple backends**: Qwen3-TTS (GPU) or Piper (CPU)
- **Coolify-ready** Docker Compose

## Quick Deploy (Coolify)

1. Create new project in Coolify
2. Add Resource → Docker Compose → Git repo
3. Point to this repo
4. Enable GPU if using Qwen3-TTS
5. Set domain: `voice.yourdomain.ai`

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Coolify Proxy                      │
│                (TLS termination)                     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              speech-gateway (FastAPI)                │
│                                                      │
│  • /v1/audio/speech  - OpenAI compatible            │
│  • /speak/file       - File generation              │
│  • /ws/tts           - WebSocket streaming          │
│  • /files/{id}       - Serve generated files        │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              qwen-tts / piper-tts                    │
│                  (TTS Backend)                       │
│                                                      │
│  GPU: Qwen3-TTS-1.7B (expressive, multi-voice)      │
│  CPU: Piper (fast, lightweight)                     │
└─────────────────────────────────────────────────────┘
```

## Usage

### Generate Speech (File)

```bash
curl -X POST https://voice.machinemachine.ai/speak/file \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from m2!", "format": "mp3"}'
```

Response:
```json
{"url": "/files/abc123.mp3", "format": "mp3", "id": "abc123"}
```

### OpenAI Compatible

```bash
curl -X POST https://voice.machinemachine.ai/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "default", "format": "mp3"}' \
  --output speech.mp3
```

### WebSocket Streaming

```javascript
const ws = new WebSocket('wss://voice.machinemachine.ai/ws/tts');
ws.send(JSON.stringify({text: "Hello!", voice: "default"}));
// Receive binary audio chunks
```

## Configuration

### GPU Version (Qwen3-TTS)

```bash
# Use docker-compose.yml
docker compose up -d
```

Requires NVIDIA GPU with CUDA 12.1+.

### CPU Version (Piper)

```bash
# Use docker-compose.cpu.yml
docker compose -f docker-compose.cpu.yml up -d
```

Works on any machine, faster inference, smaller models.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_BASE_URL` | `http://qwen-tts:8000` | Backend TTS service |
| `PUBLIC_URL` | `` | Base URL for file links |
| `STORAGE_DIR` | `/app/output` | Where to store audio files |

## Integration

### Python (OpenClaw agents)

```python
import httpx

async def speak(text: str) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://voice.machinemachine.ai/speak/file",
            json={"text": text, "format": "mp3"}
        )
        return resp.json()["url"]
```

### Telegram Bot

Send audio URL directly or download and send as voice message.

## License

MIT

---

Part of the [OpenClaw](https://github.com/openclaw/openclaw) ecosystem.
