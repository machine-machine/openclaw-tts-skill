# m2 Voice Skill

Text-to-Speech service for OpenClaw agents.

## Quick Start

```bash
# Generate speech (file mode)
curl -X POST https://voice.machinemachine.ai/speak/file \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, I am m2!", "voice": "default", "format": "mp3"}'
# Returns: {"url": "/files/abc123.mp3", "format": "mp3", "id": "abc123"}

# OpenAI-compatible endpoint
curl -X POST https://voice.machinemachine.ai/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "default", "format": "mp3"}' \
  --output speech.mp3
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/audio/speech` | POST | OpenAI-compatible TTS |
| `/speak/file` | POST | Generate file, return URL |
| `/ws/tts` | WebSocket | Streaming audio |
| `/files/{id}` | GET | Download generated file |

## Request Format

```json
{
  "text": "Text to speak",
  "voice": "default",
  "format": "mp3",
  "stream": false
}
```

## Voices

Depends on backend:
- **Qwen3-TTS**: Expressive, multi-voice (GPU required)
- **Piper**: Fast, lightweight (CPU-friendly)

## Integration with OpenClaw

```python
# In your agent code
import httpx

async def speak(text: str) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://voice.machinemachine.ai/speak/file",
            json={"text": text, "format": "mp3"}
        )
        return resp.json()["url"]
```

## WebSocket Streaming

```javascript
const ws = new WebSocket('wss://voice.machinemachine.ai/ws/tts');

ws.onopen = () => {
  ws.send(JSON.stringify({text: "Hello!", voice: "default"}));
};

ws.onmessage = (event) => {
  if (event.data instanceof Blob) {
    // Audio chunk - append to buffer
    audioContext.decodeAudioData(event.data);
  } else {
    // JSON status message
    const status = JSON.parse(event.data);
    if (status.status === "done") console.log("Audio complete");
  }
};
```

## Deployment

See `docker-compose.yml` (GPU) or `docker-compose.cpu.yml` (CPU-only).

Domain: `voice.machinemachine.ai`
