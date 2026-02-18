# VoxCPM WebSocket TTS Server

A high-performance WebSocket server for real-time text-to-speech using the VoxCPM model.

## Features

- **Real-time streaming audio**: Get audio chunks as they're generated
- **Non-streaming mode**: Generate complete audio files
- **Voice cloning**: Use reference audio for custom voices
- **Concurrent request handling**: Queue-based management of TTS requests
- **WebSocket protocol**: Efficient bidirectional communication
- **Monitoring**: Optional Prometheus metrics export

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd voxcpm-tts-server

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and configure as needed:

```bash
cp .env.example .env
# Edit .env with your settings
```

### Running the Server

```bash
# Using the start script
./scripts/start.sh

# Or directly with Python
python -m src.main
```

The server will start on:
- **WebSocket**: `ws://localhost:9300/tts`
- **Web UI**: `http://localhost:9301`

Open your browser and navigate to `http://localhost:9301` to use the web interface.

## Web Interface

The server includes a web-based client for easy TTS testing:

1. Navigate to `http://localhost:9301`
2. Enter your text in the input area
3. Adjust parameters as needed (CFG value, inference steps, etc.)
4. Click "开始合成" to generate speech
5. Audio will stream directly to your browser
6. Download the generated audio as WAV file

### Web UI Features

- **Real-time streaming audio**: Audio plays as it's generated
- **Visual feedback**: Audio visualizer during playback
- **Parameter controls**: Adjust all TTS parameters
- **History tracking**: View and reuse previous generations
- **Download support**: Save generated audio as WAV

## Usage

### JavaScript Client Example

```javascript
class VoxCPMTTSClient {
  constructor(url = 'ws://localhost:9300/tts') {
    this.url = url;
  }

  async speak(text, onAudioChunk, onProgress) {
    const ws = new WebSocket(this.url);
    const requestId = crypto.randomUUID();

    return new Promise((resolve, reject) => {
      ws.binaryType = 'arraybuffer';

      ws.onopen = () => {
        ws.send(JSON.stringify({
          type: 'tts_request',
          request_id: requestId,
          params: {
            text: text,
            mode: 'streaming'
          }
        }));
      };

      ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
          // Binary audio data
          const { metadata, audio } = this.parseBinaryFrame(event.data);
          if (onAudioChunk) onAudioChunk(audio, metadata);
        } else {
          // JSON control message
          const msg = JSON.parse(event.data);
          if (msg.type === 'progress' && onProgress) {
            onProgress(msg);
          } else if (msg.type === 'complete') {
            resolve();
          } else if (msg.type === 'error') {
            reject(new Error(msg.error.message));
          }
        }
      };

      ws.onerror = (error) => reject(error);
    });
  }

  parseBinaryFrame(buffer) {
    const view = new DataView(buffer);
    const msgType = view.getUint8(2);
    const metadataLength = view.getUint32(4);

    const metadataBytes = new Uint8Array(buffer, 8, metadataLength);
    const metadata = JSON.parse(new TextDecoder().decode(metadataBytes));

    const payloadLength = view.getUint32(8 + metadataLength);
    const audio = new Int16Array(buffer, 8 + metadataLength + 4, payloadLength / 2);

    return { metadata, audio };
  }
}

// Usage
const client = new VoxCPMTTSClient();
await client.speak(
  'Hello, this is a test!',
  (audio, metadata) => console.log('Got audio chunk', metadata.sequence),
  (progress) => console.log('Progress:', progress.progress)
);
```

### Python Client Example

```python
import asyncio
import websockets
import json
import uuid

async def voxcpm_tts(text, url="ws://localhost:9300/tts"):
    async with websockets.connect(url) as ws:
        request = {
            "type": "tts_request",
            "request_id": str(uuid.uuid4()),
            "params": {
                "text": text,
                "mode": "streaming"
            }
        }

        await ws.send(json.dumps(request))

        audio_chunks = []
        while True:
            message = await ws.recv()

            if isinstance(message, bytes):
                # Handle binary audio data
                chunks = parse_binary_frame(message)
                audio_chunks.extend(chunks)
            else:
                # Handle JSON control messages
                msg = json.loads(message)
                if msg.get("type") == "complete":
                    break
                elif msg.get("type") == "error":
                    raise Exception(msg["error"])

        return b"".join(audio_chunks)

# Run
asyncio.run(voxcpm_tts("Hello, world!"))
```

## Protocol

See [PROTOCOL.md](PROTOCOL.md) for detailed protocol documentation.

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TTS_HOST` | Server host | `0.0.0.0` |
| `TTS_PORT` | Server port | `9300` |
| `TTS_MODEL_NAME` | Model name | `openbmb/VoxCPM-0.5B` |
| `TTS_DEVICE` | Compute device | `cuda` |
| `TTS_MAX_CONCURRENT` | Max concurrent requests | `10` |
| `TTS_LOG_LEVEL` | Log level | `INFO` |
| `TTS_METRICS_ENABLED` | Enable metrics | `false` |

### TTS Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to synthesize |
| `mode` | string | `streaming` | `streaming` or `non_streaming` |
| `prompt_wav_url` | string | `null` | Reference audio URL for voice cloning |
| `cfg_value` | float | `2.0` | LM guidance (0.1-10.0) |
| `inference_timesteps` | int | `10` | Number of inference steps (1-50) |
| `normalize` | bool | `false` | Enable text normalization |
| `denoise` | bool | `false` | Enable audio denoising |
| `retry_badcase` | bool | `true` | Enable retry for bad cases |

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Project Structure

```
voxcpm-tts-server/
├── src/                    # Source code
│   ├── config.py          # Configuration
│   ├── server.py          # WebSocket server
│   ├── connection.py      # Connection handling
│   ├── session.py         # Session management
│   ├── tasks.py           # TTS tasks
│   ├── models/            # Data models
│   └── ...
├── tests/                 # Tests
├── config/                # Configuration files
├── scripts/               # Utility scripts
├── docs/                  # Documentation
└── requirements.txt
```

## License

MIT License

## References

- [VoxCPM GitHub](https://github.com/OpenBMB/VoxCPM)
- [WebSocket Protocol RFC 6455](https://tools.ietf.org/html/rfc6455)
