# ListenAI Voice Assistant System Requirements

## Overview

ListenAI Voice Assistant is a real-time voice-enabled AI system that integrates speech recognition, language model, and text-to-speech services. The system provides a WebSocket-based interface for clients to send audio data and receive intelligent responses, supporting multi-turn conversations with context awareness.

## System Architecture

The system consists of four main services:

1. **Cloud Agent** - WebSocket server that orchestrates all components and manages client connections
2. **ASR Service** - Real-time speech recognition using Fun-ASR model
3. **LLM Service** - OpenAI-compatible language model for generating responses
4. **TTS Service** - Text-to-speech synthesis using VoxCPM model

Additional components:
- **MCP Tool Manager** - Manages device control tools (weather, volume, music)
- **Session Manager** - Maintains conversation history and context
- **VAD Module** - Voice activity detection to identify speech segments
- **Web Test Client** - Browser-based testing interface

## Core Services

### Cloud Agent (WebSocket Server)

#### Functionality

The Cloud Agent is the central orchestrator that connects all services and provides a unified WebSocket interface for clients.

#### Key Features

- **WebSocket Connection Management**
  - Accepts multiple concurrent WebSocket connections
  - Manages connection lifecycle (connect, disconnect, timeout handling)
  - Supports connection from remote addresses
  - Ping/pong mechanism for connection health monitoring

- **Session Management**
  - Creates unique sessions for each client connection
  - Maintains conversation history for multi-turn dialogues
  - Supports session restoration with session IDs
  - Automatic session expiration and cleanup
  - Context tracking for up to 10 recent messages

- **Audio Data Handling**
  - Receives PCM audio data (16-bit, 16kHz, mono) from clients
  - Processes audio in frames with configurable frame size (512 bytes)
  - Buffers audio data when currently processing another request
  - Supports streaming audio for continuous speech

- **Speech Recognition Integration**
  - Connects to ASR WebSocket service for real-time speech recognition
  - Manages ASR client lifecycle and reconnection
  - Sends audio frames to ASR service
  - Receives and processes ASR recognition results
  - Handles both intermediate and final recognition results
  - Monitors ASR connection health

- **Language Model Integration**
  - Connects to LLM HTTP service for response generation
  - Sends recognized text along with conversation history
  - Supports OpenAI-compatible API format
  - Handles tool calling requests and responses
  - Manages LLM client lifecycle

- **Text-to-Speech Integration**
  - Connects to TTS WebSocket service for audio synthesis
  - Sends LLM response text for speech synthesis
  - Receives streaming audio chunks from TTS service
  - Forwards synthesized audio to clients as binary WebSocket messages

- **Tool Calling Support**
  - Integrated MCP (Model Context Protocol) tool manager
  - Provides built-in tools for device control
  - Supports custom tool registration
  - Executes tools and returns results to LLM
  - Maintains tool execution context and history

- **Message Protocol**
  - Handles binary audio data (WebSocket)
  - Handles JSON control messages (WebSocket)
  - Supports configure, start_session, end_session, ping messages
  - Sends ASR results to clients
  - Sends LLM responses to clients
  - Sends TTS audio chunks to clients
  - Sends error messages with error codes
  - Sends status updates for processing states

- **Error Handling**
  - Comprehensive error catching and reporting
  - Error codes for different failure scenarios
  - Graceful error responses to clients
  - Detailed error logging for debugging
  - Service-specific error messages

- **Concurrent Connection Support**
  - Multiple clients can connect simultaneously
  - Each connection has its own session and state
  - Independent processing for each connection
  - Audio buffering per connection during busy periods

### HTTP Test Client Server

#### Functionality

The Cloud Agent includes an HTTP server for serving test files and health checks.

- **Static File Serving**
  - Serves test_client.html for browser-based testing
  - CORS-enabled for cross-origin requests
  - Health check endpoint at /health
  - File serving with appropriate content types

- **Health Monitoring**
  - Service status endpoint
  - Returns service health information
  - Simple JSON response format

### Session Management

#### Functionality

Session Manager handles conversation state and history across multiple interactions.

- **Session Creation**
  - Creates unique session IDs for each connection
  - Tracks session creation and activity timestamps
  - Supports custom session IDs from clients
  - Auto-generates UUID if session ID not provided

- **Conversation History**
  - Maintains list of conversation messages
  - Tracks messages with roles (user, assistant)
  - Includes timestamps for each message
  - Limits history to most recent messages (configurable)
  - Supports message retrieval in LLM format

- **Interaction Tracking**
  - Tracks individual user interactions (ASR + LLM + TTS)
  - Records user input and LLM response
  - Tracks tool calls made during interactions
  - Measures interaction duration
  - Maintains interaction statistics

- **Activity Monitoring**
  - Tracks last activity timestamp
  - Monitors session age
  - Supports session timeout configuration
  - Automatic cleanup of expired sessions

- **Session Statistics**
  - Message count per session
  - Interaction count per session
  - Total session duration
  - Average interaction duration
  - Last activity timestamp

### Protocol Layer

#### Functionality

Protocol module defines the message format and types for client-server communication.

- **Message Type Definitions**
  - Client-to-server message types (audio_data, configure, start_session, end_session, ping)
  - Server-to-client message types (asr_result, llm_response, tts_audio, error, status, pong)
  - Error code definitions

- **Message Creation Helpers**
  - Functions to create standardized message objects
  - Message field validation
  - JSON encoding/decoding
  - Error message formatting

- **Message Types**

Client to Server:
- **audio_data**: Binary PCM audio data (16-bit, 16kHz, mono)
- **configure**: Configuration parameters (language, voice_id, etc.)
- **start_session**: Initialize or restore session
- **end_session**: Terminate current session
- **ping**: Connection health check

Server to Client:
- **asr_result**: Speech recognition result with text and final flag
- **llm_response**: LLM generated response with optional tool calls
- **tts_audio**: Binary audio data (streaming TTS output)
- **error**: Error information with code and message
- **status**: Processing state updates
- **pong**: Response to ping messages

### Configuration Management

#### Functionality

Configuration module manages system settings and parameters.

- **Server Configuration**
  - Host and port settings
  - Ping interval and timeout
  - Maximum connections and concurrent connections
  - Interaction and session timeout settings
  - Rate limiting configuration
  - Logging configuration

- **ASR Service Configuration**
  - WebSocket URL for ASR service
  - Connection timeout settings

- **TTS Service Configuration**
  - WebSocket URL for TTS service
  - Voice ID selection
  - Mode selection (streaming, etc.)
  - Timeout settings

- **LLM Service Configuration**
  - Base URL for LLM API
  - Model selection
  - API key configuration
  - Timeout settings
  - Temperature and max tokens settings

- **Audio Configuration**
  - Sample rate (16000 Hz)
  - Number of channels (1)
  - Bit depth (16-bit)
  Frame size (512 bytes for 16ms at 16kHz)

- **MCP Configuration**
  - Enable/disable MCP tool calling
  - Server name and version
  - Protocol version
  - System instructions

- **Environment Variable Support**
  - Loads configuration from .env file
  - Environment-specific settings
  - Default values for all parameters
- Configuration validation and error handling

## ASR Service

### Overview

ASR Service provides real-time speech recognition using the Fun-ASR model with Voice Activity Detection.

#### Key Features

- **WebSocket Server**
  - Accepts WebSocket connections from Cloud Agent
  - Supports multiple concurrent connections
  - Ping/pong mechanism for connection health
  - Connection timeout handling

- **Real-time Speech Recognition**
  - Uses Fun-ASR-Nano model for speech recognition
  - Supports Chinese, English, and Japanese
  - Supports multiple dialects and regional accents
  - Real-time streaming recognition
  - Supports lyrics recognition and rap speech
  - Model loads on server startup

- **Voice Activity Detection**
  - Uses TenVad for voice activity detection
  - Detects speech start and end boundaries
  - Configurable threshold and silence frame count
  - Hysteresis mechanism to prevent spurious detection
  - Continuous frame-by-frame processing
  - Logging of VAD detection events

- **Speech Segmentation**
  - Identifies individual speech segments
  - Buffers audio from speech start to speech end
  - Minimum speech duration filtering (0.3 seconds)
  - Processes complete speech segments for recognition

- **Frame Buffering**
  - Cross-message audio buffering for VAD frames
  - Handles non-aligned message sizes and frame boundaries
  - Ensures no data loss between messages
  - Extracts complete frames from buffered data
  - Maintains remaining data between messages

- **Audio Buffer**
  - Circular buffer for incoming audio data
  - Maximum buffer duration limit (configurable)
  - Sample-wise audio management
  - Automatic cleanup of old data
  - Duration and sample count tracking

- **Speech Segment Recognition**
  - Processes complete speech segments
  - Saves audio segments to temporary WAV files for processing
  - Runs ASR inference on saved audio files
  - Extracts recognized text from results
- **Segment ID tracking for each recognition**

- **Result Streaming**
  - Sends recognition results to clients in real-time
  - Supports intermediate and final results
  - Includes speeching state indicator
  - Includes segment ID for tracking
  - Error handling and error messages

- **Command Handling**
  - Supports ping/pong for connection health
  - Supports reset command to clear state
  - JSON-based command messages
- **Logging and Debugging**
  - Comprehensive logging at configurable levels
-   - Connection events
-   - Audio receiving statistics
-   - VAD detection events
-   - Speech segment boundaries
-   - Recognition results
-   - Error conditions

### VAD Module

#### Functionality

TenVad provides voice activity detection to identify when the user is speaking.

- **Frame-based Processing**
  - Processes audio frame by frame
  - Calculates speech probability for each frame
  - Configurable hop size (256 samples)

- **Speech State Detection**
  - Binary speech/non-speech classification
  - Hysteresis mechanism to prevent state flutter
  - Immediate transition to speech on first speech frame
  - Delayed transition to silence (requires consecutive silence frames)

- **Configurable Parameters**
  - Speech probability threshold (default 0.5)
  - Number of consecutive silence frames to exit speech (default 5)
  - Hop size in samples (default 256)

- **State Tracking**
  - Current speeching state (boolean)
  - Consecutive silence frame counter
  - Total frames processed counter

- **Event Logging**
  - Speech start events with frame number and probability
  - Speech end events with frame count
  - Per-frame debug logging with state information

## TTS Service

### Overview

TTS Service provides text-to-speech synthesis using the VoxCPM model.

#### Key Features

- **WebSocket Server**
  - Accepts WebSocket connections from Cloud Agent
  - Supports streaming synthesis
  - Connection management and health monitoring

- **Text-to-Speech Synthesis**
  - Uses VoxCPM-0.5B model for synthesis
  - Generates human-like speech
  - Supports various voices and styles
  - Low-latency synthesis for real-time applications

- **Streaming Audio Output**
  - Streams synthesized audio in chunks
  - Sends audio data as binary WebSocket messages
  - Reduces latency through streaming
  - Allows audio playback while synthesis continues

- **Voice Configuration**
  - Voice ID selection (e.g., doubao-通用女声)
  - Multiple voice options available
- - Voice style and quality configuration

- **Model Parameters**
  - Inference timesteps configuration
  - CFG (Context-Free Grammar) value
  - Sampling parameters

- **Error Handling**
  - Synthesis error detection and reporting
  - Connection error handling
- - Client error notifications
- Comprehensive error logging

## LLM Service

### Overview

LLM Service provides natural language understanding and response generation capabilities.

#### Key Features

- **OpenAI-Compatible API**
  - Standard chat completion interface
- - Compatible with OpenAI-compatible model APIs
  - Supports message-based conversations
  - JSON request and response format

- **Language Model Integration**
  - Supports GPT-3.5-turbo and compatible models
  - Configurable model selection
- Temperature control for response randomness
- Max tokens limit for response length

- **Conversation Context**
  - Maintains conversation history
- - Supports multi-turn conversations
- - Tracks user and assistant messages
  - Provides context window for model

- **Function Calling Support**
  - Supports OpenAI function calling format
- - Tool definition and schema
- Tool selection and execution
  - Function call result handling
- Supports multiple function calls in single response
- Tool result integration into conversation

- **API Endpoint**
  - Chat completion endpoint
  - Streaming response support
- Request timeout configuration
- API key authentication support

- **Error Handling**
  - Request and response error detection
- Timeout handling
- Retry mechanism support
- Comprehensive error logging

## MCP Tool Management

### Overview

MCP (Model Context Protocol) Tool Manager provides device control and external tool integration capabilities.

#### Key Features

- **Tool Registry**
  - Maintains registry of available tools
  - Supports tool registration and unregistration
  - Tool metadata storage (name, description, parameters, schema)

- **Built-in Tools**
  - get_weather: Get weather information for a specified city
  - set_volume: Set device volume (0-100 range)
  - play_music: Play music on the device

- **Tool Execution**
  - Synchronous tool execution
  - Parameter parsing and validation
  - Tool result capture
- - Error detection and reporting
  - Execution timeout handling

- **LLM Integration**
  - Provides tool definitions to LLM in standard format
  - Formats tools for function calling
  - Includes tool descriptions and schemas
- - Manages tool request/response cycle

- **Tool Call History**
  - Tracks tool executions for each interaction
  - Maintains tool execution order
- - Records tool results and success/failure status
- - Provides context for tool result interpretation

- **Tool Protocol Support**
  - Model Context Protocol version 1.0 compatible
- - JSON-based tool definitions
  - Standard parameter passing format
- - Tool result format specification

## Web Test Client

### Overview

Browser-based testing client for the Cloud Agent with comprehensive audio recording and visualization features.

#### Key Features

- **WebSocket Connection Management**
  - Connect to Cloud Agent WebSocket server
  - Connection status indicator
  - Reconnection support
  - Remote address configuration
  - Connection error handling

- **Real-time Audio Recording**
  - Microphone access with browser permission handling
  - Echo cancellation for noise reduction
- Noise suppression for clear audio
  - Sample rate configuration (16000 Hz)
  - Real-time audio capture

- **Audio Processing**
  - Float32 to Int16 PCM conversion
  - Configurable chunk size for sending
  - Sample count calculation
  - Real-time audio streaming to WebSocket
- Frame-based audio capture with ScriptProcessor

- **Real-time Visualization**
  - Frequency visualizer display
- Real-time frequency spectrum analysis
- FFT-based visualization
- Configurable update rate for performance

- **Message Display and Categorization**
  - Color-coded messages by type:
    - Blue: System messages
    - Purple: ASR (speech recognition) results
    - Orange: LLM (AI) responses
    - Red: Error messages
    - Green: Tool call results
  - Message timestamps
  - Scrollable message history

- **Tool Call Visualization**
  - Shows executed tools with icons
  - Displays tool parameters
  - Success/failure status indicators
  - Tool result display
  - Tool execution timing

- **Audio Playback**
  - Built-in audio player for TTS output
- Volume control slider
- Play/pause controls
- Automatic playback of TTS audio chunks

- **Statistics Dashboard**
  - Real-time metrics display:
    - Total messages count
    - Tool calls count
    - Average latency tracking
  - Session duration display
  - Connection status indicator

- **Configuration Panel**
  - Language selection dropdown
- Voice ID input field
- Apply settings button
- Configuration persistence

- **Browser Compatibility**
  - Chrome/Edge (recommended)
  - Firefox
  - Safari
  - Requires: WebSocket, Web Audio API, MediaStream API

### HTTP Server Integration

The Cloud Agent includes an HTTP server for serving the test client and providing health check endpoints.

- **Static File Serving**
  - Serves test_client.html from the cloud directory
- - CORS-enabled headers for cross-origin requests
- - Content-type header based on file extension
- Health check endpoint at /health

- **Auto-start Script**
- Convenient script to start both WebSocket Agent and HTTP Test Client
- Automatic dependency checking
- Clear startup logging and status messages
- Single command to launch complete test environment

## Audio Processing Pipeline

### Complete Data Flow

1. **Audio Input**: Client sends PCM audio (16-bit, 16kHz, mono) via WebSocket
2. **Audio Buffering**: Cloud Agent buffers audio and extracts 512-byte frames
3. **Frame Transmission**: Frames are sent to ASR service via WebSocket
4. **VAD Processing**: TenVad processes each frame for speech activity
5. **Speech Detection**: Speech segment boundaries are detected (start and end)
6. **Speech Buffering**: Audio from speech start to end is buffered
7. **Segment Recognition**: ASR processes complete speech segment
8. **Result Streaming**: Recognition result is sent to Cloud Agent
9. **Conversation Context**: Result is added to conversation history
10. **LLM Processing**: Text with context is sent to LLM service
11. **Tool Execution**: LLM may request tool calls (if enabled)
12. **Tool Results**: MCP tool manager executes tools and returns results
13. **LLM Final Response**: LLM generates final response with tool results
14. **Response Streaming**: Final LLM response is sent to client
15. **TTS Synthesis**: Response text is sent to TTS service
16. **Audio Streaming**: TTS returns streaming audio chunks
17. **Audio Forwarding**: Audio chunks are forwarded to client
18. **Audio Playback**: Client plays audio as it arrives

### Concurrent Processing

- **Multiple Audio Streams**: Supports simultaneous audio input from multiple clients
- **Independent Sessions**: Each connection has its own session and state
- **Background Task Processing**: ASR result processing runs in background tasks
- **Non-blocking Message Loop**: Main message loop continues while ASR processes results
- **Audio Buffering**: New audio is buffered while processing previous request

## Configuration

### Environment Variables

#### Server Configuration
- CLOUD_HOST: Server binding host (default: 0.0.0.0)
- CLOUD_PORT: Server port (default: 9400)
- CLOUD_PING_INTERVAL: WebSocket ping interval (default: 30)
- CLOUD_PING_TIMEOUT: WebSocket ping timeout (default: 300)
- CLOUD_MAX_CONNECTIONS: Maximum concurrent connections (default: 100)
- CLOUD_MAX_CONCURRENT: Maximum concurrent processing (default: 50)
- CLOUD_INTERACTION_TIMEOUT: Request timeout (default: 600)
- CLOUD_SESSION_TIMEOUT: Session timeout (default: 3600)
- CLOUD_RATE_LIMIT_ENABLED: Enable rate limiting (default: true)
- CLOUD_RATE_LIMIT_PER_MINUTE: Requests per minute limit (default: 60)
- CLOUD_LOG_LEVEL: Logging level (default: INFO)
- CLOUD_LOG_FORMAT: Log format (default: json)

#### Backend Services
- ASR_SERVICE_URL: ASR WebSocket URL (default: ws://192.168.1.169:9200)
- ASR_TIMEOUT: ASR connection timeout (default: 60)
- TTS_SERVICE_URL: TTS WebSocket URL (default: ws://192.168.1.169:9300/tts)
- TTS_TIMEOUT: TTS connection timeout (default: 300)
- TTS_VOICE_ID: TTS voice selection (default: doubao-通用女声)
- TTS_MODE: TTS mode (default: streaming)
- TTS_CFG_VALUE: CFG value for TTS (default: 2.0)
- TTS_INFERENCE_TIMESTEPS: Inference timesteps (default: 30)
- LLM_BASE_URL: LLM API base URL (default: http://192.168.13.228:8000/v1/)
- LLM_MODEL: LLM model selection (default: gpt-3.5-turbo)
- LLM_API_KEY: LLM API key for authentication
- LLM_TIMEOUT: LLM request timeout (default: 120)
- LLM_TEMPERATURE: Temperature for response randomness (default: 0.7)
- LLM_MAX_TOKENS: Maximum response tokens (default: 2048)

#### MCP Configuration
- MCP_ENABLED: Enable MCP tool calling (default: true)
- MCP_SERVER_NAME: MCP server name (default: arcs-mini-mcp-server)
- MCP_SERVER_VERSION: MCP server version (default: 1.0.0)
- MCP_PROTOCOL_VERSION: MCP protocol version (default: 2024-11-05)
- MCP_INSTRUCTIONS: MCP system instructions

#### Audio Configuration
- AUDIO_SAMPLE_RATE: Sample rate in Hz (default: 16000)
- AUDIO_CHANNELS: Number of audio channels (default: 1)
- AUDIO_BITS_PER_SAMPLE: Bit depth (default: 16)
- AUDIO_FRAME_SIZE: Frame size in bytes (default: 512)

### File Structure

```
listenai_server/
├── cloud/                    # Cloud Agent (WebSocket server + test client)
│   ├── .env                    # Configuration
│   ├── requirements.txt          # Python dependencies
│   ├── README.md               # Cloud Agent documentation
│   ├── auto.sh                 # Auto-start script
│   ├── test_client.html          # Web test client
│   ├── start_test_client.sh     # Test client launcher
│   ├── diagnose.sh              # Diagnostic tool
│   └── src/
│       ├── __init__.py         # Package initialization
│       ├── config.py              # Configuration management
│       ├── server.py              # Main WebSocket server
│       ├── asr_client.py          # ASR WebSocket client
│       ├── llm_client.py          # LLM HTTP client
│       ├── tts_client.py          # TTS WebSocket client
│       ├── mcp_manager.py         # MCP tool manager
│       ├── session.py              # Session management
│       ├── protocol.py             # Message protocol definitions
│       └── audio_buffer.py         # Audio buffering
├── asr/                       # ASR Service
│   ├── Fun-ASR/               # Fun-ASR model
│   │   └── asr_websocket_server.py  # ASR WebSocket server
│   └── ten-vad/                # VAD module
│       └── lib/Web/            # TenVad library
├── tts/                       # TTS Service
│   ├── VoxCPM-0.5B/            # VoxCPM model
│   └── TTS_WEBSOCKET_PROTOCOL.md  # TTS WebSocket protocol docs
├── llm/                       # LLM Service (OpenAI-compatible API)
│   └── main.py                # LLM server
├── docs/                       # Documentation
│   ├── websocket-message-flow.md  # WebSocket message processing flow
│   ├── asr-troubleshooting.md  # ASR troubleshooting guide
│   ├── frame-size-mismatch-fix.md  # Frame size mismatch fix
│   ├── asr-getresults-infinite-loop-fix.md  # ASR results collection fix
│   ├── asr-blocking-async-for-fix.md  # Async blocking issue fix
│   └── complete-asr-fixes-summary.md  # Complete ASR fixes summary
└── arcs_mini/                 # Device firmware
    └── modules/           # Device modules
```

## Client Types

### WebSocket Clients

**Supported Clients**:
- Browser-based JavaScript client (test_client.html)
- Python WebSocket client
- Any WebSocket-compatible client
- Mobile applications with WebSocket support

**Client Requirements**:
- WebSocket protocol support
- Audio capture capability (for voice input)
- Audio playback capability (for TTS output)
- JSON message handling
- Binary message handling for audio

### Connection Modes

**Single Connection Mode**:
- One client per conversation
- Direct WebSocket connection to Cloud Agent
- Independent session per connection

**Multiple Concurrent Connections**:
- Multiple clients can connect simultaneously
- Each client has independent session and state
- Shared ASR, LLM, TTS services
- Independent audio processing per connection

## Error Handling

### Error Codes

| Code | Description | Component |
|-------|-------------|------------|
| INVALID_MESSAGE | Invalid message format | Cloud Agent |
| UNKNOWN_MESSAGE_TYPE | Unknown message type | Cloud Agent |
| ASR_ERROR | ASR service error | Cloud Agent |
| LLM_ERROR | LLM service error | Cloud Agent |
| TTS_ERROR | TTS service error | Cloud Agent |
| SESSION_ERROR | Session management error | Cloud Agent |
| TIMEOUT | Request timeout | Cloud Agent |
| INTERNAL_ERROR | Internal server error | Cloud Agent |

### Error Handling Strategy

- **Detection**: All components have error detection and handling
- **Logging**: Comprehensive error logging with context
- **Reporting**: Errors are sent to clients via WebSocket
- **Recovery**: Connection retry where applicable
- **Graceful Degradation**: Partial functionality on errors

## Security Considerations

### Connection Security

- WebSocket secure connection (wss://) for production
- API key authentication for LLM service
- Connection rate limiting
- Session timeout enforcement

### Data Privacy

- Audio data is ephemeral (not stored permanently)
- Session history is in-memory (cleared on session end)
- No persistent logging of conversation content (except metadata)

### Deployment Considerations

- **Network Requirements**: Low-latency connection between services
- **Resource Requirements**: Sufficient CPU/GPU for model inference
- **Scalability**: Horizontal scaling by adding service instances
- **High Availability**: Multiple service instances for failover

## Monitoring and Observability

### Logging

- Structured logging with timestamps and component names
- Log levels: DEBUG, INFO, WARNING, ERROR
- Key events logged:
  - Connection lifecycle events
  - Audio processing stages
  - ASR recognition events
  - LLM interactions
  - Tool executions
  - Errors and exceptions

### Metrics Tracking

- Connection count and statistics
- Message throughput
- Request latency tracking
- Error rates and types
- Resource utilization

## Performance Characteristics

- **Real-time Response**: Streaming audio recognition and synthesis
- **Low Latency**: Optimized frame size (16ms @ 16kHz)
- **High Throughput**: Concurrent connection support
- **Scalable**: Service-based architecture allows horizontal scaling

## Development and Testing

### Development Environment

- Python 3.10+
- Asyncio for asynchronous operations
- WebSocket for real-time communication
- Type hints for code clarity

### Testing Tools

- Web test client for manual testing
- Diagnostic script for automated checks
- Comprehensive logging for debugging

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-19  
**Status**: Complete requirements documentation
