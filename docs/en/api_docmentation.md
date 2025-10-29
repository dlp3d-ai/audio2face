# API Documentation

## Streaming APIs

The system provides a comprehensive streaming API for real-time audio-to-face conversion:

- **StreamingAudio2FaceV1**: Main streaming interface for real-time audio processing
  - WebSocket-based communication for low-latency streaming
  - Chunk-based audio processing for continuous real-time response
  - Configurable postprocessing pipelines for different emotional profiles
  - Asynchronous processing with thread pool management
  - Request expiration and cache management for optimal performance

## Request/Response Format

The API uses Protocol Buffers for efficient serialization and supports:

- **Chunk-based Processing**: Audio input is processed in configurable chunks for real-time response
- **Blendshape Output**: Facial animation data represented as blendshape values
- **Frame-based Timeline**: Precise frame-based timeline management for animation sequencing
- **Streaming Protocol**: WebSocket-based streaming with start/body/end message types
- **Error Handling**: Comprehensive error responses with detailed error codes and messages

### Data Flow

1. **Audio Input**: PCM audio data in configurable chunk sizes
2. **Feature Extraction**: Wav2Vec2-based audio feature extraction
3. **Inference**: ONNX Unitalker model generates blendshape predictions
4. **Postprocessing**: Configurable pipeline applies emotional profiles and effects
5. **Output**: Structured blendshape data with frame timing information

