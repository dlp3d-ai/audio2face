# Configuration

The system supports flexible configuration for different deployment scenarios:

## Local Development Configuration

For local development, use `configs/cpu.py` which configures:

- **CPU Inference**: PyTorch CPU-only and ONNX Runtime CPU configurations

## Production Configuration

For production deployment, use `configs/cuda.py` which supports:

- **GPU Acceleration**: CUDA 12.1 support for high-performance inference

## Configuration Components

- **Feature Extractor**: Wav2Vec2 model configuration and audio processing parameters
- **Inference Engine**: ONNX Unitalker model paths and inference settings
- **Audio Splitting**: Energy-based silence detection parameters
- **Postprocessing**: Configurable pipelines with emotional profiles and effects
- **Server Settings**: FastAPI server configuration, CORS, and worker settings
- **Logging**: Comprehensive logging configuration with multiple output targets

