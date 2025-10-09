# Audio2Face

> **English Documentation** | [中文文档](docs/README_CN.md)

## Table of Contents

- [Overview](#overview)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Development](#development)
- [Citation](#citation)
- [License](#license)

## Overview

Audio2Face is a real-time audio-to-face animation service that converts streaming audio input into synchronized facial animation data. The system uses advanced machine learning models to extract audio features and generate corresponding facial expressions, supporting both CPU and GPU-accelerated inference for optimal performance.

The service is designed for real-time applications such as virtual avatars, live streaming, video conferencing, and interactive entertainment platforms where low-latency audio-to-face conversion is essential.

### Key Features

- **Real-time Streaming Processing**: Process audio chunks in real-time with low latency for live applications
- **Dual Inference Engines**: Support for both ONNX-based Unitalker and PyTorch-based feature extraction
- **GPU Acceleration**: CUDA 12.1 support for high-performance inference on NVIDIA GPUs
- **Comprehensive Postprocessing**: 9 postprocessing modules including blendshape clipping, scaling, thresholding, and custom blink animations
- **Flexible Audio Splitting**: Energy-based silence detection for intelligent audio segmentation
- **WebSocket API**: FastAPI-based streaming interface with Protocol Buffers serialization
- **Docker Support**: Pre-built Docker images for both CPU and CUDA environments
- **Configurable Pipelines**: Modular architecture allowing custom postprocessing pipeline configurations
- **Multi-threaded Processing**: Asynchronous processing with configurable worker pools
- **Comprehensive Testing**: Full test coverage with pytest and async testing support

### System Architecture

The system consists of several key components:

- **Streaming API Layer**: FastAPI-based WebSocket server handling real-time audio streaming requests
- **Feature Extraction**: Audio feature extraction using Wav2Vec2 models via PyTorch or ONNX runtime
- **Inference Engine**: ONNX-based Unitalker model for audio-to-blendshape conversion
- **Audio Splitting**: Energy-based silence detection for intelligent audio segmentation
- **Postprocessing Pipeline**: Modular postprocessing system with 9 specialized modules:
  - Blendshape clipping, scaling, and thresholding
  - Custom blink animation injection
  - Linear and exponential blending
  - Offset adjustment and name mapping
- **Data Structures**: FaceClip class for managing facial animation data with format conversion
- **Configuration System**: Flexible configuration management for different deployment scenarios
- **Logging & Monitoring**: Comprehensive logging with AWS CloudWatch integration support

## Data Preparation

To use Audio2Face, you need to download the ONNX model file and set up the required directory structure.

### Download ONNX Model

1. **Download the ONNX model file:**
   - **GitHub Download:** [unitalker_v0.4.0_base.onnx](https://github.com/LazyBusyYang/CatStream/releases/download/a2f_cicd_files/unitalker_v0.4.0_base.onnx)
   - **Google Drive Download:** [unitalker_v0.4.0_base.onnx](https://drive.google.com/file/d/1E0NTrsh4mciRPb265n64Dd5vR3Sa7Dgx/view?usp=drive_link)

2. **Organize the data:**
   - Create a `weights` directory in your project root if it doesn't exist
   - Place the downloaded `unitalker_v0.4.0_base.onnx` file in the `weights` directory
   - Ensure the following directory structure is created:

```
├─audio2face
├─configs
├─docs
└─weights
   └─unitalker_v0.4.0_base.onnx
```

### Directory Structure Explanation

- `weights/`: A folder for storing ONNX model files.
- `weights/unitalker_v0.4.0_base.onnx`: The main ONNX model file for audio-to-face conversion.

## Quick Start

### Using Docker

The easiest way to get started with Audio2Face is using the pre-built Docker image:

**Linux/macOS:**
```bash
# Pull and run the pre-built image (CPU version)
docker run -it \
  -p 18083:18083 \
  -v $(pwd)/weights:/workspace/audio2face/weights \
  dockersenseyang/dlp3d_audio2face:latest

# Or run with CUDA support (requires NVIDIA GPU with Docker support)
docker run -it \
  --gpus all \
  -p 18083:18083 \
  -v $(pwd)/weights:/workspace/audio2face/weights \
  dockersenseyang/dlp3d_audio2face:latest-cuda12
```

**Windows:**
```cmd
# Pull and run the pre-built image
docker run -it -p 18083:18083 -v .\weights:/workspace/audio2face/weights dockersenseyang/dlp3d_audio2face:latest
```

**Command Explanation:**
- `-p 18083:18083`: Maps the container's port 18083 to your host machine's port 18083
- `-v $(pwd)/weights:/workspace/audio2face/weights` (Linux/macOS): Mounts your local `weights` directory to the container's weights directory
- `-v .\weights:/workspace/audio2face/weights` (Windows): Mounts your local `weights` directory to the container's weights directory
- `dockersenseyang/dlp3d_audio2face:latest`: Uses the pre-built public image

**Prerequisites:**
- Ensure you have a `weights` directory in your project root
- Ensure you have a `weights/unitalker_v0.4.0_base.onnx` file in your `weights` directory
- Make sure Docker is installed and running on your system

**Alternative: Build from Source**

If you prefer to build the image from source:

**Linux/macOS:**
```bash
# Build the Docker image
docker build -f dockerfiles/Dockerfile-cpu -t audio2face:local .

# Run the container
docker run -it \
  -p 18083:18083 \
  -v $(pwd)/weights:/workspace/audio2face/weights \
  audio2face:local
```

**Windows:**
```cmd
# Build the Docker image
docker build -f dockerfiles/Dockerfile-cpu -t audio2face:local .

# Run the container
docker run -it -p 18083:18083 -v .\weights:/workspace/audio2face/weights audio2face:local
```

## Environment Setup

For local development and deployment, please follow the detailed installation guide:

📖 **[Complete Installation Guide](docs/install.md)**

The installation guide provides step-by-step instructions for:
- Setting up Python 3.10+ environment
- Installing Protocol Buffers compiler
- Configuring the development environment
- Installing project dependencies

### Local Development

After completing the environment setup as described in the installation guide, you can start the service locally:

```bash
# Activate the conda environment
conda activate audio2face

# Start the service
python main.py
```

## API Documentation

### Streaming APIs

The system provides a comprehensive streaming API for real-time audio-to-face conversion:

- **StreamingAudio2FaceV1**: Main streaming interface for real-time audio processing
  - WebSocket-based communication for low-latency streaming
  - Chunk-based audio processing for continuous real-time response
  - Configurable postprocessing pipelines for different emotional profiles
  - Asynchronous processing with thread pool management
  - Request expiration and cache management for optimal performance

### Request/Response Format

The API uses Protocol Buffers for efficient serialization and supports:

- **Chunk-based Processing**: Audio input is processed in configurable chunks for real-time response
- **Blendshape Output**: Facial animation data represented as blendshape values
- **Frame-based Timeline**: Precise frame-based timeline management for animation sequencing
- **Streaming Protocol**: WebSocket-based streaming with start/body/end message types
- **Error Handling**: Comprehensive error responses with detailed error codes and messages

**Data Flow:**
1. **Audio Input**: PCM audio data in configurable chunk sizes
2. **Feature Extraction**: Wav2Vec2-based audio feature extraction
3. **Inference**: ONNX Unitalker model generates blendshape predictions
4. **Postprocessing**: Configurable pipeline applies emotional profiles and effects
5. **Output**: Structured blendshape data with frame timing information

## Configuration

The system supports flexible configuration for different deployment scenarios:

### Local Development Configuration

For local development, use `configs/cpu.py` which configures:
- **CPU Inference**: PyTorch CPU-only and ONNX Runtime CPU configurations

### Production Configuration

For production deployment, use `configs/cuda.py` which supports:
- **GPU Acceleration**: CUDA 12.1 support for high-performance inference

### Configuration Components

- **Feature Extractor**: Wav2Vec2 model configuration and audio processing parameters
- **Inference Engine**: ONNX Unitalker model paths and inference settings
- **Audio Splitting**: Energy-based silence detection parameters
- **Postprocessing**: Configurable pipelines with emotional profiles and effects
- **Server Settings**: FastAPI server configuration, CORS, and worker settings
- **Logging**: Comprehensive logging configuration with multiple output targets

## Development

### Project Structure

```
audio2face/
├── apis/                    # Streaming API implementations
├── data_structures/         # Core data models and Protocol Buffers
├── infer/                   # Inference engines (ONNX and PyTorch)
├── postprocess/             # Postprocessing modules and pipelines
├── service/                 # FastAPI server and error handling
├── split/                   # Audio splitting and segmentation
├── utils/                   # Utility functions and base classes
└── tests/                   # Comprehensive test suite
```

### Testing

The project includes comprehensive tests with pytest and async testing support. Before running tests, you need to prepare the test input data.

#### Test Data Preparation

Download the required test files and organize them in the correct directory structure:

1. **Create test input directory:**
   ```bash
   mkdir -p input
   ```

2. **Download test files:**
   - **Test Audio:** [test_audio.wav](https://github.com/LazyBusyYang/CatStream/releases/download/a2f_cicd_files/test_audio.wav) - Download and save to `input/test_audio.wav`
   - **Test Feature:** [test_feature.npy](https://github.com/LazyBusyYang/CatStream/releases/download/a2f_cicd_files/test_feature.npy) - Download and save to `input/test_feature.npy`

3. **Ensure weights directory exists:**
   ```bash
   mkdir -p weights
   ```
   Make sure you have the ONNX model file `weights/unitalker_v0.4.0_base.onnx` (see [Data Preparation](#data-preparation) section).

4. **Final directory structure should be:**
   ```
   ├─audio2face
   ├─configs
   ├─docs
   ├─input
   │  ├─test_audio.wav
   │  └─test_feature.npy
   └─weights
      └─unitalker_v0.4.0_base.onnx
   ```

#### Running Tests

```bash
# Run all tests
pytest tests --log-cli-level=ERROR

# Run specific test categories
pytest tests/apis/           # API functionality tests
pytest tests/infer/          # Inference engine tests
pytest tests/postprocess/    # Postprocessing module tests
pytest tests/data_structures/ # Data structure tests
```

**Test Coverage:**
- **API Tests**: Streaming API functionality
- **Inference Tests**: ONNX Unitalker and PyTorch feature extractor validation
- **Postprocessing Tests**: All postprocessing modules with various configurations
- **Data Structure Tests**: FaceClip class functionality and format conversions

### Code Quality

The project maintains high code quality with:

- **Linting**: Ruff for code style and quality checks
- **Type Hints**: Full type annotation support
- **CI/CD**: Automated testing and deployment pipelines

## Citation

This project uses the UniTalker algorithm for facial animation generation:

```bibtex
@article{unitalker2024,
  title={UniTalker: Scaling up Audio-Driven 3D Facial Animation through A Unified Model},
  journal={ECCV},
  year={2024}
}
```

**Reference**: [UniTalker GitHub Repository](https://github.com/X-niper/UniTalker) - ECCV 2024

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

The MIT License is a permissive free software license that allows you to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software with minimal restrictions. The only requirement is that the original copyright notice and license text must be included in all copies or substantial portions of the software.

---


