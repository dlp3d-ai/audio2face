# Audio2Face

> **English Documentation** | [中文文档](README_CN.md)

## Table of Contents

- [Overview](#overview)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
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
  dlp3d/audio2face:latest

# Or run with CUDA support (requires NVIDIA GPU with Docker support)
docker run -it \
  --gpus all \
  -p 18083:18083 \
  -v $(pwd)/weights:/workspace/audio2face/weights \
  dlp3d/audio2face:latest-cuda12
```

**Windows:**
```bash
# Pull and run the pre-built image
docker run -it -p 18083:18083 -v .\weights:/workspace/audio2face/weights dlp3d/audio2face:latest
```

**Command Explanation:**
- `-p 18083:18083`: Maps the container's port 18083 to your host machine's port 18083
- `-v $(pwd)/weights:/workspace/audio2face/weights` (Linux/macOS): Mounts your local `weights` directory to the container's weights directory
- `-v .\weights:/workspace/audio2face/weights` (Windows): Mounts your local `weights` directory to the container's weights directory
- `dlp3d/audio2face:latest`: Uses the pre-built public image

**Prerequisites:**
- Ensure you have a `weights` directory in your project root
- Ensure you have a `weights/unitalker_v0.4.0_base.onnx` file in your `weights` directory
- Make sure Docker is installed and running on your system

## Documentation

For detailed documentation, please visit our comprehensive documentation site:

📖 **[Full Documentation](https://dlp3d.readthedocs.io/en/latest/_subrepos/audio2face/overview.html)**

The documentation provides detailed information on:

- **Data Preparation**: Step-by-step guide for downloading and organizing model files
- **Quick Start**: Comprehensive Docker setup and local development instructions
- **Installation Guide**: Detailed environment setup for Linux and Windows
- **API Documentation**: Complete streaming API reference with request/response formats
- **Configuration**: Configuration options for different deployment scenarios
- **Development**: Project structure, testing guidelines, and code quality standards

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
