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

## Overview

TODO

### Key Features

TODO

### System Architecture

The system consists of several key components:

TODO

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
- The `weights` directory contains the pre-trained model required for inference.

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
docker run -it -p 18083:18083 -v .\weights:/workspace/audio2face/weights ockersenseyang/dlp3d_audio2face:latest
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

TODO

### Request/Response Format

TODO

## Configuration

TODO

## Development

### Project Structure

TODO

### Testing

TODO

### Code Quality

The project maintains high code quality with:

- **Linting**: Ruff for code style and quality checks
- **Type Hints**: Full type annotation support
- **CI/CD**: Automated testing and deployment pipelines

---


