# Quick Start

The easiest way to get started with Audio2Face is using the pre-built Docker image. This guide will help you quickly set up and run the service.

## Prerequisites

Before starting, ensure you have:

- Docker installed and running on your system
- A `weights` directory in your project root
- The `unitalker_v0.4.0_base.onnx` file in your `weights` directory (see [Data Preparation](data_preparation.md))

## Using Docker (Recommended)

### Option 1: Use Pre-built Docker Images

#### Linux/macOS

**CPU Version:**

```bash
# Pull and run the pre-built image (CPU version)
docker run -it \
  -p 18083:18083 \
  -v $(pwd)/weights:/workspace/audio2face/weights \
  dlp3d/audio2face:latest
```

**CUDA Version (requires NVIDIA GPU):**

```bash
# Run with CUDA support (requires NVIDIA GPU with Docker support)
docker run -it \
  --gpus all \
  -p 18083:18083 \
  -v $(pwd)/weights:/workspace/audio2face/weights \
  dlp3d/audio2face:latest-cuda12
```

#### Windows

```bash
# Pull and run the pre-built image
docker run -it -p 18083:18083 -v .\weights:/workspace/audio2face/weights dlp3d/audio2face:latest
```

**Command Parameters Explained:**

- `-p 18083:18083`: Maps the container's port 18083 to your host machine's port 18083
- `-v $(pwd)/weights:/workspace/audio2face/weights` (Linux/macOS): Mounts your local `weights` directory to the container's weights directory
- `-v .\weights:/workspace/audio2face/weights` (Windows): Mounts your local `weights` directory to the container's weights directory
- `dlp3d/audio2face:latest`: Uses the pre-built public CPU image
- `dlp3d/audio2face:latest-cuda12`: Uses the pre-built CUDA GPU image
- `--gpus all`: Enables GPU support for Docker (required for CUDA version)

### Option 2: Build from Source

If you prefer to build the Docker image from source:

#### Linux/macOS

```bash
# Build the Docker image
docker build -f dockerfiles/Dockerfile-cpu -t audio2face:local .

# Run the container
docker run -it \
  -p 18083:18083 \
  -v $(pwd)/weights:/workspace/audio2face/weights \
  audio2face:local
```

#### Windows

```bash
# Build the Docker image
docker build -f dockerfiles/Dockerfile-cpu -t audio2face:local .

# Run the container
docker run -it -p 18083:18083 -v .\weights:/workspace/audio2face/weights audio2face:local
```

## Verify Installation

After starting the Docker container, you should see output indicating that the service is running. The service will be accessible at:

- **Local**: http://localhost:18083
- **WebSocket API**: ws://localhost:18083

You can verify the service is running by checking the container logs:

```bash
# Check container logs
docker logs <container_id>
```

## Local Development Setup

For local development without Docker, please follow the detailed installation guide:

- **Installation Guide**: See [Installation Guide](install.md) for step-by-step instructions

The installation guide provides:

- Setting up Python 3.10+ environment
- Installing Protocol Buffers compiler
- Configuring the development environment
- Installing project dependencies

### Starting the Service Locally

After completing the environment setup as described in the installation guide:

```bash
# Activate the conda environment
conda activate audio2face

# Start the service
python main.py
```

## Next Steps

Once you have the service running:

1. **Explore the API**: Check the API documentation to understand available endpoints
2. **Test the Service**: Use the streaming API to process audio and generate facial animations
3. **Configuration**: Customize the postprocessing pipeline and inference settings
4. **Development**: See the [Development Guide](development.md) for more information
