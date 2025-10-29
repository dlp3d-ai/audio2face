# Overview

Audio2Face is a real-time audio-to-face animation service that converts streaming audio input into synchronized facial animation data. The system uses advanced machine learning models to extract audio features and generate corresponding facial expressions, supporting both CPU and GPU-accelerated inference for optimal performance.

The service is designed for real-time applications such as virtual avatars, live streaming, video conferencing, and interactive entertainment platforms where low-latency audio-to-face conversion is essential.

## Key Features

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

## System Architecture

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

