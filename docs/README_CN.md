# Audio2Face

> **English Documentation** | [中文文档](docs/README_CN.md)

## 目录

- [概述](#概述)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [环境设置](#环境设置)
- [API文档](#api文档)
- [配置](#配置)
- [开发](#开发)
- [引用](#引用)
- [许可证](#许可证)

## 概述

Audio2Face是一个实时音频到面部表情动画服务，将流式音频输入转换为同步的面部动画数据。该系统使用先进的机器学习模型提取音频特征并生成相应的面部表情，支持CPU和GPU加速推理以获得最佳性能。

该服务专为虚拟人驱动、直播、视频会议和互动娱乐平台等需要低延迟音频到面部转换的实时应用而设计。

### 主要特性

- **实时流式处理**：实时处理音频块，为直播应用提供低延迟
- **双推理引擎**：支持基于ONNX的Unitalker和基于PyTorch的特征提取
- **GPU加速**：支持CUDA 12.1，在NVIDIA GPU上实现高性能推理
- **全面后处理**：9个后处理模块，包括blendshape裁剪、缩放、阈值处理和自定义眨眼动画
- **灵活音频分割**：基于能量的静音检测，实现智能音频分割
- **WebSocket API**：基于FastAPI的流式接口，使用Protocol Buffers序列化
- **Docker支持**：预构建的CPU和CUDA环境Docker镜像
- **可配置管道**：模块化架构，允许自定义后处理管道配置
- **多线程处理**：异步处理，支持可配置的工作池
- **全面测试**：使用pytest和异步测试支持的完整测试覆盖

### 系统架构

系统由以下几个关键组件组成：

- **流式API层**：基于FastAPI的WebSocket服务器，处理实时音频流请求
- **特征提取**：使用Wav2Vec2模型通过PyTorch或ONNX运行时进行音频特征提取
- **推理引擎**：基于ONNX的Unitalker模型，用于音频到blendshape转换
- **音频分割**：基于能量的静音检测，实现智能音频分割
- **后处理管道**：模块化后处理系统，包含9个专用模块：
  - Blendshape裁剪、缩放和阈值处理
  - 自定义眨眼动画注入
  - 线性和指数混合
  - 偏移调整和名称映射
- **数据结构**：FaceClip类，用于管理面部动画数据和格式转换
- **配置系统**：灵活的配置管理，支持不同部署场景
- **日志和监控**：全面的日志记录，支持AWS CloudWatch集成

## 数据准备

要使用Audio2Face，您需要下载ONNX模型文件并设置所需的目录结构。

### 下载ONNX模型

1. **下载ONNX模型文件：**
   - **GitHub下载：** [unitalker_v0.4.0_base.onnx](https://github.com/LazyBusyYang/CatStream/releases/download/a2f_cicd_files/unitalker_v0.4.0_base.onnx)
   - **Google Drive下载：** [unitalker_v0.4.0_base.onnx](https://drive.google.com/file/d/1E0NTrsh4mciRPb265n64Dd5vR3Sa7Dgx/view?usp=drive_link)
   - **百度网盘：** [unitalker_v0.4.0_base.onnx](https://pan.baidu.com/s/1A_vUj_ZBMFPbO1lgUYVCPA)（提取码：`shre`）

2. **组织数据：**
   - 在项目根目录创建`weights`目录（如果不存在）
   - 将下载的`unitalker_v0.4.0_base.onnx`文件放置在`weights`目录中
   - 确保创建以下目录结构：

```
├─audio2face
├─configs
├─docs
└─weights
   └─unitalker_v0.4.0_base.onnx
```

### 目录结构说明

- `weights/`：存储ONNX模型文件的文件夹
- `weights/unitalker_v0.4.0_base.onnx`：用于音频到面部转换的主要ONNX模型文件

## 快速开始

### 使用Docker

使用Audio2Face最简单的方法是使用预构建的Docker镜像：

**Linux/macOS：**
```bash
# 拉取并运行预构建镜像（CPU版本）
docker run -it \
  -p 18083:18083 \
  -v $(pwd)/weights:/workspace/audio2face/weights \
  dlp3d/audio2face:latest

# 或使用CUDA支持运行（需要支持Docker的NVIDIA GPU）
docker run -it \
  --gpus all \
  -p 18083:18083 \
  -v $(pwd)/weights:/workspace/audio2face/weights \
  dlp3d/audio2face:latest-cuda12
```

**Windows：**
```cmd
# 拉取并运行预构建镜像
docker run -it -p 18083:18083 -v .\weights:/workspace/audio2face/weights dlp3d/audio2face:latest
```

**命令说明：**
- `-p 18083:18083`：将容器的18083端口映射到主机的18083端口
- `-v $(pwd)/weights:/workspace/audio2face/weights`（Linux/macOS）：将本地`weights`目录挂载到容器的weights目录
- `-v .\weights:/workspace/audio2face/weights`（Windows）：将本地`weights`目录挂载到容器的weights目录
- `dlp3d/audio2face:latest`：使用预构建的公共镜像

**前提条件：**
- 确保项目根目录中有`weights`目录
- 确保`weights`目录中有`weights/unitalker_v0.4.0_base.onnx`文件
- 确保系统已安装并运行Docker

**替代方案：从源码构建**

如果您更喜欢从源码构建镜像：

**Linux/macOS：**
```bash
# 构建Docker镜像
docker build -f dockerfiles/Dockerfile-cpu -t audio2face:local .

# 运行容器
docker run -it \
  -p 18083:18083 \
  -v $(pwd)/weights:/workspace/audio2face/weights \
  audio2face:local
```

**Windows：**
```cmd
# 构建Docker镜像
docker build -f dockerfiles/Dockerfile-cpu -t audio2face:local .

# 运行容器
docker run -it -p 18083:18083 -v .\weights:/workspace/audio2face/weights audio2face:local
```

## 环境设置

对于本地开发和部署，请按照详细的安装指南操作：

📖 **[完整安装指南](install.md)**

安装指南提供了以下步骤说明：
- 设置Python 3.10+环境
- 安装Protocol Buffers编译器
- 配置开发环境
- 安装项目依赖

### 本地开发

按照安装指南完成环境设置后，您可以在本地启动服务：

```bash
# 激活conda环境
conda activate audio2face

# 启动服务
python main.py
```

## API文档

### 流式API

系统提供全面的实时音频到面部转换流式API：

- **StreamingAudio2FaceV1**：实时音频处理的主要流式接口
  - 基于WebSocket的通信，实现低延迟流式传输
  - 基于块的音频处理，提供连续实时响应
  - 可配置的后处理管道，支持不同情感配置
  - 异步处理，支持线程池管理
  - 请求过期和缓存管理，实现最佳性能

### 请求/响应格式

API使用Protocol Buffers进行高效序列化，支持：

- **基于块的处理**：音频输入以可配置的块进行实时响应处理
- **Blendshape输出**：面部动画数据表示为blendshape值
- **基于帧的时间线**：精确的基于帧的时间线管理，用于动画序列
- **流式协议**：基于WebSocket的流式传输，支持开始/主体/结束消息类型
- **错误处理**：全面的错误响应，包含详细的错误代码和消息

**数据流程：**
1. **音频输入**：可配置块大小的PCM音频数据
2. **特征提取**：基于Wav2Vec2的音频特征提取
3. **推理**：ONNX Unitalker模型生成blendshape预测
4. **后处理**：可配置管道应用情感配置和效果
5. **输出**：带有帧时间信息的结构化blendshape数据

## 配置

系统支持不同部署场景的灵活配置：

### 本地开发配置

对于本地开发，使用`configs/local.py`，配置：
- **CPU推理**：PyTorch仅CPU和ONNX Runtime CPU配置

### 生产配置

对于生产部署，使用`configs/cuda.py`，支持：
- **GPU加速**：CUDA 12.1支持，实现高性能推理

### 配置组件

- **特征提取器**：Wav2Vec2模型配置和音频处理参数
- **推理引擎**：ONNX Unitalker模型路径和推理设置
- **音频分割**：基于能量的静音检测参数
- **后处理**：带有情感配置和效果的可配置管道
- **服务器设置**：FastAPI服务器配置、CORS和工作器设置
- **日志**：全面的日志配置，支持多个输出目标

## 开发

### 项目结构

```
audio2face/
├── apis/                    # 流式API实现
├── data_structures/         # 核心数据模型和Protocol Buffers
├── infer/                   # 推理引擎（ONNX和PyTorch）
├── postprocess/             # 后处理模块和管道
├── service/                 # FastAPI服务器和错误处理
├── split/                   # 音频分割和分段
├── utils/                   # 工具函数和基类
└── tests/                   # 全面测试套件
```

### 测试

项目包含使用pytest和异步测试支持的全面测试。在运行测试之前，您需要准备测试输入数据。

#### 测试数据准备

下载所需的测试文件并将其组织到正确的目录结构中：

1. **创建测试输入目录：**
   ```bash
   mkdir -p input
   ```

2. **下载测试文件：**
   - **测试音频：** [test_audio.wav](https://github.com/LazyBusyYang/CatStream/releases/download/a2f_cicd_files/test_audio.wav) - 下载并保存到 `input/test_audio.wav`
   - **测试特征：** [test_feature.npy](https://github.com/LazyBusyYang/CatStream/releases/download/a2f_cicd_files/test_feature.npy) - 下载并保存到 `input/test_feature.npy`

3. **确保weights目录存在：**
   ```bash
   mkdir -p weights
   ```
   确保您有ONNX模型文件 `weights/unitalker_v0.4.0_base.onnx`（请参见[数据准备](#数据准备)部分）。

4. **最终目录结构应为：**
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

#### 运行测试

```bash
# 运行所有测试
pytest tests --log-cli-level=ERROR

# 运行特定测试类别
pytest tests/apis/           # API功能测试
pytest tests/infer/          # 推理引擎测试
pytest tests/postprocess/    # 后处理模块测试
pytest tests/data_structures/ # 数据结构测试
```

**测试覆盖：**
- **API测试**：流式API功能
- **推理测试**：ONNX Unitalker和PyTorch特征提取器验证
- **后处理测试**：所有后处理模块的各种配置
- **数据结构测试**：FaceClip类功能和格式转换

### 代码质量

项目通过以下方式保持高代码质量：

- **代码检查**：使用Ruff进行代码风格和质量检查
- **类型提示**：完整的类型注解支持
- **CI/CD**：自动化测试和部署管道

## 引用

本项目使用UniTalker算法进行面部动画生成：

```bibtex
@article{unitalker2024,
  title={UniTalker: Scaling up Audio-Driven 3D Facial Animation through A Unified Model},
  journal={ECCV},
  year={2024}
}
```

**参考**：[UniTalker GitHub仓库](https://github.com/X-niper/UniTalker) - ECCV 2024

## 许可证

本项目采用MIT许可证。详情请参见[LICENSE](LICENSE)文件。

MIT许可证是一个宽松的自由软件许可证，允许您使用、复制、修改、合并、发布、分发、再许可和/或销售软件副本，限制很少。唯一的要求是在所有副本或软件的重要部分中必须包含原始版权声明和许可证文本。

---
