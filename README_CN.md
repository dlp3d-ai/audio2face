# Audio2Face

> [English Documentation](README.md) | **中文文档**

## 目录

- [概述](#概述)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [文档](#文档)
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
   - **百度网盘：** [unitalker_v0.4.0_base.onnx](https://pan.baidu.com/s/1A_vUj_ZBMFPbO1lgUYVCPA?pwd=shre)

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
```bash
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

## 文档

有关详细文档，请访问我们的完整文档站点：

📖 **[完整文档](https://dlp3d.readthedocs.io/zh-cn/latest/_subrepos/audio2face/overview.html)**

文档提供以下详细信息：

- **概述**：详细的系统架构和主要特性
- **数据准备**：下载和组织模型文件的分步指南
- **快速开始**：全面的Docker设置和本地开发说明
- **安装指南**：适用于Linux和Windows的详细环境设置
- **API文档**：完整的流式API参考，包含请求/响应格式
- **配置**：不同部署场景的配置选项
- **开发**：项目结构、测试指南和代码质量标准

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
