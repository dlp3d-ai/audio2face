# 开发

## 项目结构

```
audio2face/
├── apis/                    # 流式API实现
├── data_structures/         # 核心数据模型和Protocol Buffers
├── infer/                   # 推理引擎（ONNX和PyTorch）
├── postprocess/             # 后处理模块和流水线
├── service/                 # FastAPI服务器和错误处理
├── split/                   # 音频分割和分段
├── utils/                   # 工具函数和基类
└── tests/                   # 全面测试套件
```

## 测试

项目包含使用pytest和异步测试支持的全面测试。在运行测试之前，您需要准备测试输入数据。

### 测试数据准备

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
   确保您有ONNX模型文件 `weights/unitalker_v0.4.0_base.onnx`（请参见[数据准备](data_preparation.md)部分）。

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

### 运行测试

```bash
# 运行所有测试
pytest tests --log-cli-level=ERROR

# 运行特定测试类别
pytest tests/apis/           # API功能测试
pytest tests/infer/          # 推理引擎测试
pytest tests/postprocess/    # 后处理模块测试
pytest tests/data_structures/ # 数据结构测试
```

### 测试覆盖

- **API测试**：流式API功能
- **推理测试**：ONNX Unitalker和PyTorch特征提取器验证
- **后处理测试**：所有后处理模块的各种配置
- **数据结构测试**：FaceClip类功能和格式转换

## 代码质量

项目通过以下方式保持高代码质量：

- **代码检查**：使用Ruff进行代码风格和质量检查
- **类型提示**：完整的类型注解支持
- **CI/CD**：自动化测试和部署管道

