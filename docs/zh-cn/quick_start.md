# 快速开始

## 使用Docker

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

## 替代方案：从源码构建

如果您更希望从源码构建镜像：

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
```bash
# 构建Docker镜像
docker build -f dockerfiles/Dockerfile-cpu -t audio2face:local .

# 运行容器
docker run -it -p 18083:18083 -v .\weights:/workspace/audio2face/weights audio2face:local
```
