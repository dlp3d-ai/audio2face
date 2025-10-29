# 数据准备

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

## 目录结构说明

- `weights/`：存储ONNX模型文件的文件夹
- `weights/unitalker_v0.4.0_base.onnx`：用于音频到面部转换的主要ONNX模型文件

