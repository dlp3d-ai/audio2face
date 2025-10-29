# Data Preparation

To use Audio2Face, you need to download the ONNX model file and set up the required directory structure.

## Download ONNX Model

### Step 1: Download the Model File

Download the ONNX model file from one of the following sources:

- **GitHub Download**: [unitalker_v0.4.0_base.onnx](https://github.com/LazyBusyYang/CatStream/releases/download/a2f_cicd_files/unitalker_v0.4.0_base.onnx)
- **Google Drive Download**: [unitalker_v0.4.0_base.onnx](https://drive.google.com/file/d/1E0NTrsh4mciRPb265n64Dd5vR3Sa7Dgx/view?usp=drive_link)
- **Baidu Cloud**: [unitalker_v0.4.0_base.onnx](https://pan.baidu.com/s/1A_vUj_ZBMFPbO1lgUYVCPA?pwd=shre)

### Step 2: Organize the Data

1. **Create the weights directory:**
   - Create a `weights` directory in your project root if it doesn't exist
   - This directory will store the ONNX model files

2. **Place the model file:**
   - Place the downloaded `unitalker_v0.4.0_base.onnx` file in the `weights` directory
   - Ensure the file is named exactly `unitalker_v0.4.0_base.onnx`

3. **Verify directory structure:**
   - Ensure the following directory structure is created:

```
├─audio2face
├─configs
├─docs
└─weights
   └─unitalker_v0.4.0_base.onnx
```

## Directory Structure

The following directory structure is required for Audio2Face to function properly:

- **`weights/`**: A folder for storing ONNX model files
  - This directory must exist in the project root
  - All model files should be placed in this directory

- **`weights/unitalker_v0.4.0_base.onnx`**: The main ONNX model file for audio-to-face conversion
  - This is the core inference model used by the service
  - The file must be placed in the `weights` directory
  - Ensure the file is not corrupted and is the correct version (v0.4.0)

## Verification

After downloading and organizing the data, you can verify the setup:

```bash
# Check if the weights directory exists
ls -la weights/

# Verify the model file is present
ls -la weights/unitalker_v0.4.0_base.onnx
```

## Next Steps

Once you have completed the data preparation:

1. **Follow the Quick Start**: See [Quick Start](quick_start.md) for detailed quick start instructions
2. **Follow the Installation Guide**: See [Installation Guide](install.md) for detailed environment setup instructions
3. **Local Development**: Set up your local development environment as described in the installation guide

