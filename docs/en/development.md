# Development

## Project Structure

```
audio2face/
├── apis/                    # Streaming API implementations
├── data_structures/         # Core data models and Protocol Buffers
├── infer/                   # Inference engines (ONNX and PyTorch)
├── postprocess/             # Postprocessing modules and pipelines
├── service/                 # FastAPI server and error handling
├── split/                   # Audio splitting and segmentation
├── utils/                   # Utility functions and base classes
└── tests/                   # Comprehensive test suite
```

## Testing

The project includes comprehensive tests with pytest and async testing support. Before running tests, you need to prepare the test input data.

### Test Data Preparation

Download the required test files and organize them in the correct directory structure:

1. **Create test input directory:**
   ```bash
   mkdir -p input
   ```

2. **Download test files:**
   - **Test Audio:** [test_audio.wav](https://github.com/LazyBusyYang/CatStream/releases/download/a2f_cicd_files/test_audio.wav) - Download and save to `input/test_audio.wav`
   - **Test Feature:** [test_feature.npy](https://github.com/LazyBusyYang/CatStream/releases/download/a2f_cicd_files/test_feature.npy) - Download and save to `input/test_feature.npy`

3. **Ensure weights directory exists:**
   ```bash
   mkdir -p weights
   ```
   Make sure you have the ONNX model file `weights/unitalker_v0.4.0_base.onnx` (see [Data Preparation](data_preparation.md) section).

4. **Final directory structure should be:**
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

### Running Tests

```bash
# Run all tests
pytest tests --log-cli-level=ERROR

# Run specific test categories
pytest tests/apis/           # API functionality tests
pytest tests/infer/          # Inference engine tests
pytest tests/postprocess/    # Postprocessing module tests
pytest tests/data_structures/ # Data structure tests
```

### Test Coverage

- **API Tests**: Streaming API functionality
- **Inference Tests**: ONNX Unitalker and PyTorch feature extractor validation
- **Postprocessing Tests**: All postprocessing modules with various configurations
- **Data Structure Tests**: FaceClip class functionality and format conversions

## Code Quality

The project maintains high code quality with:

- **Linting**: Ruff for code style and quality checks
- **Type Hints**: Full type annotation support
- **CI/CD**: Automated testing and deployment pipelines

