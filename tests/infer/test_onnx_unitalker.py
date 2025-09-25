import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import onnxruntime
import pytest

from audio2face.data_structures.face_clip import FaceClip
from audio2face.infer.onnx_unitalker import (
    OnnxUnitalker,
    UnitalkerInputInvalid,
    UnitalkerInputTooShort,
)

UNITALKER_BLENDSHAPE_NAMES_PATH = 'configs/unitalker_output_names.json'

def is_cuda_provider_available() -> bool:
    """Check if CUDA provider is available.

    Checks if the CUDA execution provider is available in ONNX Runtime
    by verifying provider availability, cuDNN library presence, and
    library loading capability.

    Returns:
        bool: True if CUDA provider is available and functional,
            False otherwise.
    """
    try:
        # Check if ONNX Runtime CUDA provider is available
        providers = onnxruntime.get_available_providers()
        if 'CUDAExecutionProvider' not in providers:
            return False

        # Check cuDNN related environment variables and library files
        import ctypes
        import os

        # Check common cuDNN library files
        cudnn_libs = [
            'libcudnn.so',
            'libcudnn.so.8',
            'libcudnn.so.7',
            'libcudnn.so.6'
        ]

        # Check libraries in LD_LIBRARY_PATH
        ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
        library_paths = ld_library_path.split(':') if ld_library_path else []

        # Add system default library paths
        library_paths.extend([
            '/usr/lib', '/usr/lib64', '/usr/local/lib', '/usr/local/lib64'
        ])

        cudnn_found = False
        for lib_name in cudnn_libs:
            for lib_path in library_paths:
                if lib_path and os.path.exists(os.path.join(lib_path, lib_name)):
                    cudnn_found = True
                    break
            if cudnn_found:
                break

        # If cuDNN library is not found, return False
        if not cudnn_found:
            return False

        # Try to load cuDNN library to verify its availability
        try:
            ctypes.CDLL('libcudnn.so')
        except (OSError, ImportError):
            return False

        return True

    except Exception:
        return False


@pytest.fixture
def model_path() -> str:
    """Fixture for ONNX model path.

    Provides the path to the Unitalker ONNX model file for testing.

    Returns:
        str: Path to the Unitalker ONNX model file.
    """
    return 'weights/unitalker_v0.4.0_base.onnx'


@pytest.fixture
def test_feature_path() -> str:
    """Fixture for test audio feature file path.

    Provides the path to the test audio feature file for testing.

    Returns:
        str: Path to the test audio feature file.
    """
    return 'input/test_feature.npy'


@pytest.fixture
def test_audio_feature(test_feature_path: str) -> np.ndarray:
    """Fixture for test audio feature data.

    Loads test audio feature data from file or creates mock data
    if the file does not exist.

    Args:
        test_feature_path (str): Path to the test audio feature file.

    Returns:
        np.ndarray: Audio feature data with shape [1, 100, 768].
    """
    if os.path.exists(test_feature_path):
        return np.load(test_feature_path)
    else:
        # Create a mock audio feature for testing
        return np.random.randn(1, 100, 768).astype(np.float32)


@pytest.fixture
def onnx_unitalker_cuda(model_path: str) -> OnnxUnitalker:
    """Fixture for creating OnnxUnitalker instance with CUDA provider.

    Creates an OnnxUnitalker instance configured to use CUDA execution
    provider for GPU-accelerated inference.

    Args:
        model_path (str): Path to the ONNX model file.

    Returns:
        OnnxUnitalker: Configured OnnxUnitalker instance with CUDA provider.
    """
    return OnnxUnitalker(
        blendshape_names=UNITALKER_BLENDSHAPE_NAMES_PATH,
        model_path=model_path,
        max_workers=2,
        onnx_providers='CUDAExecutionProvider'
    )


@pytest.fixture
def onnx_unitalker_cpu(model_path: str) -> OnnxUnitalker:
    """Fixture for creating OnnxUnitalker instance with CPU provider.

    Creates an OnnxUnitalker instance configured to use CPU execution
    provider for CPU-based inference.

    Args:
        model_path (str): Path to the ONNX model file.

    Returns:
        OnnxUnitalker: Configured OnnxUnitalker instance with CPU provider.
    """
    return OnnxUnitalker(
        blendshape_names=UNITALKER_BLENDSHAPE_NAMES_PATH,
        model_path=model_path,
        max_workers=3,
        onnx_providers='CPUExecutionProvider'
    )


@pytest.fixture
def custom_thread_pool() -> ThreadPoolExecutor:
    """Fixture for custom thread pool executor.

    Creates a custom ThreadPoolExecutor for testing thread pool
    management functionality.

    Returns:
        ThreadPoolExecutor: Custom thread pool executor with 3 workers.
    """
    return ThreadPoolExecutor(max_workers=3)


@pytest.fixture
def onnx_unitalker_custom_pool(
        model_path: str,
        custom_thread_pool: ThreadPoolExecutor) -> OnnxUnitalker:
    """Fixture for creating OnnxUnitalker instance with custom thread pool.

    Creates an OnnxUnitalker instance using a custom thread pool
    executor for testing custom thread pool integration.

    Args:
        model_path (str): Path to the ONNX model file.
        custom_thread_pool (ThreadPoolExecutor): Custom thread pool executor.

    Returns:
        OnnxUnitalker: Configured OnnxUnitalker instance with custom thread pool.
    """
    return OnnxUnitalker(
        blendshape_names=UNITALKER_BLENDSHAPE_NAMES_PATH,
        model_path=model_path,
        thread_pool_executor=custom_thread_pool,
        onnx_providers='CPUExecutionProvider'
    )


@pytest.fixture
def logger_cfg() -> dict[str, Any]:
    """Fixture for logger configuration.

    Provides a logger configuration dictionary for testing
    logger initialization.

    Returns:
        dict[str, Any]: Logger configuration dictionary.
    """
    return dict()


@pytest.fixture
def onnx_unitalker_with_logger(
        model_path: str,
        logger_cfg: dict[str, Any]) -> OnnxUnitalker:
    """Fixture for creating OnnxUnitalker instance with custom logger.

    Creates an OnnxUnitalker instance with custom logger configuration
    for testing logger functionality.

    Args:
        model_path (str): Path to the ONNX model file.
        logger_cfg (dict[str, Any]): Logger configuration dictionary.

    Returns:
        OnnxUnitalker: Configured OnnxUnitalker instance with custom logger.
    """
    return OnnxUnitalker(
        blendshape_names=UNITALKER_BLENDSHAPE_NAMES_PATH,
        model_path=model_path,
        max_workers=1,
        onnx_providers='CPUExecutionProvider',
        logger_cfg=logger_cfg
    )


class TestOnnxUnitalker:
    """Test class for OnnxUnitalker.

    This test class covers initialization, inference functionality,
    error handling, and resource management for the OnnxUnitalker class.
    """

    def test_init_with_max_workers(self, model_path: str):
        """Test initialization with max_workers parameter.

        Verifies that OnnxUnitalker correctly initializes with the
        specified number of worker sessions and busy status tracking.

        Args:
            model_path (str): Path to the ONNX model file.
        """
        unitalker = OnnxUnitalker(
            blendshape_names=UNITALKER_BLENDSHAPE_NAMES_PATH,
            model_path=model_path,
            max_workers=3,
            onnx_providers='CPUExecutionProvider'
        )
        assert unitalker.max_workers == 3
        assert len(unitalker.unitalker_sessions) == 3
        assert len(unitalker.session_busy) == 3
        assert all(not busy for busy in unitalker.session_busy)

    def test_init_with_custom_thread_pool(
            self,
            model_path: str,
            custom_thread_pool: ThreadPoolExecutor):
        """Test initialization with custom thread pool executor.

        Verifies that OnnxUnitalker correctly uses a custom thread pool
        executor when provided.

        Args:
            model_path (str): Path to the ONNX model file.
            custom_thread_pool (ThreadPoolExecutor): Custom thread pool executor.
        """
        unitalker = OnnxUnitalker(
            blendshape_names=UNITALKER_BLENDSHAPE_NAMES_PATH,
            model_path=model_path,
            thread_pool_executor=custom_thread_pool,
            onnx_providers='CPUExecutionProvider'
        )
        assert unitalker.thread_pool_executor == custom_thread_pool
        assert unitalker.max_workers == 1
        assert len(unitalker.unitalker_sessions) == 1

    def test_init_with_cuda_provider(self, model_path: str):
        """Test initialization with CUDA provider.

        Verifies that OnnxUnitalker correctly initializes with CUDA
        execution provider when available.

        Args:
            model_path (str): Path to the ONNX model file.
        """
        # Skip test if CUDA provider is not available
        if not is_cuda_provider_available():
            pytest.skip('CUDAExecutionProvider is not available')

        unitalker = OnnxUnitalker(
            blendshape_names=UNITALKER_BLENDSHAPE_NAMES_PATH,
            model_path=model_path,
            max_workers=1,
            onnx_providers='CUDAExecutionProvider'
        )
        assert len(unitalker.unitalker_sessions) == 1
        # Check if session uses CUDA provider
        session = unitalker.unitalker_sessions[0]
        providers = session.get_providers()
        assert 'CUDAExecutionProvider' in providers

    def test_init_with_cpu_provider(self, model_path: str):
        """Test initialization with CPU provider.

        Verifies that OnnxUnitalker correctly initializes with CPU
        execution provider.

        Args:
            model_path (str): Path to the ONNX model file.
        """
        unitalker = OnnxUnitalker(
            blendshape_names=UNITALKER_BLENDSHAPE_NAMES_PATH,
            model_path=model_path,
            max_workers=1,
            onnx_providers='CPUExecutionProvider'
        )
        assert len(unitalker.unitalker_sessions) == 1
        # Check if session uses CPU provider
        session = unitalker.unitalker_sessions[0]
        providers = session.get_providers()
        assert 'CPUExecutionProvider' in providers

    def test_init_with_logger_cfg(self, model_path: str, logger_cfg: dict[str, Any]):
        """Test initialization with logger configuration.

        Verifies that OnnxUnitalker correctly initializes with custom
        logger configuration.

        Args:
            model_path (str): Path to the ONNX model file.
            logger_cfg (dict[str, Any]): Logger configuration dictionary.
        """
        unitalker = OnnxUnitalker(
            blendshape_names=UNITALKER_BLENDSHAPE_NAMES_PATH,
            model_path=model_path,
            max_workers=1,
            onnx_providers='CPUExecutionProvider',
            logger_cfg=logger_cfg
        )
        assert unitalker.logger is not None

    def test_init_with_none_logger_cfg(self, model_path: str):
        """Test initialization with None logger configuration.

        Verifies that OnnxUnitalker correctly initializes with default
        logger configuration when None is provided.

        Args:
            model_path (str): Path to the ONNX model file.
        """
        unitalker = OnnxUnitalker(
            blendshape_names=UNITALKER_BLENDSHAPE_NAMES_PATH,
            model_path=model_path,
            max_workers=1,
            onnx_providers='CPUExecutionProvider',
            logger_cfg=None
        )
        assert unitalker.logger is not None
        assert unitalker.logger_cfg['logger_name'] == 'OnnxUnitalker'

    def test_init_invalid_model_path(self):
        """Test initialization with invalid model path.

        Verifies that OnnxUnitalker raises FileNotFoundError when
        initialized with a non-existent model path.
        """
        with pytest.raises(FileNotFoundError):
            OnnxUnitalker(
                blendshape_names=UNITALKER_BLENDSHAPE_NAMES_PATH,
                model_path='invalid/path/model.onnx',
                max_workers=1,
                onnx_providers='CPUExecutionProvider'
            )

    @pytest.mark.asyncio
    async def test_infer_success(
            self,
            onnx_unitalker_cpu: OnnxUnitalker,
            test_audio_feature: np.ndarray):
        """Test successful inference.

        Verifies that OnnxUnitalker can successfully perform inference
        and return a valid FaceClip with correct shape and data type.

        Args:
            onnx_unitalker_cpu (OnnxUnitalker): OnnxUnitalker instance for testing.
            test_audio_feature (np.ndarray): Test audio feature data.
        """
        emotion_id = 10  # neutral
        fps = 30.0
        sample_rate = 16000
        timeout = 30.0

        result = await onnx_unitalker_cpu.infer(
            audio_feature=test_audio_feature,
            emotion_id=emotion_id,
            fps=fps,
            sample_rate=sample_rate,
            timeout=timeout
        )

        assert isinstance(result, FaceClip)
        assert result.dtype == np.float16
        # Check output shape, n_blendshapes is usually 51
        expected_frames = int(test_audio_feature.shape[1] * fps // sample_rate)
        assert result.blendshape_values.shape[0] == expected_frames  # n_frames
        assert result.blendshape_values.shape[1] == 51  # n_blendshapes

    @pytest.mark.asyncio
    async def test_infer_with_short_audio(
            self,
            onnx_unitalker_cpu: OnnxUnitalker,
            test_audio_feature: np.ndarray):
        """Test inference with short audio feature.

        Verifies that OnnxUnitalker raises UnitalkerInputTooShort when
        provided with audio feature that is too short for processing.

        Args:
            onnx_unitalker_cpu (OnnxUnitalker): OnnxUnitalker instance for testing.
            test_audio_feature (np.ndarray): Test audio feature data.
        """
        slice_audio_feature = test_audio_feature[:, :100]
        emotion_id = 10  # neutral
        fps = 30.0
        sample_rate = 16000
        timeout = 30.0

        with pytest.raises(UnitalkerInputTooShort):
            await onnx_unitalker_cpu.infer(
                audio_feature=slice_audio_feature,
                emotion_id=emotion_id,
                fps=fps,
                sample_rate=sample_rate,
                timeout=timeout
            )

    @pytest.mark.asyncio
    async def test_infer_with_different_emotion(
            self,
            onnx_unitalker_cpu: OnnxUnitalker,
            test_audio_feature: np.ndarray):
        """Test inference with different emotion ID.

        Verifies that OnnxUnitalker can handle different emotion IDs
        and produce valid results.

        Args:
            onnx_unitalker_cpu (OnnxUnitalker): OnnxUnitalker instance for testing.
            test_audio_feature (np.ndarray): Test audio feature data.
        """
        emotion_id = 5  # Different emotion ID
        fps = 30.0
        sample_rate = 16000

        result = await onnx_unitalker_cpu.infer(
            audio_feature=test_audio_feature,
            emotion_id=emotion_id,
            fps=fps,
            sample_rate=sample_rate
        )

        assert isinstance(result, FaceClip)
        assert result.blendshape_values.shape[1] == 51  # n_blendshapes

    @pytest.mark.asyncio
    async def test_infer_with_different_fps(
            self,
            onnx_unitalker_cpu: OnnxUnitalker,
            test_audio_feature: np.ndarray):
        """Test inference with different FPS.

        Verifies that OnnxUnitalker can handle different FPS values
        and produce results with correct frame counts.

        Args:
            onnx_unitalker_cpu (OnnxUnitalker): OnnxUnitalker instance for testing.
            test_audio_feature (np.ndarray): Test audio feature data.
        """
        emotion_id = 10
        fps = 60.0  # Different frame rate
        sample_rate = 16000

        result = await onnx_unitalker_cpu.infer(
            audio_feature=test_audio_feature,
            emotion_id=emotion_id,
            fps=fps,
            sample_rate=sample_rate
        )

        assert isinstance(result, FaceClip)
        expected_frames = int(test_audio_feature.shape[1] * fps // sample_rate)
        assert result.blendshape_values.shape[0] == expected_frames

    @pytest.mark.asyncio
    async def test_infer_with_different_sample_rate(
            self,
            onnx_unitalker_cpu: OnnxUnitalker,
            test_audio_feature: np.ndarray):
        """Test inference with different sample rate.

        Verifies that OnnxUnitalker can handle different sample rates
        and produce results with correct frame counts.

        Args:
            onnx_unitalker_cpu (OnnxUnitalker): OnnxUnitalker instance for testing.
            test_audio_feature (np.ndarray): Test audio feature data.
        """
        emotion_id = 10
        fps = 30.0
        sample_rate = 8000  # Different sample rate

        result = await onnx_unitalker_cpu.infer(
            audio_feature=test_audio_feature,
            emotion_id=emotion_id,
            fps=fps,
            sample_rate=sample_rate
        )

        assert isinstance(result, FaceClip)
        expected_frames = int(test_audio_feature.shape[1] * fps // sample_rate)
        assert result.blendshape_values.shape[0] == expected_frames

    @pytest.mark.asyncio
    async def test_infer_concurrent_access(
            self,
            onnx_unitalker_cpu: OnnxUnitalker,
            test_audio_feature: np.ndarray):
        """Test concurrent access to multiple sessions.

        Verifies that OnnxUnitalker can handle multiple concurrent
        inference requests using different sessions.

        Args:
            onnx_unitalker_cpu (OnnxUnitalker): OnnxUnitalker instance for testing.
            test_audio_feature (np.ndarray): Test audio feature data.
        """
        emotion_id = 10
        fps = 30.0
        sample_rate = 16000

        # Create multiple concurrent tasks
        tasks = []
        for _ in range(3):
            task = onnx_unitalker_cpu.infer(
                audio_feature=test_audio_feature,
                emotion_id=emotion_id,
                fps=fps,
                sample_rate=sample_rate,
                timeout=5
            )
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        # Verify all results
        for result in results:
            assert isinstance(result, FaceClip)
            assert result.blendshape_values.shape[1] == 51

    @pytest.mark.asyncio
    async def test_infer_with_invalid_audio_feature(
            self,
            onnx_unitalker_cpu: OnnxUnitalker):
        """Test inference with invalid audio feature.

        Verifies that OnnxUnitalker raises UnitalkerInputInvalid when
        provided with audio feature that has incorrect shape.

        Args:
            onnx_unitalker_cpu (OnnxUnitalker): OnnxUnitalker instance for testing.
        """
        # Create invalid audio feature
        invalid_audio = np.random.randn(1, 16000, 16000)  # Wrong shape
        emotion_id = 10
        fps = 30.0
        sample_rate = 16000

        with pytest.raises(UnitalkerInputInvalid):
            await onnx_unitalker_cpu.infer(
                audio_feature=invalid_audio,
                emotion_id=emotion_id,
                fps=fps,
                sample_rate=sample_rate
            )

    @pytest.mark.asyncio
    async def test_infer_session_release(
            self,
            onnx_unitalker_cpu: OnnxUnitalker,
            test_audio_feature: np.ndarray):
        """Test that session is properly released after inference.

        Verifies that OnnxUnitalker properly releases sessions after
        inference completion, allowing them to be reused.

        Args:
            onnx_unitalker_cpu (OnnxUnitalker): OnnxUnitalker instance for testing.
            test_audio_feature (np.ndarray): Test audio feature data.
        """
        emotion_id = 10
        fps = 30.0
        sample_rate = 16000

        # Check initial state
        assert all(not busy for busy in onnx_unitalker_cpu.session_busy)

        # Execute inference
        await onnx_unitalker_cpu.infer(
            audio_feature=test_audio_feature,
            emotion_id=emotion_id,
            fps=fps,
            sample_rate=sample_rate
        )

        # Check that session is released after inference
        assert all(not busy for busy in onnx_unitalker_cpu.session_busy)

    @pytest.mark.asyncio
    async def test_infer_with_exception_handling(
            self,
            onnx_unitalker_cpu: OnnxUnitalker):
        """Test that session is released even when exception occurs.

        Verifies that OnnxUnitalker properly releases sessions even
        when exceptions occur during inference.

        Args:
            onnx_unitalker_cpu (OnnxUnitalker): OnnxUnitalker instance for testing.
        """
        emotion_id = 10
        fps = 30.0
        sample_rate = 16000

        # Check initial state
        assert all(not busy for busy in onnx_unitalker_cpu.session_busy)

        # Create an audio feature that will cause an error (e.g., wrong shape)
        invalid_audio = np.random.randn(1, 10, 1000).astype(np.float32)
        # Wrong feature dimension

        try:
            await onnx_unitalker_cpu.infer(
                audio_feature=invalid_audio,
                emotion_id=emotion_id,
                fps=fps,
                sample_rate=sample_rate
            )
        except Exception:
            # Even if there is an error, session should be released
            pass

        # Check that session is released
        assert all(not busy for busy in onnx_unitalker_cpu.session_busy)

    def test_thread_pool_executor_cleanup(self, model_path: str):
        """Test thread pool executor cleanup.

        Verifies that OnnxUnitalker correctly creates and manages
        thread pool executors.

        Args:
            model_path (str): Path to the ONNX model file.
        """
        unitalker = OnnxUnitalker(
            blendshape_names=UNITALKER_BLENDSHAPE_NAMES_PATH,
            model_path=model_path,
            max_workers=2,
            onnx_providers='CPUExecutionProvider'
        )

        # Check if thread pool is created
        assert unitalker.thread_pool_executor is not None
        assert unitalker.thread_pool_executor._max_workers == 2

    def test_custom_thread_pool_executor_cleanup(
            self,
            model_path: str,
            custom_thread_pool: ThreadPoolExecutor):
        """Test custom thread pool executor handling.

        Verifies that OnnxUnitalker correctly uses custom thread pool
        executors when provided.

        Args:
            model_path (str): Path to the ONNX model file.
            custom_thread_pool (ThreadPoolExecutor): Custom thread pool executor.
        """
        unitalker = OnnxUnitalker(
            blendshape_names=UNITALKER_BLENDSHAPE_NAMES_PATH,
            model_path=model_path,
            thread_pool_executor=custom_thread_pool,
            onnx_providers='CPUExecutionProvider'
        )

        # Check using custom thread pool
        assert unitalker.thread_pool_executor == custom_thread_pool
        assert unitalker.max_workers > 0
