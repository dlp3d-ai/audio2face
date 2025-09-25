import asyncio
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from transformers import Wav2Vec2FeatureExtractor

from audio2face.infer.torch_feature_extractor import TorchFeatureExtractor

pretrained_path = 'configs/wavlm-base-plus/config.json'


@pytest.fixture
def test_audio_path() -> Path:
    """Fixture for test audio file path."""
    return Path('input/test_audio.wav')


@pytest.fixture
def mock_pretrained_path() -> str:
    """Fixture for mock pretrained model path."""
    return pretrained_path


@pytest.fixture
def thread_pool_executor() -> ThreadPoolExecutor:
    """Fixture for ThreadPoolExecutor."""
    return ThreadPoolExecutor(max_workers=2)


@pytest.fixture
def logger_cfg() -> dict[str, Any]:
    """Fixture for logger configuration."""
    return dict()


@pytest.fixture
def torch_feature_extractor() -> TorchFeatureExtractor:
    """Fixture for creating a TorchFeatureExtractor instance."""
    return TorchFeatureExtractor(pretrained_path=pretrained_path)


@pytest.fixture
def torch_feature_extractor_with_executor(
    thread_pool_executor: ThreadPoolExecutor,
) -> TorchFeatureExtractor:
    """Fixture for creating a TorchFeatureExtractor instance with custom executor."""
    return TorchFeatureExtractor(
        pretrained_path=pretrained_path,
        thread_pool_executor=thread_pool_executor,
    )


@pytest.fixture
def torch_feature_extractor_with_logger(
    logger_cfg: dict[str, Any],
) -> TorchFeatureExtractor:
    """Fixture for creating a TorchFeatureExtractor instance with logger config."""
    return TorchFeatureExtractor(
        pretrained_path=pretrained_path,
        logger_cfg=logger_cfg,
    )


@pytest.fixture
def sample_pcm_bytes() -> bytes:
    """Fixture for sample PCM bytes."""
    # Create a simple sine wave as PCM data
    sample_rate = 16000
    duration = 0.1  # 100ms
    frequency = 440  # A4 note
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t)
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    return audio_data.tobytes()


class TestTorchFeatureExtractor:
    """Test class for TorchFeatureExtractor."""

    def test_init_default_params(self) -> None:
        """Test initialization with default parameters."""
        extractor = TorchFeatureExtractor(pretrained_path=pretrained_path)
        assert extractor.feature_extractor is not None
        assert isinstance(extractor.thread_pool_executor, ThreadPoolExecutor)
        assert extractor.thread_pool_executor._max_workers == 1

    def test_init_custom_max_workers(self) -> None:
        """Test initialization with custom max_workers."""
        extractor = TorchFeatureExtractor(
            pretrained_path=pretrained_path,
            max_workers=4,
        )
        assert extractor.thread_pool_executor._max_workers == 4

    def test_init_custom_executor(
        self,
        thread_pool_executor: ThreadPoolExecutor,
    ) -> None:
        """Test initialization with custom thread pool executor."""
        extractor = TorchFeatureExtractor(
            pretrained_path=pretrained_path,
            thread_pool_executor=thread_pool_executor,
        )
        assert extractor.thread_pool_executor is thread_pool_executor

    def test_init_with_logger_cfg(
        self,
        logger_cfg: dict[str, Any],
    ) -> None:
        """Test initialization with logger configuration."""
        extractor = TorchFeatureExtractor(
            pretrained_path=pretrained_path,
            logger_cfg=logger_cfg,
        )
        assert extractor.logger is not None

    @pytest.mark.asyncio
    async def test_infer_success(
        self,
        torch_feature_extractor: TorchFeatureExtractor,
        sample_pcm_bytes: bytes,
    ) -> None:
        """Test successful inference."""
        result = await torch_feature_extractor.infer(sample_pcm_bytes)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[0] == 1

    @pytest.mark.asyncio
    async def test_infer_with_custom_params(
        self,
        torch_feature_extractor: TorchFeatureExtractor,
        sample_pcm_bytes: bytes,
    ) -> None:
        """Test inference with custom parameters."""
        result = await torch_feature_extractor.infer(
            pcm_bytes=sample_pcm_bytes,
            sample_rate=16000,
            n_channels=1,
            sample_width=2,
        )
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2

    @pytest.mark.asyncio
    async def test_infer_invalid_n_channels(
        self,
        torch_feature_extractor: TorchFeatureExtractor,
        sample_pcm_bytes: bytes,
    ) -> None:
        """Test inference with invalid n_channels raises ValueError."""
        with pytest.raises(ValueError, match='n_channels must be 1, but got 2'):
            await torch_feature_extractor.infer(
                pcm_bytes=sample_pcm_bytes,
                n_channels=2,
            )

    @pytest.mark.asyncio
    async def test_infer_invalid_sample_width(
        self,
        torch_feature_extractor: TorchFeatureExtractor,
        sample_pcm_bytes: bytes,
    ) -> None:
        """Test inference with invalid sample_width raises ValueError."""
        with pytest.raises(ValueError, match='sample_width must be 2, but got 1'):
            await torch_feature_extractor.infer(
                pcm_bytes=sample_pcm_bytes,
                sample_width=1,
            )

    def test_convert_pcm_to_array(
        self,
        torch_feature_extractor: TorchFeatureExtractor,
        sample_pcm_bytes: bytes,
    ) -> None:
        """Test PCM to array conversion."""
        result = torch_feature_extractor._convert_pcm_to_array(
            pcm_bytes=sample_pcm_bytes,
            sample_rate=16000,
            n_channels=1,
            sample_width=2,
        )
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert len(result) > 0

    def test_convert_pcm_to_array_different_sample_rate(
        self,
        torch_feature_extractor: TorchFeatureExtractor,
        sample_pcm_bytes: bytes,
    ) -> None:
        """Test PCM to array conversion with different sample rate."""
        result = torch_feature_extractor._convert_pcm_to_array(
            pcm_bytes=sample_pcm_bytes,
            sample_rate=8000,
            n_channels=1,
            sample_width=2,
        )
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    def test_extract_feature(
        self,
        torch_feature_extractor: TorchFeatureExtractor,
    ) -> None:
        """Test feature extraction."""
        # Create a simple audio array
        sample_rate = 16000
        duration = 0.1
        audio_array = np.sin(
            2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))

        result = torch_feature_extractor._extract_feature(
            audio_array=audio_array,
            sample_rate=sample_rate,
        )
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[0] == 1

    def test_extract_feature_different_sample_rate(
        self,
        torch_feature_extractor: TorchFeatureExtractor,
    ) -> None:
        """Test feature extraction with different sample rate."""
        sample_rate = 8000
        duration = 0.1
        audio_array = np.sin(
            2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        with pytest.raises(ValueError):
            torch_feature_extractor._extract_feature(
                audio_array=audio_array,
                sample_rate=sample_rate,
            )

    @pytest.mark.asyncio
    async def test_infer_with_real_audio_file(
        self,
        torch_feature_extractor: TorchFeatureExtractor,
        test_audio_path: Path,
    ) -> None:
        """Test inference with real audio file."""
        if not test_audio_path.exists():
            pytest.skip(f'Test audio file not found: {test_audio_path}')

        # Read the WAV file and convert to PCM bytes
        with wave.open(str(test_audio_path), 'rb') as wav_file:
            pcm_bytes = wav_file.readframes(wav_file.getnframes())
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()

        result = await torch_feature_extractor.infer(
            pcm_bytes=pcm_bytes,
            sample_rate=sample_rate,
            n_channels=n_channels,
            sample_width=sample_width,
        )
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[0] == 1

    def test_thread_pool_executor_cleanup(self) -> None:
        """Test that ThreadPoolExecutor is properly managed."""
        extractor = TorchFeatureExtractor(pretrained_path=pretrained_path)
        executor = extractor.thread_pool_executor

        # Test that executor is working
        assert not executor._shutdown

        # Cleanup
        executor.shutdown(wait=True)
        assert executor._shutdown

    @pytest.mark.asyncio
    async def test_concurrent_inference(
        self,
        torch_feature_extractor: TorchFeatureExtractor,
        sample_pcm_bytes: bytes,
    ) -> None:
        """Test concurrent inference operations."""
        tasks = []
        for _ in range(3):
            task = torch_feature_extractor.infer(sample_pcm_bytes)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        for result in results:
            assert isinstance(result, np.ndarray)
            assert result.ndim == 2
            assert result.shape[0] == 1

    def test_feature_extractor_model_type(
        self,
        torch_feature_extractor: TorchFeatureExtractor,
    ) -> None:
        """Test that the feature extractor is the correct type."""
        assert isinstance(
            torch_feature_extractor.feature_extractor,
            Wav2Vec2FeatureExtractor)

    @pytest.mark.asyncio
    async def test_infer_empty_pcm_bytes(
        self,
        torch_feature_extractor: TorchFeatureExtractor,
    ) -> None:
        """Test inference with empty PCM bytes."""
        empty_pcm = b''
        result = await torch_feature_extractor.infer(empty_pcm)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
