import asyncio
import io
import wave
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor

from ..utils.super import Super


class TorchFeatureExtractor(Super):
    """PyTorch-based audio feature extractor.

    Uses Wav2Vec2 model to extract features from PCM audio data.
    Supports asynchronous processing and thread pool concurrent execution.
    """

    def __init__(self,
                 pretrained_path: str,
                 max_workers: int = 1,
                 thread_pool_executor: ThreadPoolExecutor | None = None,
                 logger_cfg: None | dict[str, Any] = None) -> None:
        """Initialize the TorchFeatureExtractor feature extractor.

        Args:
            pretrained_path (str):
                Path to the Wav2Vec2 pretrained model.
            max_workers (int, optional):
                Maximum number of worker threads. Defaults to 1.
            thread_pool_executor (ThreadPoolExecutor | None, optional):
                Thread pool executor. If None, creates a new thread pool
                based on max_workers. Defaults to None.
            logger_cfg (None | dict[str, Any], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        super().__init__(logger_cfg=logger_cfg)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            pretrained_path)
        if thread_pool_executor is None:
            self.thread_pool_executor = ThreadPoolExecutor(max_workers=max_workers)
        else:
            self.thread_pool_executor = thread_pool_executor
        self.lock = Lock()

    async def warmup(self) -> None:
        """Warm up the model by processing random audio data to initialize model state.

        Uses randomly generated audio data for inference to ensure the model is
        loaded into memory before first use, avoiding delays in subsequent inference.
        """
        coroutines = list()
        # 1 is enough for this class
        n_coroutines = 1
        for _ in range(n_coroutines):
            duration = 2.0
            sample_rate = 16000
            audio_array = np.random.randint(
                -32768, 32767, int(duration * sample_rate), dtype=np.int16)
            coroutine = self.infer(
                audio_array,
                sample_rate=sample_rate)
            coroutines.append(coroutine)
        await asyncio.gather(*coroutines)

    async def infer(
            self,
            pcm_bytes: bytes,
            sample_rate: int = 16000,
            n_channels: int = 1,
            sample_width: int = 2) -> np.ndarray:
        """Asynchronously extract features from PCM audio data.

        Converts PCM byte data to audio array, then uses
        Wav2Vec2 model to extract features.

        Args:
            pcm_bytes (bytes):
                PCM format audio byte data.
            sample_rate (int, optional):
                Audio sample rate, must be 16000. Defaults to 16000.
            n_channels (int, optional):
                Number of audio channels, must be 1 (mono). Defaults to 1.
            sample_width (int, optional):
                Audio sample width, must be 2 (16-bit). Defaults to 2.

        Returns:
            np.ndarray:
                Extracted audio features with shape (1, feature_dim).

        Raises:
            ValueError: When audio parameters do not meet requirements
                (sample rate, number of channels, sample width).
        """
        if sample_rate != 16000:
            msg = f'sample_rate must be 16000, but got {sample_rate}'
            self.logger.error(msg)
            raise ValueError(msg)
        if n_channels != 1:
            msg = f'n_channels must be 1, but got {n_channels}'
            self.logger.error(msg)
            raise ValueError(msg)
        if sample_width != 2:
            msg = f'sample_width must be 2, but got {sample_width}'
            self.logger.error(msg)
            raise ValueError(msg)
        loop = asyncio.get_event_loop()
        audio_array = await loop.run_in_executor(
            self.thread_pool_executor,
            self._convert_pcm_to_array,
            pcm_bytes, sample_rate, n_channels, sample_width)
        audio_feature = await loop.run_in_executor(
            self.thread_pool_executor,
            self._extract_feature,
            audio_array, sample_rate)
        return audio_feature

    def _convert_pcm_to_array(
            self,
            pcm_bytes: bytes,
            sample_rate: int,
            n_channels: int,
            sample_width: int) -> np.ndarray:
        """Convert PCM byte data to audio array.

        Args:
            pcm_bytes (bytes):
                PCM format audio byte data.
            sample_rate (int):
                Audio sample rate.
            n_channels (int):
                Number of audio channels.
            sample_width (int):
                Audio sample width.

        Returns:
            np.ndarray:
                Converted audio array with shape (n_samples,).
        """
        wave_io = io.BytesIO()
        with wave.open(wave_io, 'wb') as wav_file:
            wav_file.setnchannels(n_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)
        wave_io.seek(0)
        audio_array, _ = librosa.load(wave_io, sr=sample_rate)
        return audio_array

    def _extract_feature(
            self,
            audio_array: np.ndarray,
            sample_rate: int) -> np.ndarray:
        """Extract audio features using Wav2Vec2 model.

        Args:
            audio_array (np.ndarray):
                Audio array with shape (n_samples,).
            sample_rate (int):
                Audio sample rate.

        Returns:
            np.ndarray:
                Extracted audio features with shape (1, feature_dim).
        """
        with torch.no_grad():
            with self.lock:
                audio_feature = np.squeeze(
                    self.feature_extractor(
                        audio_array,
                        sampling_rate=sample_rate).input_values)
            audio_feature = audio_feature.reshape(1, -1)
        return audio_feature
