import asyncio
from typing import Any

import numpy as np

from ..utils.super import Super


class DummyFeatureExtractor(Super):
    """Dummy feature extractor.

    A dummy audio feature extractor for testing and debugging purposes.
    Does not perform actual audio feature extraction, but returns zero-filled
    feature arrays. Supports asynchronous processing and delay simulation.
    """

    def __init__(self,
                 delay_time: float = 0.0,
                 logger_cfg: None | dict[str, Any] = None,
                 **kwargs
                 ) -> None:
        """Initialize the dummy feature extractor.

        Args:
            delay_time (float):
                Time in seconds to simulate processing delay. Defaults to 0.0.
            logger_cfg (None | dict[str, Any], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        super().__init__(logger_cfg=logger_cfg)
        self.delay_time = delay_time

    async def warmup(self) -> None:
        """Warm up the model.

        Dummy warmup method that performs no actual operations.
        """
        pass

    async def infer(
            self,
            pcm_bytes: bytes,
            **kwargs) -> np.ndarray:
        """Asynchronously extract features from PCM audio data.

        Dummy feature extraction method that generates zero-filled feature
        arrays with corresponding shapes based on input audio data size.

        Args:
            pcm_bytes (bytes):
                PCM format audio byte data.
            **kwargs:
                Additional optional parameters, currently unused.

        Returns:
            np.ndarray:
                Dummy audio feature array with shape (1, input_size/2),
                data type is float32.
        """
        input_size = len(pcm_bytes)
        audio_feature = np.zeros((1, int(input_size / 2)), dtype=np.float32)
        if self.delay_time > 0:
            await asyncio.sleep(self.delay_time)
        return audio_feature
