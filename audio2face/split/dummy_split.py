import time
from typing import Any

from .base_split import BaseSplit


class DummySplit(BaseSplit):
    """Dummy audio splitter for testing and debugging.

    A dummy audio splitter used for testing and debugging purposes.
    Does not perform actual audio segmentation analysis, but returns
    the audio end as the only split point. Supports delay simulation.
    """

    def __init__(
            self,
            delay_time: float = 0.0,
            logger_cfg: None | dict[str, Any] = None,
            **kwargs):
        """Initialize the dummy audio splitter.

        Args:
            delay_time (float, optional):
                Time in seconds to simulate processing delay.
                Defaults to 0.0.
            logger_cfg (None | dict[str, Any], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
            **kwargs:
                Additional keyword arguments, currently unused.
        """
        super().__init__(logger_cfg)
        self.delay_time = delay_time

    def __call__(
            self,
            pcm_bytes: bytes,
            sample_rate: int = 16000,
            sample_width: int = 2,
            n_channels: int = 1,
            not_before: float = 0.0,
            interval_lowerbound: float = 0.5,
            **kwargs) -> list[int]:
        """Dummy audio splitting method.

        Returns the audio end as the only split point without performing
        actual audio analysis.

        Args:
            pcm_bytes (bytes):
                PCM format audio byte data.
            sample_rate (int, optional):
                Audio sample rate in Hz. Defaults to 16000.
            sample_width (int, optional):
                Audio sample width in bytes. Defaults to 2 (16-bit).
            n_channels (int, optional):
                Number of audio channels. Defaults to 1 (mono).
            not_before (float, optional):
                Minimum time in seconds before the first split point
                can occur. Defaults to 0.0.
            interval_lowerbound (float, optional):
                Minimum interval in seconds between split points.
                Defaults to 0.5.
            **kwargs:
                Additional keyword arguments, currently unused.

        Returns:
            list[int]:
                List of split points containing only the audio end
                byte index position.
        """
        return_list = [len(pcm_bytes) - 1, ]
        if self.delay_time > 0:
            time.sleep(self.delay_time)
        return return_list
