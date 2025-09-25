from abc import ABC, abstractmethod
from typing import Any

from ..utils.super import Super


class BaseSplit(Super, ABC):
    """Base class for audio splitters.

    Defines the common interface for audio splitters. All concrete splitting
    algorithms should inherit from this class and implement the __call__ method.
    Used for splitting long audio into multiple segments.
    """

    def __init__(self, logger_cfg: None | dict[str, Any] = None):
        """Initialize the BaseSplit base class.

        Args:
            logger_cfg (None | dict[str, Any], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        ABC.__init__(self)
        Super.__init__(self, logger_cfg)

    @abstractmethod
    def __call__(
            self,
            pcm_bytes: bytes,
            sample_rate: int = 16000,
            sample_width: int = 2,
            n_channels: int = 1,
            not_before: float = 0.0,
            interval_lowerbound: float = 0.5,
            **kwargs) -> list[int]:
        """Find all split points in PCM audio data.

        Abstract method that must be implemented by subclasses to define
        specific splitting algorithms. Returns an empty list if no split
        points are found in the audio.

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
                Additional keyword arguments.

        Returns:
            list[int]:
                List of split points, where each element is a byte index
                position in pcm_bytes.
        """
        pass
