from typing import Any

import numpy as np

from .base_split import BaseSplit


class EnergySplit(BaseSplit):
    """Audio splitter based on energy threshold.

    Detects silence segments by calculating short-time energy of audio,
    thereby determining audio split points. Suitable for voice activity
    detection and audio segmentation processing.
    """

    def __init__(
            self,
            energy_threshold: float = 1e4,
            window_duration: float = 0.1,
            hop_duration: float = 0.005,
            logger_cfg: None | dict[str, Any] = None):
        """Initialize the EnergySplit splitter.

        Args:
            energy_threshold (float, optional):
                Energy threshold for determining silence segments.
                Defaults to 1e4.
            window_duration (float, optional):
                Window length in seconds for energy calculation.
                Defaults to 0.1.
            hop_duration (float, optional):
                Window step size in seconds for energy calculation.
                Defaults to 0.005.
            logger_cfg (None | dict[str, Any], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        BaseSplit.__init__(self, logger_cfg)
        self.energy_threshold = energy_threshold
        self.window_duration = window_duration
        self.hop_duration = hop_duration

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

        Detects silence segments by calculating short-time energy of audio.
        When energy of consecutive windows falls below the threshold,
        a split point is considered found. Returns an empty list if no
        split points are found in the audio.

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
        # Convert byte data to numpy array
        dtype = np.int16 if sample_width == 2 else np.int8
        audio_data = np.frombuffer(pcm_bytes, dtype=dtype)

        # If multi-channel, take the first channel
        if n_channels > 1:
            audio_data = audio_data[::n_channels]

        # Calculate short-time energy
        frame_length = int(self.window_duration * sample_rate)
        hop_length = int(self.hop_duration * sample_rate)

        min_silence_frames = 1
        silence_count = 0

        # Calculate window energy from front to back
        byte_idx_lowerbound = not_before * sample_rate * sample_width * n_channels
        ret_list = list()
        for i in range(0, len(audio_data) - frame_length, hop_length):
            frame = audio_data[i:i + frame_length]
            energy = np.sum(frame.astype(np.float32)**2) / frame_length

            if energy < self.energy_threshold:
                silence_count += 1
                if silence_count >= min_silence_frames:
                    # Convert sample index to byte index
                    sample_idx = i + min_silence_frames * hop_length
                    byte_idx = sample_idx * sample_width * n_channels
                    if byte_idx >= byte_idx_lowerbound:
                        ret_list.append(byte_idx)
                        byte_idx_lowerbound = byte_idx + \
                            interval_lowerbound * sample_rate * \
                            sample_width * n_channels
            else:
                silence_count = 0
        if len(ret_list) > 0:
            self.logger.debug(
                f'Found {len(ret_list)} split points under constraints ' +
                f'not_before={not_before:.2f}s and ' +
                f'interval_lowerbound={interval_lowerbound:.2f}s: {ret_list}')
        return ret_list
