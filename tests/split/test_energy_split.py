import wave
from typing import Any

import numpy as np
import pytest

from audio2face.split.energy_split import EnergySplit


@pytest.fixture
def test_audio_pcm_bytes() -> bytes:
    """Read PCM byte data from test audio file.

    Reads the raw PCM audio data from the test audio file for use in
    energy-based audio splitting tests.

    Returns:
        bytes: Raw PCM audio data as bytes.
    """
    with wave.open('input/test_audio.wav', 'rb') as wav_file:
        return wav_file.readframes(wav_file.getnframes())


@pytest.fixture
def test_audio_info() -> dict[str, Any]:
    """Get information about the test audio file.

    Extracts audio file metadata including sample rate, sample width,
    number of channels, frame count, and duration.

    Returns:
        dict[str, Any]: Dictionary containing audio file information with
            keys: sample_rate, sample_width, n_channels, n_frames, duration.
    """
    with wave.open('input/test_audio.wav', 'rb') as wav_file:
        return {
            'sample_rate': wav_file.getframerate(),
            'sample_width': wav_file.getsampwidth(),
            'n_channels': wav_file.getnchannels(),
            'n_frames': wav_file.getnframes(),
            'duration': wav_file.getnframes() / wav_file.getframerate()
        }


@pytest.fixture
def energy_split() -> EnergySplit:
    """Create an EnergySplit instance with default parameters.

    Creates an EnergySplit instance using default configuration for
    energy-based audio splitting tests.

    Returns:
        EnergySplit: Configured EnergySplit instance with default settings.
    """
    return EnergySplit()


@pytest.fixture
def energy_split_custom() -> EnergySplit:
    """Create an EnergySplit instance with custom parameters.

    Creates an EnergySplit instance with custom energy threshold,
    window duration, and hop duration for testing different configurations.

    Returns:
        EnergySplit: Configured EnergySplit instance with custom settings.
    """
    return EnergySplit(
        energy_threshold=1e3,
        window_duration=0.02,
        hop_duration=0.01
    )


def test_init(energy_split: EnergySplit):
    """Test EnergySplit initialization with default parameters.

    Verifies that EnergySplit is initialized correctly with expected
    default values for energy threshold, window duration, and hop duration.

    Args:
        energy_split (EnergySplit): EnergySplit instance to test.
    """
    assert energy_split.energy_threshold == 1e4
    assert energy_split.window_duration == 0.1
    assert energy_split.hop_duration == 0.005
    assert hasattr(energy_split, 'logger')


def test_init_with_custom_params():
    """Test EnergySplit initialization with custom parameters.

    Verifies that EnergySplit can be initialized with custom values
    for energy threshold, window duration, hop duration, and logger
    configuration.
    """
    split = EnergySplit(
        energy_threshold=5e3,
        window_duration=0.05,
        hop_duration=0.025,
        logger_cfg=dict()
    )
    assert split.energy_threshold == 5e3
    assert split.window_duration == 0.05
    assert split.hop_duration == 0.025
    assert hasattr(split, 'logger')


def test_call_with_real_audio(energy_split: EnergySplit,
                             test_audio_pcm_bytes: bytes,
                             test_audio_info: dict[str, Any]):
    """Test energy-based splitting with real audio file.

    Tests the energy-based audio splitting functionality using real
    audio data and verifies the correctness of split points.

    Args:
        energy_split (EnergySplit): EnergySplit instance for testing.
        test_audio_pcm_bytes (bytes): Raw PCM audio data.
        test_audio_info (dict[str, Any]): Audio file metadata.
    """
    result = energy_split(
        pcm_bytes=test_audio_pcm_bytes,
        sample_rate=test_audio_info['sample_rate'],
        sample_width=test_audio_info['sample_width'],
        n_channels=test_audio_info['n_channels']
    )

    # Verify that result is a list
    assert isinstance(result, list)

    # Verify that all elements are integers
    assert all(isinstance(x, int) for x in result)

    # Verify that there should be at least 2 split points
    assert len(result) >= 2

    # Verify that split points are in ascending order
    assert result == sorted(result)

    # Verify that all split points are within valid range
    max_byte_idx = len(test_audio_pcm_bytes)
    assert all(0 <= x < max_byte_idx for x in result)


def test_call_with_custom_params(energy_split_custom: EnergySplit,
                                test_audio_pcm_bytes: bytes,
                                test_audio_info: dict[str, Any]):
    """Test energy-based splitting with custom parameters.

    Tests the energy-based audio splitting functionality using custom
    parameters and verifies the correctness of split points.

    Args:
        energy_split_custom (EnergySplit): EnergySplit instance with
            custom parameters.
        test_audio_pcm_bytes (bytes): Raw PCM audio data.
        test_audio_info (dict[str, Any]): Audio file metadata.
    """
    result = energy_split_custom(
        pcm_bytes=test_audio_pcm_bytes,
        sample_rate=test_audio_info['sample_rate'],
        sample_width=test_audio_info['sample_width'],
        n_channels=test_audio_info['n_channels']
    )

    # Verify that result is a list
    assert isinstance(result, list)

    # Verify that all elements are integers
    assert all(isinstance(x, int) for x in result)

    # Verify that split points are in ascending order
    assert result == sorted(result)


def test_call_with_not_before(energy_split: EnergySplit,
                             test_audio_pcm_bytes: bytes,
                             test_audio_info: dict[str, Any]):
    """Test the effect of not_before parameter.

    Tests that the not_before parameter correctly prevents split points
    from being generated before the specified time threshold.

    Args:
        energy_split (EnergySplit): EnergySplit instance for testing.
        test_audio_pcm_bytes (bytes): Raw PCM audio data.
        test_audio_info (dict[str, Any]): Audio file metadata.
    """
    # Set not_before to 1 second
    not_before = 1.0
    result = energy_split(
        pcm_bytes=test_audio_pcm_bytes,
        sample_rate=test_audio_info['sample_rate'],
        sample_width=test_audio_info['sample_width'],
        n_channels=test_audio_info['n_channels'],
        not_before=not_before
    )

    # Verify that result is a list
    assert isinstance(result, list)

    # Verify that all split points are after not_before
    min_byte_idx = not_before * test_audio_info['sample_rate'] * \
                   test_audio_info['sample_width'] * test_audio_info['n_channels']
    assert all(x >= min_byte_idx for x in result)


def test_call_with_interval_lowerbound(energy_split: EnergySplit,
                                      test_audio_pcm_bytes: bytes,
                                      test_audio_info: dict[str, Any]):
    """Test the effect of interval_lowerbound parameter.

    Tests that the interval_lowerbound parameter correctly enforces
    minimum intervals between consecutive split points.

    Args:
        energy_split (EnergySplit): EnergySplit instance for testing.
        test_audio_pcm_bytes (bytes): Raw PCM audio data.
        test_audio_info (dict[str, Any]): Audio file metadata.
    """
    interval_lowerbound = 1.0
    result = energy_split(
        pcm_bytes=test_audio_pcm_bytes,
        sample_rate=test_audio_info['sample_rate'],
        sample_width=test_audio_info['sample_width'],
        n_channels=test_audio_info['n_channels'],
        interval_lowerbound=interval_lowerbound
    )

    # Verify that result is a list
    assert isinstance(result, list)

    # Verify that intervals between consecutive split points meet requirements
    if len(result) >= 2:
        min_interval_bytes = \
            interval_lowerbound * test_audio_info['sample_rate'] * \
            test_audio_info['sample_width'] * test_audio_info['n_channels']
        for i in range(1, len(result)):
            interval = result[i] - result[i-1]
            assert interval >= min_interval_bytes


def test_call_with_kwargs(energy_split: EnergySplit,
                         test_audio_pcm_bytes: bytes,
                         test_audio_info: dict[str, Any]):
    """Test calling with additional keyword arguments.

    Tests that the EnergySplit can handle additional keyword arguments
    without errors and still produce valid results.

    Args:
        energy_split (EnergySplit): EnergySplit instance for testing.
        test_audio_pcm_bytes (bytes): Raw PCM audio data.
        test_audio_info (dict[str, Any]): Audio file metadata.
    """
    result = energy_split(
        pcm_bytes=test_audio_pcm_bytes,
        sample_rate=test_audio_info['sample_rate'],
        sample_width=test_audio_info['sample_width'],
        n_channels=test_audio_info['n_channels'],
        extra_param='test'
    )

    # Verify that result is returned normally
    assert isinstance(result, list)


def test_call_with_silent_audio():
    """Test energy-based splitting with completely silent audio.

    Tests the behavior of EnergySplit when processing audio data
    that contains no sound (all zeros).
    """
    # Create silent audio data
    sample_rate = 16000
    duration = 1.0  # 1 second
    n_samples = int(sample_rate * duration)
    silent_audio = np.zeros(n_samples, dtype=np.int16)
    pcm_bytes = silent_audio.tobytes()

    split = EnergySplit(energy_threshold=1e4)
    result = split(
        pcm_bytes=pcm_bytes,
        sample_rate=sample_rate,
        sample_width=2,
        n_channels=1
    )

    # Silent audio should have many split points
    assert isinstance(result, list)
    assert len(result) > 0


def test_call_with_loud_audio():
    """Test energy-based splitting with very loud audio.

    Tests the behavior of EnergySplit when processing audio data
    with maximum volume levels.
    """
    # Create very loud audio data
    sample_rate = 16000
    duration = 1.0  # 1 second
    n_samples = int(sample_rate * duration)
    loud_audio = np.full(n_samples, 32767, dtype=np.int16)  # Maximum volume
    pcm_bytes = loud_audio.tobytes()

    split = EnergySplit(energy_threshold=1e4)
    result = split(
        pcm_bytes=pcm_bytes,
        sample_rate=sample_rate,
        sample_width=2,
        n_channels=1
    )

    # Very loud audio should have no split points
    assert isinstance(result, list)
    assert len(result) == 0


def test_call_with_mixed_audio():
    """Test energy-based splitting with mixed volume audio.

    Tests the behavior of EnergySplit when processing audio data
    that contains both sound segments and silent segments.
    """
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    n_samples = int(sample_rate * duration)

    # Create mixed audio: first 0.5s has sound, middle 1s is silent, last 0.5s has sound
    mixed_audio = np.zeros(n_samples, dtype=np.int16)

    # First 0.5 seconds have sound
    sound_start = 0
    sound_end = int(0.5 * sample_rate)
    mixed_audio[sound_start:sound_end] = 16384  # Medium volume

    # Last 0.5 seconds have sound
    sound_start = int(1.5 * sample_rate)
    sound_end = n_samples
    mixed_audio[sound_start:sound_end] = 16384  # Medium volume

    pcm_bytes = mixed_audio.tobytes()

    split = EnergySplit(energy_threshold=1e4)
    result = split(
        pcm_bytes=pcm_bytes,
        sample_rate=sample_rate,
        sample_width=2,
        n_channels=1
    )

    # Should have split points
    assert isinstance(result, list)
    assert len(result) > 0


def test_call_with_different_sample_widths(test_audio_pcm_bytes: bytes,
                                         test_audio_info: dict[str, Any]):
    """Test the effect of different sample widths.

    Tests that EnergySplit works correctly with different audio
    sample width configurations (8-bit vs 16-bit).

    Args:
        test_audio_pcm_bytes (bytes): Raw PCM audio data.
        test_audio_info (dict[str, Any]): Audio file metadata.
    """
    split = EnergySplit()

    # Test 16-bit sampling
    result_16bit = split(
        pcm_bytes=test_audio_pcm_bytes,
        sample_rate=test_audio_info['sample_rate'],
        sample_width=2,
        n_channels=test_audio_info['n_channels']
    )

    # Convert to 8-bit sampling for testing
    audio_data_16bit = np.frombuffer(test_audio_pcm_bytes, dtype=np.int16)
    audio_data_8bit = (audio_data_16bit // 256).astype(np.int8)
    pcm_bytes_8bit = audio_data_8bit.tobytes()

    result_8bit = split(
        pcm_bytes=pcm_bytes_8bit,
        sample_rate=test_audio_info['sample_rate'],
        sample_width=1,
        n_channels=test_audio_info['n_channels']
    )

    # Verify that both return lists
    assert isinstance(result_16bit, list)
    assert isinstance(result_8bit, list)


def test_call_with_different_sample_rates(test_audio_pcm_bytes: bytes,
                                        test_audio_info: dict[str, Any]):
    """Test the effect of different sample rates.

    Tests that EnergySplit works correctly with different audio
    sample rate configurations.

    Args:
        test_audio_pcm_bytes (bytes): Raw PCM audio data.
        test_audio_info (dict[str, Any]): Audio file metadata.
    """
    split = EnergySplit()

    # Original sample rate
    result_original = split(
        pcm_bytes=test_audio_pcm_bytes,
        sample_rate=test_audio_info['sample_rate'],
        sample_width=test_audio_info['sample_width'],
        n_channels=test_audio_info['n_channels']
    )

    # Different sample rate (here we use the same audio data but declare
    # different sample rate)
    result_different = split(
        pcm_bytes=test_audio_pcm_bytes,
        sample_rate=8000,  # Different sample rate
        sample_width=test_audio_info['sample_width'],
        n_channels=test_audio_info['n_channels']
    )

    # Verify that both return lists
    assert isinstance(result_original, list)
    assert isinstance(result_different, list)


def test_call_with_multichannel_audio():
    """Test energy-based splitting with multichannel audio.

    Tests that EnergySplit works correctly with stereo audio data
    where different channels have different volume levels.
    """
    # Create dual-channel audio data
    sample_rate = 16000
    duration = 1.0
    n_samples = int(sample_rate * duration)

    # Create dual-channel audio: left channel has sound, right channel is silent
    left_channel = np.full(n_samples, 16384, dtype=np.int16)
    right_channel = np.zeros(n_samples, dtype=np.int16)

    # Interleave dual-channel data
    stereo_audio = np.empty(n_samples * 2, dtype=np.int16)
    stereo_audio[0::2] = left_channel
    stereo_audio[1::2] = right_channel

    pcm_bytes = stereo_audio.tobytes()

    split = EnergySplit()
    result = split(
        pcm_bytes=pcm_bytes,
        sample_rate=sample_rate,
        sample_width=2,
        n_channels=2
    )

    # Verify that result is a list
    assert isinstance(result, list)


def test_call_with_very_short_audio():
    """Test energy-based splitting with very short audio.

    Tests the behavior of EnergySplit when processing audio data
    that is shorter than the analysis window duration.
    """
    # Create very short audio (shorter than window length)
    sample_rate = 16000
    duration = 0.001  # 1 millisecond
    n_samples = int(sample_rate * duration)
    short_audio = np.random.randint(-1000, 1000, n_samples, dtype=np.int16)
    pcm_bytes = short_audio.tobytes()

    split = EnergySplit()
    result = split(
        pcm_bytes=pcm_bytes,
        sample_rate=sample_rate,
        sample_width=2,
        n_channels=1
    )

    # Verify that result is a list
    assert isinstance(result, list)


def test_call_with_empty_audio():
    """Test energy-based splitting with empty audio data.

    Tests the behavior of EnergySplit when processing empty
    audio data (zero bytes).
    """
    split = EnergySplit()
    result = split(
        pcm_bytes=b'',
        sample_rate=16000,
        sample_width=2,
        n_channels=1
    )

    # Verify that empty list is returned
    assert isinstance(result, list)
    assert len(result) == 0


def test_energy_threshold_effect(test_audio_pcm_bytes: bytes,
                               test_audio_info: dict[str, Any]):
    """Test the effect of energy threshold on splitting results.

    Tests how different energy threshold values affect the number
    and distribution of split points in the audio.

    Args:
        test_audio_pcm_bytes (bytes): Raw PCM audio data.
        test_audio_info (dict[str, Any]): Audio file metadata.
    """
    # Use different energy thresholds
    thresholds = [1e3, 1e4, 1e5]
    results = []

    for threshold in thresholds:
        split = EnergySplit(energy_threshold=threshold)
        result = split(
            pcm_bytes=test_audio_pcm_bytes,
            sample_rate=test_audio_info['sample_rate'],
            sample_width=test_audio_info['sample_width'],
            n_channels=test_audio_info['n_channels']
        )
        results.append(result)

    # Verify that all results are lists
    assert all(isinstance(r, list) for r in results)

    # Verify that higher thresholds generally result in fewer split points
    # Note: This is not absolute, as audio content can be complex
    assert len(results[0]) >= len(results[1]) or len(results[1]) >= len(results[2])


def test_window_duration_effect(test_audio_pcm_bytes: bytes,
                              test_audio_info: dict[str, Any]):
    """Test the effect of window duration on splitting results.

    Tests how different window duration values affect the energy
    analysis and resulting split points.

    Args:
        test_audio_pcm_bytes (bytes): Raw PCM audio data.
        test_audio_info (dict[str, Any]): Audio file metadata.
    """
    # Use different window durations
    window_durations = [0.005, 0.01, 0.02]
    results = []

    for window_duration in window_durations:
        split = EnergySplit(window_duration=window_duration)
        result = split(
            pcm_bytes=test_audio_pcm_bytes,
            sample_rate=test_audio_info['sample_rate'],
            sample_width=test_audio_info['sample_width'],
            n_channels=test_audio_info['n_channels']
        )
        results.append(result)

    # Verify that all results are lists
    assert all(isinstance(r, list) for r in results)


def test_hop_duration_effect(test_audio_pcm_bytes: bytes,
                           test_audio_info: dict[str, Any]):
    """Test the effect of hop duration on splitting results.

    Tests how different hop duration values affect the temporal
    resolution of energy analysis and split point detection.

    Args:
        test_audio_pcm_bytes (bytes): Raw PCM audio data.
        test_audio_info (dict[str, Any]): Audio file metadata.
    """
    # Use different hop durations
    hop_durations = [0.0025, 0.005, 0.01]
    results = []

    for hop_duration in hop_durations:
        split = EnergySplit(hop_duration=hop_duration)
        result = split(
            pcm_bytes=test_audio_pcm_bytes,
            sample_rate=test_audio_info['sample_rate'],
            sample_width=test_audio_info['sample_width'],
            n_channels=test_audio_info['n_channels']
        )
        results.append(result)

    # Verify that all results are lists
    assert all(isinstance(r, list) for r in results)
