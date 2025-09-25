import json

import numpy as np
import pytest

from audio2face.data_structures.face_clip import FaceClip
from audio2face.postprocess.unitalker_random_blink import UnitalkerRandomBlink


@pytest.fixture
def unitalker_blendshape_names() -> list[str]:
    """Load Unitalker blendshape names from configuration file.

    Loads the list of blendshape names used by Unitalker from the
    configuration JSON file for testing purposes.

    Returns:
        list[str]: List of blendshape names used by Unitalker.
    """
    with open('configs/unitalker_output_names.json') as f:
        return json.load(f)


@pytest.fixture
def test_face_clip(unitalker_blendshape_names: list[str]) -> FaceClip:
    """Create a test FaceClip instance using configuration blendshape names.

    Creates a FaceClip instance with random data using the blendshape
    names from the Unitalker configuration for testing random blink
    functionality.

    Args:
        unitalker_blendshape_names (list[str]): List of blendshape names
            from Unitalker configuration.

    Returns:
        FaceClip: Test FaceClip instance with 51 blendshapes and 300 frames.
    """
    # Create random data, 51 blendshapes, 300 frames
    blendshape_values = np.random.random((300, 51)).astype(np.float16)

    face_clip = FaceClip(
        blendshape_names=unitalker_blendshape_names,
        blendshape_values=blendshape_values,
        dtype=np.float16
    )
    return face_clip


@pytest.fixture
def unitalker_random_blink() -> UnitalkerRandomBlink:
    """Create a UnitalkerRandomBlink instance for testing.

    Creates a UnitalkerRandomBlink instance with default FPS setting
    for testing random blink functionality.

    Returns:
        UnitalkerRandomBlink: Configured UnitalkerRandomBlink instance
            with 30.0 FPS.
    """
    return UnitalkerRandomBlink(fps=30.0)


def test_init(unitalker_random_blink: UnitalkerRandomBlink):
    """Test UnitalkerRandomBlink initialization.

    Verifies that UnitalkerRandomBlink is initialized correctly with
    the expected FPS value and logger attribute.

    Args:
        unitalker_random_blink (UnitalkerRandomBlink): UnitalkerRandomBlink
            instance to test.
    """
    assert unitalker_random_blink.fps == 30.0
    assert hasattr(unitalker_random_blink, 'logger')


def test_init_with_logger_cfg():
    """Test UnitalkerRandomBlink initialization with logger configuration.

    Verifies that UnitalkerRandomBlink can be initialized with custom
    logger configuration and FPS setting.
    """
    logger_cfg = dict()
    blink = UnitalkerRandomBlink(fps=25.0, logger_cfg=logger_cfg)
    assert blink.fps == 25.0

def test_call_normal_case(unitalker_random_blink: UnitalkerRandomBlink,
                         test_face_clip: FaceClip):
    """Test normal case random blink processing.

    Verifies that UnitalkerRandomBlink correctly processes FaceClip data,
    applying random blink effects to eye-related blendshapes while
    preserving the overall structure.

    Args:
        unitalker_random_blink (UnitalkerRandomBlink): UnitalkerRandomBlink
            instance for testing.
        test_face_clip (FaceClip): FaceClip instance to process.
    """
    result = unitalker_random_blink(test_face_clip)

    # Verify that result is a FaceClip instance
    assert isinstance(result, FaceClip)
    assert result is not test_face_clip  # Should be a clone

    # Verify that blendshape_names remain unchanged
    assert result.blendshape_names == test_face_clip.blendshape_names

    # Verify that data shape remains unchanged
    assert result.blendshape_values.shape == test_face_clip.blendshape_values.shape

    # Verify that some frames have blink values modified, first two columns are
    # eye-related blendshapes
    # Due to randomness, we check if any values are modified to 0.4, 0.6, or 1.0
    eye_values = result.blendshape_values[:, 0:2]
    assert np.any(eye_values == 0.4) or \
        np.any(eye_values == 0.6) or \
        np.any(eye_values == 1.0)


def test_call_with_different_fps(test_face_clip: FaceClip):
    """Test UnitalkerRandomBlink with different FPS values.

    Verifies that UnitalkerRandomBlink works correctly with different
    FPS settings and produces valid results.

    Args:
        test_face_clip (FaceClip): FaceClip instance to test.
    """
    blink_30 = UnitalkerRandomBlink(fps=30.0)
    blink_60 = UnitalkerRandomBlink(fps=60.0)

    result_30 = blink_30(test_face_clip)
    result_60 = blink_60(test_face_clip)

    # Verify that both return FaceClip instances
    assert isinstance(result_30, FaceClip)
    assert isinstance(result_60, FaceClip)


def test_call_with_single_frame(unitalker_blendshape_names: list[str]):
    """Test UnitalkerRandomBlink processing with single frame data.

    Verifies that UnitalkerRandomBlink can handle FaceClip instances
    with only one frame of data.

    Args:
        unitalker_blendshape_names (list[str]): List of blendshape names
            from Unitalker configuration.
    """
    # Create data with only 1 frame
    blendshape_values = np.random.random((1, 51)).astype(np.float16)
    single_frame_clip = FaceClip(
        blendshape_names=unitalker_blendshape_names,
        blendshape_values=blendshape_values,
        dtype=np.float16
    )

    blink = UnitalkerRandomBlink(fps=30.0)
    result = blink(single_frame_clip)

    # Verify that result is returned normally
    assert isinstance(result, FaceClip)
    assert len(result) == 1


def test_call_with_wrong_blendshape_count():
    """Test UnitalkerRandomBlink with incorrect blendshape count.

    Verifies that UnitalkerRandomBlink raises a ValueError when the
    FaceClip has an incorrect number of blendshapes (50 instead of 51).
    """
    # Create 50 blendshapes, should be 51
    blendshape_names = [f'BlendShape_{i}' for i in range(50)]
    blendshape_values = np.random.random((10, 50)).astype(np.float16)

    wrong_clip = FaceClip(
        blendshape_names=blendshape_names,
        blendshape_values=blendshape_values,
        dtype=np.float16
    )

    blink = UnitalkerRandomBlink(fps=30.0)

    with pytest.raises(ValueError, match='blendshape_names length is incorrect'):
        blink(wrong_clip)


def test_call_without_mouth_shrug_upper():
    """Test UnitalkerRandomBlink without required MouthShrugUpper blendshape.

    Verifies that UnitalkerRandomBlink raises a ValueError when the
    FaceClip does not contain the required MouthShrugUpper blendshape.
    """
    # Create 51 blendshapes but without MouthShrugUpper
    blendshape_names = [f'BlendShape_{i}' for i in range(51)]
    blendshape_values = np.random.random((10, 51)).astype(np.float16)

    wrong_clip = FaceClip(
        blendshape_names=blendshape_names,
        blendshape_values=blendshape_values,
        dtype=np.float16
    )

    blink = UnitalkerRandomBlink(fps=30.0)

    with pytest.raises(ValueError, match='MouthShrugUpper not in blendshape_names'):
        blink(wrong_clip)


def test_call_with_xrmogen_npz(unitalker_blendshape_names: list[str]):
    """Test UnitalkerRandomBlink with randomly generated FaceClip input.

    Verifies that UnitalkerRandomBlink can process randomly generated
    FaceClip data with proper blendshape names.

    Args:
        unitalker_blendshape_names (list[str]): List of blendshape names
            from Unitalker configuration.
    """
    blink = UnitalkerRandomBlink(fps=30.0)

    # Create random FaceClip
    blendshape_values = np.random.random((20, 51)).astype(np.float16)
    face_clip = FaceClip(
        blendshape_names=unitalker_blendshape_names,
        blendshape_values=blendshape_values,
        dtype=np.float16
    )

    result = blink(face_clip)

    # Verify that result is a FaceClip instance
    assert isinstance(result, FaceClip)
    assert result is not face_clip

    # Verify that data shape remains unchanged
    assert result.blendshape_values.shape == face_clip.blendshape_values.shape

    # Verify that blendshape_names remain unchanged
    assert result.blendshape_names == face_clip.blendshape_names


def test_call_preserves_original_data(test_face_clip: FaceClip):
    """Test that UnitalkerRandomBlink preserves original data.

    Verifies that UnitalkerRandomBlink does not modify the original
    FaceClip data and returns a new instance.

    Args:
        test_face_clip (FaceClip): FaceClip instance to test.
    """
    blink = UnitalkerRandomBlink(fps=30.0)

    # Save original data
    original_values = test_face_clip.blendshape_values.copy()
    original_names = test_face_clip.blendshape_names.copy()

    blink(test_face_clip)

    # Verify that original data is not modified
    np.testing.assert_array_equal(test_face_clip.blendshape_values, original_values)
    assert test_face_clip.blendshape_names == original_names


def test_call_with_kwargs(test_face_clip: FaceClip):
    """Test UnitalkerRandomBlink call with additional keyword arguments.

    Verifies that UnitalkerRandomBlink can handle additional keyword
    arguments without errors.

    Args:
        test_face_clip (FaceClip): FaceClip instance to test.
    """
    blink = UnitalkerRandomBlink(fps=30.0)

    # Call with additional parameters
    result = blink(test_face_clip, extra_param='test')

    # Verify that result is returned normally
    assert isinstance(result, FaceClip)


def test_blink_pattern_consistency(test_face_clip: FaceClip):
    """Test blink pattern consistency in UnitalkerRandomBlink.

    Verifies that UnitalkerRandomBlink produces consistent blink patterns
    with expected blink values (0.4, 0.6, or 1.0) for eye blendshapes.

    Args:
        test_face_clip (FaceClip): FaceClip instance to test.
    """
    blink = UnitalkerRandomBlink(fps=30.0)

    result = blink(test_face_clip)

    # Check blink pattern: some frames should have blink values set to specific values
    eye_values = result.blendshape_values[:, 0:2]

    # Check if there are blink effects (values set to 0.4, 0.6, or 1.0)
    has_blink_04 = np.any(eye_values == 0.4)
    has_blink_06 = np.any(eye_values == 0.6)
    has_blink_10 = np.any(eye_values == 1.0)

    # At least one type of blink effect should be present
    assert has_blink_04 or has_blink_06 or has_blink_10


def test_call_with_zero_fps(unitalker_blendshape_names: list[str]):
    """Test UnitalkerRandomBlink with zero FPS (edge case).

    Verifies that UnitalkerRandomBlink raises a ZeroDivisionError when
    initialized with zero FPS, which would cause division by zero.

    Args:
        unitalker_blendshape_names (list[str]): List of blendshape names
            from Unitalker configuration.
    """
    blendshape_values = np.random.random((10, 51)).astype(np.float16)

    test_clip = FaceClip(
        blendshape_names=unitalker_blendshape_names,
        blendshape_values=blendshape_values,
        dtype=np.float16
    )

    blink = UnitalkerRandomBlink(fps=0.0)

    # Should handle normally, but may not have blink effects
    with pytest.raises(ZeroDivisionError):
        blink(test_clip)


def test_call_with_very_high_fps(test_face_clip: FaceClip):
    """Test UnitalkerRandomBlink with very high FPS value.

    Verifies that UnitalkerRandomBlink can handle very high FPS values
    without errors and produces valid results.

    Args:
        test_face_clip (FaceClip): FaceClip instance to test.
    """
    blink = UnitalkerRandomBlink(fps=1000.0)

    result = blink(test_face_clip)

    # Verify that result is returned normally
    assert isinstance(result, FaceClip)
    assert result.blendshape_values.shape == test_face_clip.blendshape_values.shape
