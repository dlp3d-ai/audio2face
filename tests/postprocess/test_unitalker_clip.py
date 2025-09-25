import json

import numpy as np
import pytest

from audio2face.data_structures.face_clip import FaceClip
from audio2face.postprocess.unitalker_clip import UnitalkerClip


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
    names from the Unitalker configuration for testing purposes.

    Args:
        unitalker_blendshape_names (list[str]): List of blendshape names
            from Unitalker configuration.

    Returns:
        FaceClip: Test FaceClip instance with 51 blendshapes and 30 frames.
    """
    # Create random data, 51 blendshapes, 30 frames
    blendshape_values = np.random.random((30, 51)).astype(np.float16)

    face_clip = FaceClip(
        blendshape_names=unitalker_blendshape_names,
        blendshape_values=blendshape_values,
        dtype=np.float16
    )
    return face_clip


@pytest.fixture
def unitalker_clip() -> UnitalkerClip:
    """Create a UnitalkerClip instance for testing.

    Creates a UnitalkerClip instance with default parameters for
    testing the Unitalker clipping functionality.

    Returns:
        UnitalkerClip: Configured UnitalkerClip instance.
    """
    return UnitalkerClip()


def test_init(unitalker_clip: UnitalkerClip):
    """Test UnitalkerClip initialization.

    Verifies that UnitalkerClip is initialized correctly with
    the required logger attribute.

    Args:
        unitalker_clip (UnitalkerClip): UnitalkerClip instance to test.
    """
    assert hasattr(unitalker_clip, 'logger')


def test_init_with_logger_cfg():
    """Test UnitalkerClip initialization with logger configuration.

    Verifies that UnitalkerClip can be initialized with custom
    logger configuration and has the required logger attribute.
    """
    logger_cfg = dict()
    clip = UnitalkerClip(logger_cfg=logger_cfg)
    assert hasattr(clip, 'logger')


def test_call_normal_case(unitalker_clip: UnitalkerClip, test_face_clip: FaceClip):
    """Test normal case processing with UnitalkerClip.

    Verifies that UnitalkerClip correctly processes FaceClip data,
    applying appropriate clipping and scaling to specific blendshapes
    while preserving the overall structure.

    Args:
        unitalker_clip (UnitalkerClip): UnitalkerClip instance for testing.
        test_face_clip (FaceClip): FaceClip instance to process.
    """
    result = unitalker_clip(test_face_clip)

    # Verify that result is a FaceClip instance
    assert isinstance(result, FaceClip)
    assert result is not test_face_clip  # Should be a clone

    # Verify that blendshape_names remain unchanged
    assert result.blendshape_names == test_face_clip.blendshape_names

    # Verify that data shape remains unchanged
    assert result.blendshape_values.shape == test_face_clip.blendshape_values.shape

    # Verify that eye-related blendshapes are processed (indices 2:14)
    eye_values = result.blendshape_values[:, 2:14]
    # All values should be in [0, 0.8] range
    assert np.all(eye_values >= 0)
    assert np.all(eye_values <= 0.8)

    # Verify that MouthShrugUpper is processed (index 45)
    mouth_shrug_values = result.blendshape_values[:, 45]
    # All values should be in [0, 0.01] range
    assert np.all(mouth_shrug_values >= 0)
    assert np.all(mouth_shrug_values <= 0.01)

    # Verify that EyeBlinkLeft is processed (index 0)
    eye_blink_left_values = result.blendshape_values[:, 0]
    # All values should be in [0, 0.01] range
    assert np.all(eye_blink_left_values >= 0)
    assert np.all(eye_blink_left_values <= 0.01)

    # Verify that EyeBlinkRight is processed (index 1)
    eye_blink_right_values = result.blendshape_values[:, 1]
    # All values should be in [0, 0.01] range
    assert np.all(eye_blink_right_values >= 0)
    assert np.all(eye_blink_right_values <= 0.01)


def test_call_with_single_frame(unitalker_blendshape_names: list[str]):
    """Test UnitalkerClip processing with single frame data.

    Verifies that UnitalkerClip can handle FaceClip instances
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

    clip = UnitalkerClip()
    result = clip(single_frame_clip)

    # Verify that result is returned normally
    assert isinstance(result, FaceClip)
    assert len(result) == 1


def test_call_with_wrong_blendshape_count():
    """Test UnitalkerClip with incorrect blendshape count.

    Verifies that UnitalkerClip raises a ValueError when the
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

    clip = UnitalkerClip()

    with pytest.raises(ValueError, match='blendshape_names length is incorrect'):
        clip(wrong_clip)


def test_call_without_mouth_shrug_upper():
    """Test UnitalkerClip without required MouthShrugUpper blendshape.

    Verifies that UnitalkerClip raises a ValueError when the
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

    clip = UnitalkerClip()

    with pytest.raises(ValueError, match='MouthShrugUpper not in blendshape_names'):
        clip(wrong_clip)


def test_call_with_xrmogen_npz(unitalker_blendshape_names: list[str]):
    """Test UnitalkerClip with randomly generated FaceClip input.

    Verifies that UnitalkerClip can process randomly generated
    FaceClip data with proper blendshape names.

    Args:
        unitalker_blendshape_names (list[str]): List of blendshape names
            from Unitalker configuration.
    """
    clip = UnitalkerClip()

    # Create random FaceClip
    blendshape_values = np.random.random((20, 51)).astype(np.float16)
    face_clip = FaceClip(
        blendshape_names=unitalker_blendshape_names,
        blendshape_values=blendshape_values,
        dtype=np.float16
    )

    result = clip(face_clip)

    # Verify that result is a FaceClip instance
    assert isinstance(result, FaceClip)
    assert result is not face_clip

    # Verify that data shape remains unchanged
    assert result.blendshape_values.shape == face_clip.blendshape_values.shape

    # Verify that blendshape_names remain unchanged
    assert result.blendshape_names == face_clip.blendshape_names


def test_call_preserves_original_data(test_face_clip: FaceClip):
    """Test that UnitalkerClip preserves original data.

    Verifies that UnitalkerClip does not modify the original
    FaceClip data and returns a new instance.

    Args:
        test_face_clip (FaceClip): FaceClip instance to test.
    """
    clip = UnitalkerClip()

    # Save original data
    original_values = test_face_clip.blendshape_values.copy()
    original_names = test_face_clip.blendshape_names.copy()

    clip(test_face_clip)

    # Verify that original data is not modified
    np.testing.assert_array_equal(test_face_clip.blendshape_values, original_values)
    assert test_face_clip.blendshape_names == original_names


def test_call_with_kwargs(test_face_clip: FaceClip):
    """Test UnitalkerClip call with additional keyword arguments.

    Verifies that UnitalkerClip can handle additional keyword
    arguments without errors.

    Args:
        test_face_clip (FaceClip): FaceClip instance to test.
    """
    clip = UnitalkerClip()

    # Call with additional parameters
    result = clip(test_face_clip, extra_param='test')

    # Verify that result is returned normally
    assert isinstance(result, FaceClip)


def test_eye_blendshape_processing(unitalker_blendshape_names: list[str]):
    """Test eye-related blendshape processing by UnitalkerClip.

    Verifies that UnitalkerClip correctly clips and scales
    eye-related blendshapes (indices 2:14) to the [0, 0.8] range.

    Args:
        unitalker_blendshape_names (list[str]): List of blendshape names
            from Unitalker configuration.
    """
    # Create test data with eye-related blendshapes having out-of-range values
    # Create test data with eye-related blendshapes (indices 2:14) having
    # negative and >1 values
    blendshape_values = np.random.random((5, 51)).astype(np.float16)
    # Out-of-range values
    blendshape_values[:, 2:14] = np.array([-0.5, 0.3, 1.5, 0.8, 0.1])[:, None]

    test_clip = FaceClip(
        blendshape_names=unitalker_blendshape_names,
        blendshape_values=blendshape_values,
        dtype=np.float16
    )

    clip = UnitalkerClip()
    result = clip(test_clip)

    # Verify that eye-related blendshapes are correctly clipped and scaled
    eye_values = result.blendshape_values[:, 2:14]
    assert np.all(eye_values >= 0)  # All values should be >= 0
    assert np.all(eye_values <= 0.8)  # All values should be <= 0.8


def test_mouth_shrug_processing(unitalker_blendshape_names: list[str]):
    """Test MouthShrugUpper blendshape processing by UnitalkerClip.

    Verifies that UnitalkerClip correctly clips MouthShrugUpper
    blendshape (index 45) to the [0, 0.01] range.

    Args:
        unitalker_blendshape_names (list[str]): List of blendshape names
            from Unitalker configuration.
    """
    # Create test data with MouthShrugUpper having out-of-range values
    # Create test data with MouthShrugUpper (index 45) having values > 0.01
    blendshape_values = np.random.random((5, 51)).astype(np.float16)
    blendshape_values[:, 45] = np.array([0.5, 0.02, 0.01, 0.0, 0.1])  # Out-of-range

    test_clip = FaceClip(
        blendshape_names=unitalker_blendshape_names,
        blendshape_values=blendshape_values,
        dtype=np.float16
    )

    clip = UnitalkerClip()
    result = clip(test_clip)

    # Verify that MouthShrugUpper is correctly clipped
    mouth_shrug_values = result.blendshape_values[:, 45]
    assert np.all(mouth_shrug_values >= 0)  # All values should be >= 0
    assert np.all(mouth_shrug_values <= 0.01)  # All values should be <= 0.01


def test_eye_blink_processing(unitalker_blendshape_names: list[str]):
    """Test eye blink blendshape processing by UnitalkerClip.

    Verifies that UnitalkerClip correctly clips eye blink blendshapes
    (EyeBlinkLeft and EyeBlinkRight, indices 0 and 1) to the [0, 0.01] range.

    Args:
        unitalker_blendshape_names (list[str]): List of blendshape names
            from Unitalker configuration.
    """
    # Create test data with eye blink blendshapes having out-of-range values
    # Create test data with EyeBlinkLeft (index 0) and EyeBlinkRight (index 1)
    # having values > 0.01
    blendshape_values = np.random.random((5, 51)).astype(np.float16)
    blendshape_values[:, 0] = np.array([0.5, 0.02, 0.01, 0.0, 0.1])  # EyeBlinkLeft
    blendshape_values[:, 1] = np.array([0.3, 0.015, 0.005, 0.0, 0.08])  # EyeBlinkRight

    test_clip = FaceClip(
        blendshape_names=unitalker_blendshape_names,
        blendshape_values=blendshape_values,
        dtype=np.float16
    )

    clip = UnitalkerClip()
    result = clip(test_clip)

    # Verify that EyeBlinkLeft is correctly clipped
    eye_blink_left_values = result.blendshape_values[:, 0]
    assert np.all(eye_blink_left_values >= 0)  # All values should be >= 0
    assert np.all(eye_blink_left_values <= 0.01)  # All values should be <= 0.01

    # Verify that EyeBlinkRight is correctly clipped
    eye_blink_right_values = result.blendshape_values[:, 1]
    assert np.all(eye_blink_right_values >= 0)  # All values should be >= 0
    assert np.all(eye_blink_right_values <= 0.01)  # All values should be <= 0.01


def test_clipping_behavior(unitalker_blendshape_names: list[str]):
    """Test specific clipping behavior of UnitalkerClip.

    Verifies the exact clipping and scaling behavior of UnitalkerClip
    on various blendshape values, including boundary conditions.

    Args:
        unitalker_blendshape_names (list[str]): List of blendshape names
            from Unitalker configuration.
    """
    # Create test data with various boundary values
    blendshape_values = np.zeros((3, 51), dtype=np.float16)

    # Set some test values
    blendshape_values[0, 2:14] = 1.0  # Set eye-related blendshapes to 1.0
    blendshape_values[0, 45] = 0.5    # Set MouthShrugUpper to 0.5
    blendshape_values[0, 0] = 0.8     # Set EyeBlinkLeft to 0.8
    blendshape_values[0, 1] = 0.6     # Set EyeBlinkRight to 0.6

    blendshape_values[1, 2:14] = 0.5  # Set eye-related blendshapes to 0.5
    blendshape_values[1, 45] = 0.01   # Set MouthShrugUpper to 0.01
    blendshape_values[1, 0] = 0.01    # Set EyeBlinkLeft to 0.01
    blendshape_values[1, 1] = 0.01    # Set EyeBlinkRight to 0.01

    blendshape_values[2, 2:14] = 0.0  # Set eye-related blendshapes to 0.0
    blendshape_values[2, 45] = 0.0    # Set MouthShrugUpper to 0.0
    blendshape_values[2, 0] = 0.0     # Set EyeBlinkLeft to 0.0
    blendshape_values[2, 1] = 0.0     # Set EyeBlinkRight to 0.0

    test_clip = FaceClip(
        blendshape_names=unitalker_blendshape_names,
        blendshape_values=blendshape_values,
        dtype=np.float16
    )

    clip = UnitalkerClip()
    result = clip(test_clip)

    # Verify processing results for first frame
    # Eye-related blendshapes: 1.0 -> 0.8 (clipped to 1.0 then multiplied by 0.8)
    assert np.allclose(result.blendshape_values[0, 2:14], 0.8)
    # MouthShrugUpper: 0.5 -> 0.01 (clipped to 0.01)
    assert np.allclose(result.blendshape_values[0, 45], 0.01)
    # EyeBlinkLeft: 0.8 -> 0.01 (clipped to 0.01)
    assert np.allclose(result.blendshape_values[0, 0], 0.01)
    # EyeBlinkRight: 0.6 -> 0.01 (clipped to 0.01)
    assert np.allclose(result.blendshape_values[0, 1], 0.01)

    # Verify processing results for second frame
    # Eye-related blendshapes: 0.5 -> 0.4 (0.5 * 0.8)
    assert np.allclose(result.blendshape_values[1, 2:14], 0.4)
    # Other values remain unchanged
    assert np.allclose(result.blendshape_values[1, 45], 0.01)
    assert np.allclose(result.blendshape_values[1, 0], 0.01)
    assert np.allclose(result.blendshape_values[1, 1], 0.01)

    # Verify processing results for third frame
    # All values should remain 0
    assert np.allclose(result.blendshape_values[2, 2:14], 0.0)
    assert np.allclose(result.blendshape_values[2, 45], 0.0)
    assert np.allclose(result.blendshape_values[2, 0], 0.0)
    assert np.allclose(result.blendshape_values[2, 1], 0.0)
