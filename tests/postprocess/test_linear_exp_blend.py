import json
import os
import tempfile
from typing import Any

import numpy as np
import pytest

from audio2face.data_structures.face_clip import FaceClip
from audio2face.postprocess.linear_exp_blend import (
    LinearExpBlend,
    _linear_exp_blend_batch,
)


@pytest.fixture
def sample_face_clip() -> FaceClip:
    """Create a sample FaceClip instance for testing.

    Creates a FaceClip with predefined blendshape names and values
    for testing linear exponential blending functionality.

    Returns:
        FaceClip: Sample FaceClip instance with 4 blendshapes and 3 frames.
    """
    blendshape_names = [
        'eye_blink_left', 'eye_blink_right', 'mouth_smile', 'brow_up'
    ]
    blendshape_values = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 0.1, 0.2]
    ])
    return FaceClip(
        blendshape_names=blendshape_names,
        blendshape_values=blendshape_values)


@pytest.fixture
def linear_exp_blend_params() -> dict[str, Any]:
    """Create test parameters for LinearExpBlend.

    Creates a dictionary containing all necessary parameters for
    initializing a LinearExpBlend instance for testing.

    Returns:
        dict[str, Any]: Dictionary containing LinearExpBlend parameters
            including name, offset, normalize_reference, exponential_strength,
            blend_weight, bs_names, and logger_cfg.
    """
    return {
        'name': 'test_processor',
        'offset': 0.1,
        'normalize_reference': 1.0,
        'exponential_strength': 2.0,
        'blend_weight': 0.5,
        'bs_names': ['eye_blink_left', 'mouth_smile'],
        'logger_cfg': None
    }


@pytest.fixture
def linear_exp_blend(linear_exp_blend_params: dict[str, Any]) -> LinearExpBlend:
    """Create a LinearExpBlend instance for testing.

    Creates a LinearExpBlend instance using the provided test parameters
    for testing linear exponential blending functionality.

    Args:
        linear_exp_blend_params (dict[str, Any]): Dictionary containing
            LinearExpBlend initialization parameters.

    Returns:
        LinearExpBlend: Configured LinearExpBlend instance.
    """
    return LinearExpBlend(**linear_exp_blend_params)


def test_init_with_list_bs_names(linear_exp_blend_params: dict[str, Any]):
    """Test LinearExpBlend initialization with list of blendshape names.

    Verifies that LinearExpBlend can be initialized correctly when
    bs_names is provided as a list of strings.

    Args:
        linear_exp_blend_params (dict[str, Any]): Dictionary containing
            LinearExpBlend initialization parameters.
    """
    processor = LinearExpBlend(**linear_exp_blend_params)

    assert processor.name == 'test_processor'
    assert processor.offset == 0.1
    assert processor.normalize_reference == 1.0
    assert processor.exponential_strength == 2.0
    assert processor.blend_weight == 0.5
    assert processor.bs_names == ['eye_blink_left', 'mouth_smile']


def test_init_with_json_file(linear_exp_blend_params: dict[str, Any]):
    """Test LinearExpBlend initialization with JSON file.

    Verifies that LinearExpBlend can be initialized correctly when
    bs_names is provided as a path to a JSON file containing
    blendshape names.

    Args:
        linear_exp_blend_params (dict[str, Any]): Dictionary containing
            LinearExpBlend initialization parameters.
    """
    # Create temporary JSON file
    bs_names = ['eye_blink_left', 'mouth_smile']
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', delete=False
    ) as f:
        json.dump(bs_names, f)
        temp_file = f.name

    try:
        params = linear_exp_blend_params.copy()
        params['bs_names'] = temp_file
        processor = LinearExpBlend(**params)

        assert processor.bs_names == bs_names
    finally:
        os.unlink(temp_file)


def test_init_with_nonexistent_file(linear_exp_blend_params: dict[str, Any]):
    """Test LinearExpBlend initialization with nonexistent file raises exception.

    Verifies that LinearExpBlend raises a FileNotFoundError when
    initialized with a path to a file that does not exist.

    Args:
        linear_exp_blend_params (dict[str, Any]): Dictionary containing
            LinearExpBlend initialization parameters.
    """
    params = linear_exp_blend_params.copy()
    params['bs_names'] = '/nonexistent/file.json'

    with pytest.raises(
        FileNotFoundError,
    ):
        LinearExpBlend(**params)


def test_call_with_matching_blendshapes(
    linear_exp_blend: LinearExpBlend, sample_face_clip: FaceClip
):
    """Test processing FaceClip with matching blendshapes.

    Verifies that LinearExpBlend correctly processes only the specified
    blendshapes while leaving other blendshapes unchanged.

    Args:
        linear_exp_blend (LinearExpBlend): LinearExpBlend instance for testing.
        sample_face_clip (FaceClip): FaceClip instance to process.
    """
    result = linear_exp_blend(sample_face_clip)

    # Verify that a new FaceClip instance is returned
    assert result is not sample_face_clip
    assert isinstance(result, FaceClip)

    # Verify that blendshape_names remain unchanged
    assert result.blendshape_names == sample_face_clip.blendshape_names

    # Verify that only specified blendshapes are processed
    original_values = sample_face_clip.blendshape_values
    result_values = result.blendshape_values

    # eye_blink_left (index 0) and mouth_smile (index 2) should be processed
    # eye_blink_left
    assert not np.array_equal(original_values[:, 0], result_values[:, 0])
    # mouth_smile
    assert not np.array_equal(original_values[:, 2], result_values[:, 2])

    # eye_blink_right (index 1) and brow_up (index 3) should remain unchanged
    assert np.array_equal(original_values[:, 1], result_values[:, 1])  # eye_blink_right
    assert np.array_equal(original_values[:, 3], result_values[:, 3])  # brow_up


def test_call_with_no_matching_blendshapes(
    linear_exp_blend_params: dict[str, Any], sample_face_clip: FaceClip
):
    """Test processing FaceClip with no matching blendshapes.

    Verifies that LinearExpBlend leaves all blendshape values unchanged
    when no blendshapes in the FaceClip match the specified bs_names.

    Args:
        linear_exp_blend_params (dict[str, Any]): Dictionary containing
            LinearExpBlend initialization parameters.
        sample_face_clip (FaceClip): FaceClip instance to process.
    """
    params = linear_exp_blend_params.copy()
    params['bs_names'] = ['nonexistent_blendshape']
    processor = LinearExpBlend(**params)

    result = processor(sample_face_clip)

    # Verify that a new FaceClip instance is returned
    assert result is not sample_face_clip
    assert isinstance(result, FaceClip)

    # Verify that all values remain unchanged
    assert np.array_equal(sample_face_clip.blendshape_values, result.blendshape_values)


def test_call_with_empty_blendshapes_list(
    linear_exp_blend_params: dict[str, Any], sample_face_clip: FaceClip
):
    """Test processing with empty blendshapes list.

    Verifies that LinearExpBlend leaves all blendshape values unchanged
    when bs_names is an empty list.

    Args:
        linear_exp_blend_params (dict[str, Any]): Dictionary containing
            LinearExpBlend initialization parameters.
        sample_face_clip (FaceClip): FaceClip instance to process.
    """
    params = linear_exp_blend_params.copy()
    params['bs_names'] = []
    processor = LinearExpBlend(**params)

    result = processor(sample_face_clip)

    # Verify that a new FaceClip instance is returned
    assert result is not sample_face_clip
    assert isinstance(result, FaceClip)

    # Verify that all values remain unchanged
    assert np.array_equal(sample_face_clip.blendshape_values, result.blendshape_values)


def test_linear_exp_blend_batch_basic():
    """Test basic functionality of _linear_exp_blend_batch function.

    Verifies that the _linear_exp_blend_batch function correctly applies
    linear exponential blending to input arrays and produces valid output.
    """
    x = np.array([[0.1, 0.5], [0.8, 0.2]])
    offset = 0.1
    normalize_reference = 1.0
    exponential_strength = 2.0
    blend_weight = 0.5

    result = _linear_exp_blend_batch(
        x, offset, normalize_reference, exponential_strength, blend_weight
    )

    # Verify that output shape is the same as input
    assert result.shape == x.shape

    # Verify that output values are in [0, 1] range
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)

    # Verify that output differs from input (indicating transformation occurred)
    assert not np.array_equal(x, result)


def test_linear_exp_blend_batch_with_zero_offset():
    """Test _linear_exp_blend_batch with zero offset.

    Verifies that the function works correctly when offset parameter
    is set to zero.
    """
    x = np.array([[0.1, 0.5], [0.8, 0.2]])
    offset = 0.0
    normalize_reference = 1.0
    exponential_strength = 2.0
    blend_weight = 0.5

    result = _linear_exp_blend_batch(
        x, offset, normalize_reference, exponential_strength, blend_weight
    )

    assert result.shape == x.shape
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_linear_exp_blend_batch_with_zero_blend_weight():
    """Test _linear_exp_blend_batch with zero blend_weight (pure linear).

    Verifies that when blend_weight is zero, the function produces
    only the linear component of the blending operation.
    """
    x = np.array([[0.1, 0.5], [0.8, 0.2]])
    offset = 0.1
    normalize_reference = 1.0
    exponential_strength = 2.0
    blend_weight = 0.0

    result = _linear_exp_blend_batch(
        x, offset, normalize_reference, exponential_strength, blend_weight
    )

    # When blend_weight is 0, should only have linear component
    expected = np.clip((x + offset) / normalize_reference, 0.0, 1.0)
    np.testing.assert_array_almost_equal(result, expected)


def test_linear_exp_blend_batch_with_one_blend_weight():
    """Test _linear_exp_blend_batch with blend_weight of 1 (pure exponential).

    Verifies that when blend_weight is one, the function produces
    only the exponential component of the blending operation.
    """
    x = np.array([[0.1, 0.5], [0.8, 0.2]])
    offset = 0.1
    normalize_reference = 1.0
    exponential_strength = 2.0
    blend_weight = 1.0

    result = _linear_exp_blend_batch(
        x, offset, normalize_reference, exponential_strength, blend_weight
    )

    # When blend_weight is 1, should only have exponential component
    shifted_x = np.clip(x + offset, 0, 1.0)
    normalized_x = shifted_x / normalize_reference
    expected = np.clip(1 - np.exp(-exponential_strength * normalized_x), 0.0, 1.0)
    np.testing.assert_array_almost_equal(result, expected)


def test_linear_exp_blend_batch_edge_cases():
    """Test _linear_exp_blend_batch with edge case values.

    Verifies that the function handles edge cases correctly, including
    values at the boundaries of the valid range.
    """
    x = np.array([[0.0, 1.0], [-0.5, 1.5]])
    offset = 0.1
    normalize_reference = 1.0
    exponential_strength = 2.0
    blend_weight = 0.5

    result = _linear_exp_blend_batch(
        x, offset, normalize_reference, exponential_strength, blend_weight
    )

    # Verify that output shape is the same as input
    assert result.shape == x.shape

    # Verify that output values are in [0, 1] range
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_linear_exp_blend_batch_with_different_normalize_reference():
    """Test _linear_exp_blend_batch with different normalize_reference values.

    Verifies that the function works correctly with different
    normalization reference values.
    """
    x = np.array([[0.1, 0.5], [0.8, 0.2]])
    offset = 0.1
    normalize_reference = 0.5  # Different normalization reference value
    exponential_strength = 2.0
    blend_weight = 0.5

    result = _linear_exp_blend_batch(
        x, offset, normalize_reference, exponential_strength, blend_weight
    )

    assert result.shape == x.shape
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_linear_exp_blend_batch_with_different_exponential_strength():
    """Test _linear_exp_blend_batch with different exponential_strength values.

    Verifies that the function works correctly with different
    exponential strength parameters.
    """
    x = np.array([[0.1, 0.5], [0.8, 0.2]])
    offset = 0.1
    normalize_reference = 1.0
    exponential_strength = 5.0  # Higher exponential strength
    blend_weight = 0.5

    result = _linear_exp_blend_batch(
        x, offset, normalize_reference, exponential_strength, blend_weight
    )

    assert result.shape == x.shape
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)
