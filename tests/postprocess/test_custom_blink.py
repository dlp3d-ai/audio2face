import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from audio2face.data_structures.face_clip import FaceClip
from audio2face.postprocess.custom_blink import CustomBlink


@pytest.fixture
def temp_blink_json_files():
    """Create temporary blink animation JSON files.

    Creates default and custom blink animation data and saves them to
    temporary JSON files for testing purposes.

    Returns:
        dict[str, str]: Dictionary containing file paths for 'default' and
            'custom' blink animations.
    """
    temp_files = {}

    # Create default blink animation
    default_blink_data = {
        'blendshape_names': ['eyeBlinkLeft', 'eyeBlinkRight'],
        'blendshape_values': [
            [0.0, 0.0],  # Frame 1
            [0.5, 0.5],  # Frame 2
            [1.0, 1.0],  # Frame 3
            [0.5, 0.5],  # Frame 4
            [0.0, 0.0]   # Frame 5
        ]
    }

    # Create custom blink animation
    custom_blink_data = {
        'blendshape_names': ['eyeBlinkLeft', 'eyeBlinkRight', 'eyeWideLeft'],
        'blendshape_values': [
            [0.0, 0.0, 0.0],  # Frame 1
            [0.3, 0.3, 0.2],  # Frame 2
            [0.8, 0.8, 0.5],  # Frame 3
            [0.3, 0.3, 0.2],  # Frame 4
            [0.0, 0.0, 0.0]   # Frame 5
        ]
    }

    # Write to temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(default_blink_data, f)
        temp_files['default'] = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(custom_blink_data, f)
        temp_files['custom'] = f.name

    yield temp_files

    # Clean up temporary files
    for file_path in temp_files.values():
        Path(file_path).unlink(missing_ok=True)


@pytest.fixture
def custom_blink(temp_blink_json_files: dict[str, str]) -> CustomBlink:
    """Create a CustomBlink instance for testing.

    Args:
        temp_blink_json_files (dict[str, str]): Dictionary containing paths
            to temporary blink JSON files.

    Returns:
        CustomBlink: Configured CustomBlink instance with test data.
    """
    return CustomBlink(
        default_blink_json_path=temp_blink_json_files['default'],
        blink_json_paths={'custom': temp_blink_json_files['custom']},
        blink_interval_lowerbound=60,
        blink_interval_upperbound=120,
        logger_cfg=None
    )


@pytest.fixture
def test_face_clip() -> FaceClip:
    """Create a test FaceClip instance.

    Creates a FaceClip with common blendshape names and zero values
    for testing blink animation application.

    Returns:
        FaceClip: Test FaceClip instance with 300 frames and 4 blendshapes.
    """
    blendshape_names = ['eyeBlinkLeft', 'eyeBlinkRight', 'jawOpen', 'mouthSmile']
    n_frames = 300
    blendshape_values = np.zeros((n_frames, len(blendshape_names)), dtype=np.float16)

    return FaceClip(
        blendshape_names=blendshape_names,
        blendshape_values=blendshape_values,
        logger_cfg=None
    )


class TestCustomBlink:
    """Unit tests for the CustomBlink class.

    This test class covers initialization, blink animation application,
    error handling, and edge cases for the CustomBlink functionality.
    """

    def test_init_with_valid_paths(self, temp_blink_json_files: dict[str, str]) -> None:
        """Test CustomBlink initialization with valid file paths.

        Verifies that CustomBlink can be initialized correctly with
        valid JSON file paths and that the blink animations are loaded
        properly.

        Args:
            temp_blink_json_files (dict[str, str]): Dictionary containing
                paths to temporary blink JSON files.
        """
        custom_blink = CustomBlink(
            default_blink_json_path=temp_blink_json_files['default'],
            blink_json_paths={'custom': temp_blink_json_files['custom']},
            blink_interval_lowerbound=60,
            blink_interval_upperbound=120,
            logger_cfg=None
        )

        assert custom_blink.blink_interval_lowerbound == 60
        assert custom_blink.blink_interval_upperbound == 120
        assert 'default' in custom_blink.blink_animation
        assert 'custom' in custom_blink.blink_animation
        assert custom_blink.blink_animation['default']['n_frames'] == 5
        assert custom_blink.blink_animation['custom']['n_frames'] == 5

    def test_init_with_duplicate_default_key(
            self, temp_blink_json_files: dict[str, str]) -> None:
        """Test initialization when 'default' key already exists in blink_json_paths.

        Verifies that the default blink animation is correctly set even when
        a 'default' key already exists in the blink_json_paths dictionary.

        Args:
            temp_blink_json_files (dict[str, str]): Dictionary containing
                paths to temporary blink JSON files.
        """
        blink_json_paths = {
            'default': temp_blink_json_files['custom'],
            'custom': temp_blink_json_files['custom']
        }

        custom_blink = CustomBlink(
            default_blink_json_path=temp_blink_json_files['default'],
            blink_json_paths=blink_json_paths,
            logger_cfg=None
        )

        # Verify that the default key is set correctly
        assert 'default' in custom_blink.blink_animation
        assert custom_blink.blink_animation['default']['n_frames'] == 5

    def test_init_with_invalid_json_file(self) -> None:
        """Test initialization with invalid JSON file raises exception.

        Verifies that CustomBlink raises a JSONDecodeError when
        initialized with a malformed JSON file.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": "json"')
            invalid_json_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                CustomBlink(
                    default_blink_json_path=invalid_json_path,
                    blink_json_paths={},
                    logger_cfg=None
                )
        finally:
            Path(invalid_json_path).unlink(missing_ok=True)

    def test_init_with_missing_file(self) -> None:
        """Test initialization with non-existent file path raises exception.

        Verifies that CustomBlink raises a FileNotFoundError when
        initialized with a file path that does not exist.
        """
        with pytest.raises(FileNotFoundError):
            CustomBlink(
                default_blink_json_path='nonexistent_file.json',
                blink_json_paths={},
                logger_cfg=None
            )

    def test_call_with_existing_blink_name(
            self, custom_blink: CustomBlink, test_face_clip: FaceClip) -> None:
        """Test __call__ method with existing blink name.

        Verifies that the __call__ method correctly applies a blink animation
        when given a valid blink name that exists in the blink animations.

        Args:
            custom_blink (CustomBlink): CustomBlink instance for testing.
            test_face_clip (FaceClip): FaceClip instance to apply blink
                animation to.
        """
        result = custom_blink(
            face_clip=test_face_clip,
            blink_name='custom'
        )

        assert isinstance(result, FaceClip)
        assert result.blendshape_names == test_face_clip.blendshape_names
        assert result.blendshape_values.shape == test_face_clip.blendshape_values.shape
        # Verify that blink animation is applied (at least some non-zero values)
        assert not np.array_equal(
            result.blendshape_values, test_face_clip.blendshape_values)

    def test_call_with_nonexistent_blink_name(
            self, custom_blink: CustomBlink, test_face_clip: FaceClip) -> None:
        """Test __call__ method with non-existent blink name.

        Verifies that the __call__ method falls back to the default blink
        animation when given a blink name that does not exist.

        Args:
            custom_blink (CustomBlink): CustomBlink instance for testing.
            test_face_clip (FaceClip): FaceClip instance to apply blink
                animation to.
        """
        result = custom_blink(
            face_clip=test_face_clip,
            blink_name='nonexistent'
        )

        assert isinstance(result, FaceClip)
        assert result.blendshape_names == test_face_clip.blendshape_names
        assert result.blendshape_values.shape == test_face_clip.blendshape_values.shape
        # Should use default blink animation
        assert not np.array_equal(
            result.blendshape_values, test_face_clip.blendshape_values)

    def test_call_with_no_matching_blendshapes(
            self, temp_blink_json_files: dict[str, str]) -> None:
        """Test __call__ method when FaceClip has no matching blendshapes.

        Verifies that the __call__ method handles cases where the FaceClip
        does not contain any blendshapes that match the blink animation.

        Args:
            temp_blink_json_files (dict[str, str]): Dictionary containing
                paths to temporary blink JSON files.
        """
        # Create non-matching blendshape names
        blendshape_names = ['jawOpen', 'mouthSmile', 'browUp']
        n_frames = 300
        blendshape_values = np.zeros(
            (n_frames, len(blendshape_names)), dtype=np.float16)
        face_clip = FaceClip(
            blendshape_names=blendshape_names,
            blendshape_values=blendshape_values,
            logger_cfg=None
        )

        custom_blink = CustomBlink(
            default_blink_json_path=temp_blink_json_files['default'],
            blink_json_paths={},
            logger_cfg=None
        )

        result = custom_blink(
            face_clip=face_clip,
            blink_name='default'
        )

        assert isinstance(result, FaceClip)
        # Since there are no matching blendshapes, result should be same as original
        assert np.array_equal(result.blendshape_values, face_clip.blendshape_values)

    def test_call_with_short_face_clip(self, custom_blink: CustomBlink) -> None:
        """Test __call__ method with very short FaceClip.

        Verifies that the __call__ method handles FaceClips that are
        shorter than the minimum blink interval.

        Args:
            custom_blink (CustomBlink): CustomBlink instance for testing.
        """
        # Create very short FaceClip (less than blink interval)
        blendshape_names = ['eyeBlinkLeft', 'eyeBlinkRight']
        n_frames = 30  # Less than blink_interval_lowerbound
        blendshape_values = np.zeros(
            (n_frames, len(blendshape_names)), dtype=np.float16)
        short_face_clip = FaceClip(
            blendshape_names=blendshape_names,
            blendshape_values=blendshape_values,
            logger_cfg=None
        )

        result = custom_blink(
            face_clip=short_face_clip,
            blink_name='default'
        )

        assert isinstance(result, FaceClip)
        # Since it is too short, no blink animation may be applied
        assert result.blendshape_values.shape == short_face_clip.blendshape_values.shape

    def test_call_with_exact_blink_interval(self, custom_blink: CustomBlink) -> None:
        """Test __call__ method with FaceClip of exact blink interval length.

        Verifies that the __call__ method works correctly when the FaceClip
        length exactly matches the upper bound of the blink interval.

        Args:
            custom_blink (CustomBlink): CustomBlink instance for testing.
        """
        blendshape_names = ['eyeBlinkLeft', 'eyeBlinkRight']
        n_frames = 120  # Equal to blink_interval_upperbound
        blendshape_values = np.zeros(
            (n_frames, len(blendshape_names)), dtype=np.float16)
        face_clip = FaceClip(
            blendshape_names=blendshape_names,
            blendshape_values=blendshape_values,
            logger_cfg=None
        )

        result = custom_blink(
            face_clip=face_clip,
            blink_name='default'
        )

        assert isinstance(result, FaceClip)
        assert result.blendshape_values.shape == face_clip.blendshape_values.shape

    def test_call_preserves_original_face_clip(
            self, custom_blink: CustomBlink, test_face_clip: FaceClip) -> None:
        """Test that __call__ method does not modify the original FaceClip.

        Verifies that the __call__ method creates a new FaceClip instance
        and does not modify the input FaceClip.

        Args:
            custom_blink (CustomBlink): CustomBlink instance for testing.
            test_face_clip (FaceClip): FaceClip instance to apply blink
                animation to.
        """
        original_values = test_face_clip.blendshape_values.copy()

        result = custom_blink(
            face_clip=test_face_clip,
            blink_name='default'
        )

        # Verify that original FaceClip is not modified
        assert np.array_equal(test_face_clip.blendshape_values, original_values)
        # Verify that a new FaceClip is returned
        assert result is not test_face_clip

    def test_call_with_different_blink_intervals(
            self, temp_blink_json_files: dict[str, str]) -> None:
        """Test __call__ method with different blink interval parameters.

        Verifies that the __call__ method works correctly with different
        blink interval configurations.

        Args:
            temp_blink_json_files (dict[str, str]): Dictionary containing
                paths to temporary blink JSON files.
        """
        custom_blink = CustomBlink(
            default_blink_json_path=temp_blink_json_files['default'],
            blink_json_paths={},
            blink_interval_lowerbound=30,
            blink_interval_upperbound=60,
            logger_cfg=None
        )

        blendshape_names = ['eyeBlinkLeft', 'eyeBlinkRight']
        n_frames = 200
        blendshape_values = np.zeros(
            (n_frames, len(blendshape_names)), dtype=np.float16)
        face_clip = FaceClip(
            blendshape_names=blendshape_names,
            blendshape_values=blendshape_values,
            logger_cfg=None
        )

        result = custom_blink(
            face_clip=face_clip,
            blink_name='default'
        )

        assert isinstance(result, FaceClip)
        assert result.blendshape_values.shape == face_clip.blendshape_values.shape

    def test_call_with_custom_blink_animation(
            self,
            custom_blink: CustomBlink,
            test_face_clip: FaceClip) -> None:
        """Test __call__ method with custom blink animation.

        Verifies that the __call__ method correctly applies a custom
        blink animation when specified.

        Args:
            custom_blink (CustomBlink): CustomBlink instance for testing.
            test_face_clip (FaceClip): FaceClip instance to apply blink
                animation to.
        """
        result = custom_blink(
            face_clip=test_face_clip,
            blink_name='custom'
        )

        assert isinstance(result, FaceClip)
        # Verify that custom blink animation is applied
        assert not np.array_equal(
            result.blendshape_values,
            test_face_clip.blendshape_values)

    def test_call_with_additional_kwargs(
            self,
            custom_blink: CustomBlink,
            test_face_clip: FaceClip) -> None:
        """Test __call__ method with additional keyword arguments.

        Verifies that the __call__ method can handle additional keyword
        arguments without errors.

        Args:
            custom_blink (CustomBlink): CustomBlink instance for testing.
            test_face_clip (FaceClip): FaceClip instance to apply blink
                animation to.
        """
        result = custom_blink(
            face_clip=test_face_clip,
            blink_name='default',
            extra_param='test_value'
        )

        assert isinstance(result, FaceClip)
        assert result.blendshape_values.shape == test_face_clip.blendshape_values.shape

    def test_blink_animation_structure(self, custom_blink: CustomBlink) -> None:
        """Test the correctness of blink animation data structure.

        Verifies that the blink animation data is properly structured
        with correct frame counts and blendshape values.

        Args:
            custom_blink (CustomBlink): CustomBlink instance for testing.
        """
        # Verify default blink animation
        default_animation = custom_blink.blink_animation['default']
        assert 'n_frames' in default_animation
        assert default_animation['n_frames'] == 5
        assert 'eyeBlinkLeft' in default_animation
        assert 'eyeBlinkRight' in default_animation
        assert len(default_animation['eyeBlinkLeft']) == 5
        assert len(default_animation['eyeBlinkRight']) == 5

        # Verify custom blink animation
        custom_animation = custom_blink.blink_animation['custom']
        assert 'n_frames' in custom_animation
        assert custom_animation['n_frames'] == 5
        assert 'eyeBlinkLeft' in custom_animation
        assert 'eyeBlinkRight' in custom_animation
        assert 'eyeWideLeft' in custom_animation
        assert len(custom_animation['eyeBlinkLeft']) == 5
        assert len(custom_animation['eyeBlinkRight']) == 5
        assert len(custom_animation['eyeWideLeft']) == 5
