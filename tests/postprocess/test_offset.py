import json
import os
import tempfile

import numpy as np
import pytest

from audio2face.data_structures.face_clip import FaceClip
from audio2face.postprocess.offset import Offset


@pytest.fixture
def sample_blendshape_names() -> list[str]:
    """Fixture for creating sample blendshape names."""
    return ['あ', 'い', 'う', 'え', 'お', 'にやり', 'まばたき', '怒り目']


@pytest.fixture
def sample_blendshape_values() -> np.ndarray:
    """Fixture for creating sample blendshape values."""
    return np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ], dtype=np.float32)


@pytest.fixture
def face_clip(sample_blendshape_names: list[str],
              sample_blendshape_values: np.ndarray) -> FaceClip:
    """Fixture for creating a FaceClip instance."""
    return FaceClip(
        blendshape_names=sample_blendshape_names,
        blendshape_values=sample_blendshape_values,
        dtype=np.float32
    )


@pytest.fixture
def anger_offset_dict() -> dict[str, float]:
    """Fixture for creating anger offset dictionary."""
    return {
        'あ': 0.0,
        'い': 0.08,
        'う': 0.0,
        'え': 0.0,
        'お': 0.0,
        'にやり': 0.0,
        'まばたき': 0.0,
        '怒り目': 0.1875
    }


@pytest.fixture
def happiness_offset_dict() -> dict[str, float]:
    """Fixture for creating happiness offset dictionary."""
    return {
        'あ': 0.1,
        'い': 0.0,
        'う': 0.0,
        'え': 0.0,
        'お': 0.0,
        'にやり': 0.3,
        'まばたき': 0.0,
        '怒り目': 0.0
    }


@pytest.fixture
def anger_json_file(anger_offset_dict: dict[str, float]) -> str:
    """Fixture for creating a temporary JSON file with anger offset."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(anger_offset_dict, temp_file)
    temp_file.close()

    # Return the file path
    yield temp_file.name

    # Clean up the temporary file after the test
    try:
        os.unlink(temp_file.name)
    except OSError:
        pass  # File might already be deleted


@pytest.fixture
def happiness_json_file(happiness_offset_dict: dict[str, float]) -> str:
    """Fixture for creating a temporary JSON file with happiness offset."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(happiness_offset_dict, temp_file)
    temp_file.close()

    # Return the file path
    yield temp_file.name

    # Clean up the temporary file after the test
    try:
        os.unlink(temp_file.name)
    except OSError:
        pass  # File might already be deleted


@pytest.fixture
def offset_instance(anger_json_file: str, happiness_json_file: str) -> Offset:
    """Fixture for creating an Offset instance."""
    offset_json_paths = {
        'anger': anger_json_file,
        'happiness': happiness_json_file
    }
    return Offset(
        offset_json_paths=offset_json_paths,
        logger_cfg={'logger_name': 'test_logger'}
    )


class TestOffset:
    """Test class for Offset."""

    def test_init_success(self, anger_json_file: str, happiness_json_file: str):
        """Test successful initialization."""
        offset_json_paths = {
            'anger': anger_json_file,
            'happiness': happiness_json_file
        }

        offset = Offset(
            offset_json_paths=offset_json_paths,
            logger_cfg={'logger_name': 'test_logger'}
        )

        assert offset.offset_json_paths == offset_json_paths
        assert 'anger' in offset.offset_dicts
        assert 'happiness' in offset.offset_dicts
        assert len(offset.offset_dicts) == 2

    def test_init_with_empty_dict(self):
        """Test initialization with empty offset_json_paths."""
        offset = Offset(
            offset_json_paths={},
            logger_cfg={'logger_name': 'test_logger'}
        )

        assert offset.offset_json_paths == {}
        assert offset.offset_dicts == {}

    def test_call_with_existing_offset(
            self, offset_instance: Offset, face_clip: FaceClip):
        """Test __call__ method with existing offset name."""
        result = offset_instance(face_clip, 'anger')

        # Should return a clone
        assert result is not face_clip
        assert result.blendshape_names == face_clip.blendshape_names
        assert result.dtype == face_clip.dtype

        # Check that values are correctly offset and clipped
        # あ: 0.1 + 0.0 = 0.1
        assert np.isclose(result.blendshape_values[0, 0], 0.1)
        # い: 0.2 + 0.08 = 0.28
        assert np.isclose(result.blendshape_values[0, 1], 0.28)
        # う: 0.3 + 0.0 = 0.3
        assert np.isclose(result.blendshape_values[0, 2], 0.3)
        # 怒り目: 0.8 + 0.1875 = 0.9875
        assert np.isclose(result.blendshape_values[0, 7], 0.9875)

    def test_call_with_nonexistent_offset(
            self, offset_instance: Offset, face_clip: FaceClip):
        """Test __call__ method with non-existent offset name."""
        result = offset_instance(face_clip, 'nonexistent_offset')

        # Should return a clone without any modifications
        assert result is not face_clip
        assert np.array_equal(result.blendshape_values, face_clip.blendshape_values)

    def test_call_with_partial_matching_blendshapes(
            self, offset_instance: Offset, face_clip: FaceClip):
        """Test __call__ method with partial matching blendshapes."""
        result = offset_instance(face_clip, 'happiness')

        # Should return a clone
        assert result is not face_clip

        # Check that only matching blendshapes are offset
        # あ: 0.1 + 0.1 = 0.2
        assert np.isclose(result.blendshape_values[0, 0], 0.2)
        # い: 0.2 + 0.0 = 0.2 (no offset)
        assert np.isclose(result.blendshape_values[0, 1], 0.2)
        # にやり: 0.6 + 0.3 = 0.9
        assert np.isclose(result.blendshape_values[0, 5], 0.9)

    def test_call_with_negative_values(self, face_clip: FaceClip):
        """Test __call__ method with negative offset values."""
        # Create offset with negative values
        negative_offset_dict = {
            'あ': -0.05,  # 0.1 - 0.05 = 0.05
            'い': -0.3,   # 0.2 - 0.3 = -0.1 (should be clipped to 0.0)
            'う': 0.0     # 0.3 + 0.0 = 0.3
        }

        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(negative_offset_dict, temp_file)
        temp_file.close()

        try:
            offset = Offset(
                offset_json_paths={'test': temp_file.name},
                logger_cfg={'logger_name': 'test_logger'}
            )

            result = offset(face_clip, 'test')

            # Check negative clipping behavior
            # あ: 0.1 - 0.05
            assert np.isclose(result.blendshape_values[0, 0], 0.05)
            # い: 0.2 - 0.3 = -0.1
            assert np.isclose(result.blendshape_values[0, 1], -0.1)
            # う: 0.3 + 0.0 = 0.3
            assert np.isclose(result.blendshape_values[0, 2], 0.3)

        finally:
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass

    def test_call_preserves_original(
            self, offset_instance: Offset, face_clip: FaceClip):
        """Test that __call__ method preserves the original face_clip."""
        original_values = face_clip.blendshape_values.copy()
        result = offset_instance(face_clip, 'anger')

        # Original should remain unchanged
        assert np.array_equal(face_clip.blendshape_values, original_values)

        # Result should be different (offset applied)
        assert not np.array_equal(result.blendshape_values, original_values)

    def test_call_with_single_frame(self, offset_instance: Offset):
        """Test __call__ method with single frame face clip."""
        blendshape_names = ['あ', 'い', 'う']
        blendshape_values = np.array([[0.3, 0.5, 0.7]], dtype=np.float32)

        face_clip = FaceClip(
            blendshape_names=blendshape_names,
            blendshape_values=blendshape_values,
            dtype=np.float32
        )

        result = offset_instance(face_clip, 'anger')

        # Check single frame offset
        # あ: 0.3 + 0.0 = 0.3
        assert np.isclose(result.blendshape_values[0, 0], 0.3)
        # い: 0.5 + 0.08 = 0.58
        assert np.isclose(result.blendshape_values[0, 1], 0.58)
        # う: 0.7 + 0.0 = 0.7
        assert np.isclose(result.blendshape_values[0, 2], 0.7)

    def test_call_with_zero_frames(self, offset_instance: Offset):
        """Test __call__ method with zero frames face clip."""
        blendshape_names = ['あ', 'い', 'う']
        blendshape_values = np.zeros((0, 3), dtype=np.float32)

        face_clip = FaceClip(
            blendshape_names=blendshape_names,
            blendshape_values=blendshape_values,
            dtype=np.float32
        )

        result = offset_instance(face_clip, 'anger')

        # Should handle empty array gracefully
        assert result.blendshape_values.shape == (0, 3)
        assert result.blendshape_names == blendshape_names

    def test_call_with_different_dtypes(self, offset_instance: Offset):
        """Test __call__ method with different dtypes."""
        blendshape_names = ['あ', 'い', 'う']
        blendshape_values = np.array([[0.3, 0.5, 0.7]], dtype=np.float32)

        face_clip = FaceClip(
            blendshape_names=blendshape_names,
            blendshape_values=blendshape_values,
            dtype=np.float32
        )

        result = offset_instance(face_clip, 'anger')

        # Should preserve dtype
        assert result.dtype == np.float32
        assert result.blendshape_values.dtype == np.float32

    def test_call_with_timeline_start_idx(
            self, offset_instance: Offset, face_clip: FaceClip):
        """Test __call__ method preserves timeline_start_idx."""
        # Set timeline_start_idx
        face_clip.set_timeline_start_idx(100)

        result = offset_instance(face_clip, 'anger')

        # Should preserve timeline_start_idx
        assert result.timeline_start_idx == face_clip.timeline_start_idx

    def test_call_with_none_timeline_start_idx(self, offset_instance: Offset):
        """Test __call__ method with None timeline_start_idx."""
        blendshape_names = ['あ', 'い', 'う']
        blendshape_values = np.array([[0.3, 0.5, 0.7]], dtype=np.float32)

        face_clip = FaceClip(
            blendshape_names=blendshape_names,
            blendshape_values=blendshape_values,
            dtype=np.float32,
            timeline_start_idx=None
        )

        result = offset_instance(face_clip, 'anger')

        # Should preserve None timeline_start_idx
        assert result.timeline_start_idx is None

    def test_call_with_empty_offset_dict(self, face_clip: FaceClip):
        """Test __call__ method with empty offset dictionary."""
        empty_offset_dict = {}

        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(empty_offset_dict, temp_file)
        temp_file.close()

        try:
            offset = Offset(
                offset_json_paths={'empty': temp_file.name},
                logger_cfg={'logger_name': 'test_logger'}
            )

            result = offset(face_clip, 'empty')

            # Should return a clone without any modifications
            assert result is not face_clip
            assert np.array_equal(result.blendshape_values, face_clip.blendshape_values)

        finally:
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass

    def test_call_with_no_matching_blendshapes(self, face_clip: FaceClip):
        """Test __call__ method with no matching blendshapes."""
        non_matching_offset_dict = {
            'nonexistent_blendshape': 0.5
        }

        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(non_matching_offset_dict, temp_file)
        temp_file.close()

        try:
            offset = Offset(
                offset_json_paths={'test': temp_file.name},
                logger_cfg={'logger_name': 'test_logger'}
            )

            result = offset(face_clip, 'test')

            # Should return a clone without any modifications
            assert result is not face_clip
            assert np.array_equal(result.blendshape_values, face_clip.blendshape_values)

        finally:
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass

    def test_multiple_offset_applications(
            self,
            offset_instance: Offset,
            face_clip: FaceClip):
        """Test applying multiple offsets sequentially."""
        # Apply anger offset first
        result1 = offset_instance(face_clip, 'anger')

        # Apply happiness offset to the result
        result2 = offset_instance(result1, 'happiness')

        # Check that both offsets are applied correctly
        # あ: 0.1 + 0.0 (anger) + 0.1 (happiness) = 0.2
        assert np.isclose(result2.blendshape_values[0, 0], 0.2)
        # い: 0.2 + 0.08 (anger) + 0.0 (happiness) = 0.28
        assert np.isclose(result2.blendshape_values[0, 1], 0.28)
        # にやり: 0.6 + 0.0 (anger) + 0.3 (happiness) = 0.9
        assert np.isclose(result2.blendshape_values[0, 5], 0.9)

    def test_offset_dicts_loading(self, anger_json_file: str, happiness_json_file: str):
        """Test that offset dictionaries are loaded correctly."""
        offset_json_paths = {
            'anger': anger_json_file,
            'happiness': happiness_json_file
        }

        offset = Offset(
            offset_json_paths=offset_json_paths,
            logger_cfg={'logger_name': 'test_logger'}
        )

        # Check that dictionaries are loaded correctly
        assert 'anger' in offset.offset_dicts
        assert 'happiness' in offset.offset_dicts

        # Check specific values
        assert offset.offset_dicts['anger']['い'] == 0.08
        assert offset.offset_dicts['anger']['怒り目'] == 0.1875
        assert offset.offset_dicts['happiness']['あ'] == 0.1
        assert offset.offset_dicts['happiness']['にやり'] == 0.3
