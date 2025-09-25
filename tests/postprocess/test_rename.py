import json
import os
import tempfile

import numpy as np
import pytest

from audio2face.data_structures.face_clip import FaceClip
from audio2face.postprocess.rename import Rename


@pytest.fixture
def sample_blendshape_names() -> list[str]:
    """Fixture for creating sample blendshape names."""
    return ['eye_blink_L', 'eye_blink_R', 'mouth_smile', 'brow_up']


@pytest.fixture
def sample_blendshape_values() -> np.ndarray:
    """Fixture for creating sample blendshape values."""
    return np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2]
    ], dtype=np.float16)


@pytest.fixture
def face_clip(sample_blendshape_names: list[str],
              sample_blendshape_values: np.ndarray) -> FaceClip:
    """Fixture for creating a FaceClip instance."""
    return FaceClip(
        blendshape_names=sample_blendshape_names,
        blendshape_values=sample_blendshape_values,
        dtype=np.float16,
        timeline_start_idx=10
    )


@pytest.fixture
def bs_names_mapping_dict() -> dict[str, str]:
    """Fixture for creating blendshape names mapping dictionary.
    Format: {dst_name: src_name}
    """
    return {
        'EyeBlinkLeft': 'eye_blink_L',
        'EyeBlinkRight': 'eye_blink_R',
        'MouthSmile': 'mouth_smile',
        'BrowUp': 'brow_up'
    }


@pytest.fixture
def bs_names_mapping_json_file(bs_names_mapping_dict: dict[str, str]) -> str:
    """Fixture for creating a temporary JSON file with blendshape names mapping."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(bs_names_mapping_dict, temp_file)
    temp_file.close()

    # Return the file path
    yield temp_file.name

    # Clean up the temporary file after the test
    try:
        os.unlink(temp_file.name)
    except OSError:
        pass  # File might already be deleted


@pytest.fixture
def rename_with_dict(bs_names_mapping_dict: dict[str, str]) -> Rename:
    """Fixture for creating a Rename instance with dictionary input."""
    return Rename(
        name='test_rename_dict',
        bs_names_mapping=bs_names_mapping_dict,
        logger_cfg={'logger_name': 'test_logger'}
    )


@pytest.fixture
def rename_with_file(bs_names_mapping_json_file: str) -> Rename:
    """Fixture for creating a Rename instance with file input."""
    return Rename(
        name='test_rename_file',
        bs_names_mapping=bs_names_mapping_json_file,
        logger_cfg={'logger_name': 'test_logger'}
    )


class TestRename:
    """Test class for Rename."""

    def test_init_with_dict(self, bs_names_mapping_dict: dict[str, str]):
        """Test successful initialization with dictionary input."""
        rename = Rename(
            name='test_rename',
            bs_names_mapping=bs_names_mapping_dict,
            logger_cfg={'logger_name': 'test_logger'}
        )

        assert rename.name == 'test_rename'
        assert rename.bs_names_mapping == bs_names_mapping_dict
        assert rename.logger_cfg['logger_name'] == 'test_rename'
        assert len(rename.name_idx_mapping) == len(bs_names_mapping_dict)

    def test_init_with_file(self, bs_names_mapping_json_file: str,
                           bs_names_mapping_dict: dict[str, str]):
        """Test successful initialization with file input."""
        rename = Rename(
            name='test_rename_file',
            bs_names_mapping=bs_names_mapping_json_file,
            logger_cfg={'logger_name': 'test_logger'}
        )

        assert rename.name == 'test_rename_file'
        assert rename.bs_names_mapping == bs_names_mapping_dict
        assert rename.logger_cfg['logger_name'] == 'test_rename_file'
        assert len(rename.name_idx_mapping) == len(bs_names_mapping_dict)

    def test_init_with_nonexistent_file(self):
        """Test initialization with non-existent file raises FileNotFoundError."""
        with pytest.raises(
                FileNotFoundError):
            Rename(
                name='test_rename',
                bs_names_mapping='nonexistent_file.json',
                logger_cfg={'logger_name': 'test_logger'}
            )

    def test_init_with_empty_mapping(self):
        """Test initialization with empty mapping raises ValueError."""
        with pytest.raises(ValueError):
            Rename(
                name='test_rename',
                bs_names_mapping={},
                logger_cfg={'logger_name': 'test_logger'}
            )

    def test_init_with_empty_file_mapping(self):
        """Test initialization with empty file mapping raises ValueError."""
        # Create temporary file with empty mapping
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump({}, temp_file)
        temp_file.close()

        try:
            with pytest.raises(ValueError):
                Rename(
                    name='test_rename',
                    bs_names_mapping=temp_file.name,
                    logger_cfg={'logger_name': 'test_logger'}
                )
        finally:
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass

    def test_call_with_dict_input(self, rename_with_dict: Rename, face_clip: FaceClip):
        """Test __call__ method with dictionary input."""
        result = rename_with_dict(face_clip)

        # Should return a new FaceClip instance
        assert result is not face_clip
        expected_names = ['EyeBlinkLeft', 'EyeBlinkRight', 'MouthSmile', 'BrowUp']
        assert result.blendshape_names == expected_names
        assert result.dtype == face_clip.dtype
        assert result.timeline_start_idx == face_clip.timeline_start_idx
        assert result.logger_cfg == face_clip.logger_cfg

        # Check that values are correctly mapped
        # EyeBlinkLeft (index 0) <- eye_blink_L (index 0)
        assert np.array_equal(
            result.blendshape_values[:, 0],
            face_clip.blendshape_values[:, 0])
        # EyeBlinkRight (index 1) <- eye_blink_R (index 1)
        assert np.array_equal(
            result.blendshape_values[:, 1],
            face_clip.blendshape_values[:, 1])
        # MouthSmile (index 2) <- mouth_smile (index 2)
        assert np.array_equal(
            result.blendshape_values[:, 2],
            face_clip.blendshape_values[:, 2])
        # BrowUp (index 3) <- brow_up (index 3)
        assert np.array_equal(
            result.blendshape_values[:, 3],
            face_clip.blendshape_values[:, 3])

    def test_call_with_file_input(self, rename_with_file: Rename, face_clip: FaceClip):
        """Test __call__ method with file input."""
        result = rename_with_file(face_clip)

        # Should return a new FaceClip instance
        assert result is not face_clip
        expected_names = ['EyeBlinkLeft', 'EyeBlinkRight', 'MouthSmile', 'BrowUp']
        assert result.blendshape_names == expected_names
        assert result.dtype == face_clip.dtype
        assert result.timeline_start_idx == face_clip.timeline_start_idx

        # Check that values are correctly mapped
        assert np.array_equal(
            result.blendshape_values[:, 0],
            face_clip.blendshape_values[:, 0])
        assert np.array_equal(
            result.blendshape_values[:, 1],
            face_clip.blendshape_values[:, 1])
        assert np.array_equal(
            result.blendshape_values[:, 2],
            face_clip.blendshape_values[:, 2])
        assert np.array_equal(
            result.blendshape_values[:, 3],
            face_clip.blendshape_values[:, 3])

    def test_call_with_partial_mapping(self, face_clip: FaceClip):
        """Test __call__ method with partial mapping."""
        partial_mapping = {
            'EyeBlinkLeft': 'eye_blink_L',
            'MouthSmile': 'mouth_smile'
        }

        rename = Rename(
            name='test_rename_partial',
            bs_names_mapping=partial_mapping,
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = rename(face_clip)

        # Should return a new FaceClip instance with only mapped blendshapes
        assert result is not face_clip
        assert result.blendshape_names == ['EyeBlinkLeft', 'MouthSmile']
        assert result.blendshape_values.shape == (3, 2)  # 3 frames, 2 blendshapes

        # Check that values are correctly mapped
        # EyeBlinkLeft (index 0) <- eye_blink_L (index 0)
        assert np.array_equal(
            result.blendshape_values[:, 0],
            face_clip.blendshape_values[:, 0])
        # MouthSmile (index 1) <- mouth_smile (index 2)
        assert np.array_equal(
            result.blendshape_values[:, 1],
            face_clip.blendshape_values[:, 2])

    def test_call_with_reordered_mapping(self, face_clip: FaceClip):
        """Test __call__ method with reordered mapping."""
        reordered_mapping = {
            'BrowUp': 'brow_up',
            'MouthSmile': 'mouth_smile',
            'EyeBlinkRight': 'eye_blink_R',
            'EyeBlinkLeft': 'eye_blink_L'
        }

        rename = Rename(
            name='test_rename_reordered',
            bs_names_mapping=reordered_mapping,
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = rename(face_clip)

        # Should return a new FaceClip instance with reordered blendshapes
        assert result is not face_clip
        expected_names = ['BrowUp', 'MouthSmile', 'EyeBlinkRight', 'EyeBlinkLeft']
        assert result.blendshape_names == expected_names
        assert result.blendshape_values.shape == (3, 4)

        # Check that values are correctly mapped in new order
        # BrowUp (index 0) <- brow_up (index 3)
        assert np.array_equal(
            result.blendshape_values[:, 0],
            face_clip.blendshape_values[:, 3])
        # MouthSmile (index 1) <- mouth_smile (index 2)
        assert np.array_equal(
            result.blendshape_values[:, 1],
            face_clip.blendshape_values[:, 2])
        # EyeBlinkRight (index 2) <- eye_blink_R (index 1)
        assert np.array_equal(
            result.blendshape_values[:, 2],
            face_clip.blendshape_values[:, 1])
        # EyeBlinkLeft (index 3) <- eye_blink_L (index 0)
        assert np.array_equal(
            result.blendshape_values[:, 3],
            face_clip.blendshape_values[:, 0])

    def test_call_with_no_matching_blendshapes(self, face_clip: FaceClip):
        """Test __call__ method with no matching blendshapes."""
        non_matching_mapping = {
            'NewName': 'nonexistent_blendshape'
        }

        rename = Rename(
            name='test_rename_no_match',
            bs_names_mapping=non_matching_mapping,
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = rename(face_clip)

        # Should return a new FaceClip instance with empty values
        assert result is not face_clip
        assert result.blendshape_names == ['NewName']
        assert result.blendshape_values.shape == (3, 1)
        assert np.all(result.blendshape_values == 0)

    def test_call_with_none_src_names(self, face_clip: FaceClip):
        """Test __call__ method with None source names."""
        mapping_with_none = {
            'EyeBlinkLeft': 'eye_blink_L',
            'NewName': None
        }

        rename = Rename(
            name='test_rename_none',
            bs_names_mapping=mapping_with_none,
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = rename(face_clip)

        # Should return a new FaceClip instance
        assert result is not face_clip
        assert result.blendshape_names == ['EyeBlinkLeft', 'NewName']
        assert result.blendshape_values.shape == (3, 2)

        # EyeBlinkLeft should have correct values
        assert np.array_equal(
            result.blendshape_values[:, 0],
            face_clip.blendshape_values[:, 0])
        # NewName should have zero values
        assert np.all(result.blendshape_values[:, 1] == 0)

    def test_call_preserves_original(
            self,
            rename_with_dict: Rename,
            face_clip: FaceClip):
        """Test that __call__ method preserves the original face_clip."""
        original_names = face_clip.blendshape_names.copy()
        original_values = face_clip.blendshape_values.copy()
        result = rename_with_dict(face_clip)

        # Original should remain unchanged
        assert face_clip.blendshape_names == original_names
        assert np.array_equal(face_clip.blendshape_values, original_values)

        # Result should be different
        assert result.blendshape_names != original_names

    def test_call_with_single_frame(self):
        """Test __call__ method with single frame face clip."""
        blendshape_names = ['eye_blink_L', 'mouth_smile']
        blendshape_values = np.array([[0.3, 0.7]], dtype=np.float16)

        face_clip = FaceClip(
            blendshape_names=blendshape_names,
            blendshape_values=blendshape_values,
            dtype=np.float16
        )

        mapping = {
            'EyeBlinkLeft': 'eye_blink_L',
            'MouthSmile': 'mouth_smile'
        }

        rename = Rename(
            name='test_rename',
            bs_names_mapping=mapping,
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = rename(face_clip)

        # Check single frame renaming
        assert result.blendshape_names == ['EyeBlinkLeft', 'MouthSmile']
        assert result.blendshape_values.shape == (1, 2)
        assert result.blendshape_values[0, 0] == 0.3  # EyeBlinkLeft <- eye_blink_L
        assert result.blendshape_values[0, 1] == 0.7  # MouthSmile <- mouth_smile

    def test_call_with_zero_frames(self):
        """Test __call__ method with zero frames face clip."""
        blendshape_names = ['eye_blink_L', 'mouth_smile']
        blendshape_values = np.zeros((0, 2), dtype=np.float16)

        face_clip = FaceClip(
            blendshape_names=blendshape_names,
            blendshape_values=blendshape_values,
            dtype=np.float16
        )

        mapping = {
            'EyeBlinkLeft': 'eye_blink_L',
            'MouthSmile': 'mouth_smile'
        }

        rename = Rename(
            name='test_rename',
            bs_names_mapping=mapping,
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = rename(face_clip)

        # Should handle empty array gracefully
        assert result.blendshape_values.shape == (0, 2)
        assert result.blendshape_names == ['EyeBlinkLeft', 'MouthSmile']

    def test_call_with_different_dtypes(self):
        """Test __call__ method with different dtypes."""
        blendshape_names = ['eye_blink_L', 'mouth_smile']
        blendshape_values = np.array([[0.3, 0.7]], dtype=np.float32)

        face_clip = FaceClip(
            blendshape_names=blendshape_names,
            blendshape_values=blendshape_values,
            dtype=np.float32
        )

        mapping = {
            'EyeBlinkLeft': 'eye_blink_L',
            'MouthSmile': 'mouth_smile'
        }

        rename = Rename(
            name='test_rename',
            bs_names_mapping=mapping,
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = rename(face_clip)

        # Should preserve dtype
        assert result.dtype == np.float32
        assert result.blendshape_values.dtype == np.float32

    def test_call_with_timeline_start_idx(self, face_clip: FaceClip):
        """Test __call__ method preserves timeline_start_idx."""
        mapping = {
            'EyeBlinkLeft': 'eye_blink_L',
            'MouthSmile': 'mouth_smile'
        }

        rename = Rename(
            name='test_rename',
            bs_names_mapping=mapping,
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = rename(face_clip)

        # Should preserve timeline_start_idx
        assert result.timeline_start_idx == face_clip.timeline_start_idx

    def test_call_with_none_timeline_start_idx(self):
        """Test __call__ method with None timeline_start_idx."""
        blendshape_names = ['eye_blink_L', 'mouth_smile']
        blendshape_values = np.array([[0.3, 0.7]], dtype=np.float16)

        face_clip = FaceClip(
            blendshape_names=blendshape_names,
            blendshape_values=blendshape_values,
            dtype=np.float16,
            timeline_start_idx=None
        )

        mapping = {
            'EyeBlinkLeft': 'eye_blink_L',
            'MouthSmile': 'mouth_smile'
        }

        rename = Rename(
            name='test_rename',
            bs_names_mapping=mapping,
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = rename(face_clip)

        # Should preserve None timeline_start_idx
        assert result.timeline_start_idx is None

    def test_name_idx_mapping_creation(self, bs_names_mapping_dict: dict[str, str]):
        """Test that name_idx_mapping is correctly created."""
        rename = Rename(
            name='test_rename',
            bs_names_mapping=bs_names_mapping_dict,
            logger_cfg={'logger_name': 'test_logger'}
        )

        # Check that name_idx_mapping is correctly created
        # name_idx_mapping maps dst_name to dst_index
        expected_mapping = {
            'EyeBlinkLeft': 0,
            'EyeBlinkRight': 1,
            'MouthSmile': 2,
            'BrowUp': 3
        }
        assert rename.name_idx_mapping == expected_mapping

    def test_name_idx_mapping_with_reordered_dict(self):
        """Test name_idx_mapping with reordered dictionary."""
        reordered_mapping = {
            'BrowUp': 'brow_up',
            'MouthSmile': 'mouth_smile',
            'EyeBlinkRight': 'eye_blink_R',
            'EyeBlinkLeft': 'eye_blink_L'
        }

        rename = Rename(
            name='test_rename',
            bs_names_mapping=reordered_mapping,
            logger_cfg={'logger_name': 'test_logger'}
        )

        # Check that name_idx_mapping follows the order of keys in the dictionary
        expected_mapping = {
            'BrowUp': 0,
            'MouthSmile': 1,
            'EyeBlinkRight': 2,
            'EyeBlinkLeft': 3
        }
        assert rename.name_idx_mapping == expected_mapping
