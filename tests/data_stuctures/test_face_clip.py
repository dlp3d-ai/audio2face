import io
from typing import Any

import numpy as np
import pytest

from audio2face.data_structures.face_clip import FaceClip


@pytest.fixture
def sample_blendshape_names() -> list[str]:
    """Fixture for creating sample blendshape names."""
    return ['eye_blink_L', 'eye_blink_R', 'mouth_smile']


@pytest.fixture
def sample_blendshape_values() -> np.ndarray:
    """Fixture for creating sample blendshape values."""
    return np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
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
def face_clip_no_timeline(sample_blendshape_names: list[str],
                         sample_blendshape_values: np.ndarray) -> FaceClip:
    """Fixture for creating a FaceClip instance without timeline."""
    return FaceClip(
        blendshape_names=sample_blendshape_names,
        blendshape_values=sample_blendshape_values,
        dtype=np.float16
    )


@pytest.fixture
def sample_face_dict() -> dict[str, Any]:
    """Fixture for creating sample face dictionary."""
    return {
        'eye_blink_L': np.array([0.1, 0.4, 0.7], dtype=np.float16),
        'eye_blink_R': np.array([0.2, 0.5, 0.8], dtype=np.float16),
        'mouth_smile': np.array([0.3, 0.6, 0.9], dtype=np.float16),
        'n_frames': 3,
        'BlendShapeCount': 3,
        'Timecode': [0, 1, 2]
    }


class TestFaceClip:
    """Test class for FaceClip."""

    def test_init_success(self, sample_blendshape_names: list[str],
                         sample_blendshape_values: np.ndarray):
        """Test successful initialization."""
        face_clip = FaceClip(
            blendshape_names=sample_blendshape_names,
            blendshape_values=sample_blendshape_values,
            dtype=np.float16,
            timeline_start_idx=10
        )

        assert face_clip.blendshape_names == sample_blendshape_names
        assert face_clip.blendshape_values.shape == sample_blendshape_values.shape
        assert face_clip.dtype == np.float16
        assert face_clip.timeline_start_idx == 10
        assert face_clip.blendshape_values.dtype == np.float16

    def test_init_mismatch_error(self, sample_blendshape_names: list[str]):
        """Test initialization with mismatched shapes raises ValueError."""
        # Create values with wrong number of blendshapes
        wrong_values = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float16)

        with pytest.raises(
                ValueError):
            FaceClip(
                blendshape_names=sample_blendshape_names,
                blendshape_values=wrong_values
            )

    def test_len(self, face_clip: FaceClip):
        """Test __len__ method."""
        assert len(face_clip) == 3

    def test_set_timeline_start_idx(self, face_clip: FaceClip):
        """Test set_timeline_start_idx method."""
        face_clip.set_timeline_start_idx(20)
        assert face_clip.timeline_start_idx == 20

        face_clip.set_timeline_start_idx(None)
        assert face_clip.timeline_start_idx is None

    def test_set_blendshape_values_success(self, face_clip: FaceClip):
        """Test set_blendshape_values with valid values."""
        new_values = np.array([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]], dtype=np.float16)
        face_clip.set_blendshape_values(new_values)
        assert face_clip.blendshape_values.shape == (2, 3)
        assert np.array_equal(face_clip.blendshape_values, new_values)

    def test_set_blendshape_values_mismatch_error(self, face_clip: FaceClip):
        """Test set_blendshape_values with mismatched shapes raises ValueError."""
        wrong_values = np.array([[0.9, 0.8], [0.6, 0.5]], dtype=np.float16)

        with pytest.raises(
                ValueError):
            face_clip.set_blendshape_values(wrong_values)

    def test_clone(self, face_clip: FaceClip):
        """Test clone method."""
        cloned = face_clip.clone()

        assert cloned is not face_clip
        assert cloned.blendshape_names == face_clip.blendshape_names
        assert np.array_equal(cloned.blendshape_values, face_clip.blendshape_values)
        assert cloned.dtype == face_clip.dtype
        assert cloned.timeline_start_idx == face_clip.timeline_start_idx
        assert cloned.logger_cfg == face_clip.logger_cfg

    def test_slice_with_timeline(self, face_clip: FaceClip):
        """Test slice method with timeline_start_idx."""
        sliced = face_clip.slice(0, 3)

        assert sliced.blendshape_names == face_clip.blendshape_names
        assert sliced.blendshape_values.shape == (3, 3)
        assert sliced.dtype == face_clip.dtype
        assert sliced.timeline_start_idx == 10
        assert sliced.logger_cfg == face_clip.logger_cfg

        sliced = face_clip.slice(1, 3)

        assert sliced.blendshape_names == face_clip.blendshape_names
        assert sliced.blendshape_values.shape == (2, 3)
        assert sliced.dtype == face_clip.dtype
        assert sliced.timeline_start_idx is None
        assert sliced.logger_cfg == face_clip.logger_cfg

    def test_slice_without_timeline(self, face_clip_no_timeline: FaceClip):
        """Test slice method without timeline_start_idx."""
        sliced = face_clip_no_timeline.slice(1, 3)

        assert sliced.blendshape_names == face_clip_no_timeline.blendshape_names
        assert sliced.blendshape_values.shape == (2, 3)
        assert sliced.dtype == face_clip_no_timeline.dtype
        assert sliced.timeline_start_idx is None
        assert sliced.logger_cfg == face_clip_no_timeline.logger_cfg

    def test_slice_non_zero_start(self, face_clip: FaceClip):
        """Test slice method with non-zero start frame."""
        sliced = face_clip.slice(1, 3)

        # When start_frame != 0, timeline_start_idx should be None
        assert sliced.timeline_start_idx is None

    def test_to_xrmogen_dict(self, face_clip: FaceClip):
        """Test to_xrmogen_dict method."""
        result = face_clip.to_xrmogen_dict()

        assert result['n_frames'] == 3
        assert result['BlendShapeCount'] == 3
        assert result['Timecode'] == [0, 1, 2]
        assert 'eye_blink_L' in result
        assert 'eye_blink_R' in result
        assert 'mouth_smile' in result
        assert np.array_equal(result['eye_blink_L'], face_clip.blendshape_values[:, 0])

    def test_to_xrmogen_npz(self, face_clip: FaceClip):
        """Test to_xrmogen_npz method."""
        result = face_clip.to_xrmogen_npz()

        assert isinstance(result, io.BytesIO)
        assert result.tell() == 0  # Should be at beginning after seek(0)

    def test_from_xrmogen_dict(self, sample_face_dict: dict[str, Any]):
        """Test from_xrmogen_dict class method."""
        face_clip = FaceClip.from_xrmogen_dict(sample_face_dict)

        assert face_clip.blendshape_names == \
            ['eye_blink_L', 'eye_blink_R', 'mouth_smile']
        assert face_clip.blendshape_values.shape == (3, 3)
        assert face_clip.dtype == np.float16
        assert face_clip.timeline_start_idx is None

    def test_from_xrmogen_dict_with_custom_dtype(
            self,
            sample_face_dict: dict[str, Any]):
        """Test from_xrmogen_dict with custom dtype."""
        face_clip = FaceClip.from_xrmogen_dict(sample_face_dict, dtype=np.float32)

        assert face_clip.dtype == np.float32
        assert face_clip.blendshape_values.dtype == np.float32

    def test_from_xrmogen_npz(self, face_clip: FaceClip):
        """Test from_xrmogen_npz class method."""
        npz_io = face_clip.to_xrmogen_npz()
        reconstructed = FaceClip.from_xrmogen_npz(npz_io)

        assert reconstructed.blendshape_names == face_clip.blendshape_names
        assert np.array_equal(
            reconstructed.blendshape_values,
            face_clip.blendshape_values)
        assert reconstructed.dtype == face_clip.dtype

    def test_concat_success(self, face_clip: FaceClip):
        """Test concat method with valid clips."""
        # Create another clip with same blendshape names
        clip2 = FaceClip(
            blendshape_names=face_clip.blendshape_names,
            blendshape_values=np.array([[0.9, 0.8, 0.7]], dtype=np.float16),
            dtype=np.float16,
            timeline_start_idx=20
        )

        concatenated = FaceClip.concat([face_clip, clip2])

        assert concatenated.blendshape_names == face_clip.blendshape_names
        assert concatenated.blendshape_values.shape == (4, 3)  # 3 + 1 frames
        assert concatenated.dtype == face_clip.dtype
        assert concatenated.timeline_start_idx == face_clip.timeline_start_idx

    def test_concat_empty_list_error(self):
        """Test concat method with empty list raises ValueError."""
        with pytest.raises(ValueError):
            FaceClip.concat([])

    def test_concat_different_names_error(self, face_clip: FaceClip):
        """Test concat method with different blendshape names raises ValueError."""
        # Create clip with different blendshape names
        clip2 = FaceClip(
            blendshape_names=['different_name'],
            blendshape_values=np.array([[0.9]], dtype=np.float16),
            dtype=np.float16
        )

        with pytest.raises(
                ValueError):
            FaceClip.concat([face_clip, clip2])

    def test_concat_single_clip(self, face_clip: FaceClip):
        """Test concat method with single clip."""
        concatenated = FaceClip.concat([face_clip])

        assert concatenated.blendshape_names == face_clip.blendshape_names
        assert np.array_equal(
            concatenated.blendshape_values,
            face_clip.blendshape_values)
        assert concatenated.dtype == face_clip.dtype
        assert concatenated.timeline_start_idx == face_clip.timeline_start_idx
