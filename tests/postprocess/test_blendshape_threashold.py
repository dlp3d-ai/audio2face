import numpy as np
import pytest

from audio2face.data_structures.face_clip import FaceClip
from audio2face.postprocess.blendshape_threshold import BlendshapeThreshold


@pytest.fixture
def sample_blendshape_names() -> list[str]:
    """Fixture for creating sample blendshape names."""
    return ['eye_blink_L', 'eye_blink_R', 'mouth_smile', 'brow_up', 'mouth_shrug_upper']


@pytest.fixture
def sample_blendshape_values() -> np.ndarray:
    """Fixture for creating sample blendshape values."""
    return np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.6, 0.7, 0.8, 0.9],
        [0.9, 1.0, 1.1, 1.2, 1.3]
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
def thresholds() -> dict[str, float]:
    """Fixture for creating thresholds."""
    return {
        'eye_blink_L': 0.15,
        'mouth_smile': 0.20,
        'brow_up': 0.25
    }


@pytest.fixture
def blendshape_threshold_with_thresholds(
        thresholds: dict[str, float]) -> BlendshapeThreshold:
    """Fixture for creating a BlendshapeThreshold instance with thresholds."""
    return BlendshapeThreshold(
        name='test_threshold',
        thresholds=thresholds,
        logger_cfg={'logger_name': 'test_logger'}
    )


@pytest.fixture
def blendshape_threshold_empty() -> BlendshapeThreshold:
    """Fixture for creating a BlendshapeThreshold instance with empty thresholds."""
    return BlendshapeThreshold(
        name='test_threshold_empty',
        thresholds={},
        logger_cfg={'logger_name': 'test_logger'}
    )


class TestBlendshapeThreshold:
    """Test class for BlendshapeThreshold."""

    def test_init_with_thresholds(self, thresholds: dict[str, float]):
        """Test successful initialization with thresholds."""
        threshold = BlendshapeThreshold(
            name='test_threshold',
            thresholds=thresholds,
            logger_cfg={'logger_name': 'test_logger'}
        )

        assert threshold.name == 'test_threshold'
        assert threshold.thresholds == thresholds
        assert threshold.logger_cfg['logger_name'] == 'test_threshold'

    def test_init_with_empty_thresholds(self):
        """Test successful initialization with empty thresholds."""
        threshold = BlendshapeThreshold(
            name='test_threshold_empty',
            thresholds={},
            logger_cfg={'logger_name': 'test_logger'}
        )

        assert threshold.name == 'test_threshold_empty'
        assert threshold.thresholds == {}
        assert threshold.logger_cfg['logger_name'] == 'test_threshold_empty'

    def test_call_no_thresholds(self, blendshape_threshold_empty: BlendshapeThreshold,
                               face_clip: FaceClip):
        """Test __call__ method with no thresholds."""
        result = blendshape_threshold_empty(face_clip)

        # Should return a clone without any modifications
        assert result is not face_clip
        assert result.blendshape_names == face_clip.blendshape_names
        assert np.array_equal(result.blendshape_values, face_clip.blendshape_values)
        assert result.dtype == face_clip.dtype

    def test_call_with_thresholds(
            self, blendshape_threshold_with_thresholds: BlendshapeThreshold,
            face_clip: FaceClip):
        """Test __call__ method with thresholds."""
        result = blendshape_threshold_with_thresholds(face_clip)

        # Check that values below or equal to thresholds are set to 0
        # eye_blink_L: threshold=0.15, values [0.1, 0.5, 0.9] -> [0.0, 0.5, 0.9]
        eye_blink_l_idx = face_clip.blendshape_names.index('eye_blink_L')
        expected_eye_blink_l = np.array([0.0, 0.5, 0.9], dtype=np.float32)
        assert np.array_equal(
            result.blendshape_values[:, eye_blink_l_idx], expected_eye_blink_l)

        # mouth_smile: threshold=0.20, values [0.3, 0.7, 1.1] -> [0.3, 0.7, 1.1]
        mouth_smile_idx = face_clip.blendshape_names.index('mouth_smile')
        expected_mouth_smile = np.array([0.3, 0.7, 1.1], dtype=np.float32)
        assert np.array_equal(
            result.blendshape_values[:, mouth_smile_idx], expected_mouth_smile)

        # brow_up: threshold=0.25, values [0.4, 0.8, 1.2] -> [0.4, 0.8, 1.2]
        brow_up_idx = face_clip.blendshape_names.index('brow_up')
        expected_brow_up = np.array([0.4, 0.8, 1.2], dtype=np.float32)
        assert np.array_equal(
            result.blendshape_values[:, brow_up_idx], expected_brow_up)

        # Other values should remain unchanged
        eye_blink_r_idx = face_clip.blendshape_names.index('eye_blink_R')
        mouth_shrug_upper_idx = face_clip.blendshape_names.index('mouth_shrug_upper')
        assert np.array_equal(
            result.blendshape_values[:, eye_blink_r_idx],
            face_clip.blendshape_values[:, eye_blink_r_idx]
        )
        assert np.array_equal(
            result.blendshape_values[:, mouth_shrug_upper_idx],
            face_clip.blendshape_values[:, mouth_shrug_upper_idx]
        )

    def test_call_with_nonexistent_blendshape(self, face_clip: FaceClip):
        """Test __call__ method with thresholds for non-existent blendshapes."""
        threshold = BlendshapeThreshold(
            name='test_threshold_nonexistent',
            thresholds={'nonexistent_blendshape': 0.5},
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = threshold(face_clip)

        # Should return a clone without any modifications
        assert result is not face_clip
        assert result.blendshape_names == face_clip.blendshape_names
        assert np.array_equal(result.blendshape_values, face_clip.blendshape_values)
        assert result.dtype == face_clip.dtype

    def test_call_with_mixed_existent_nonexistent(self, face_clip: FaceClip):
        """Test __call__ method with mix of existent and non-existent blendshapes."""
        threshold = BlendshapeThreshold(
            name='test_threshold_mixed',
            thresholds={
                'eye_blink_L': 0.15,  # existent
                'nonexistent_blendshape': 0.5  # non-existent
            },
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = threshold(face_clip)

        # eye_blink_L should be processed
        eye_blink_l_idx = face_clip.blendshape_names.index('eye_blink_L')
        expected_eye_blink_l = np.array([0.0, 0.5, 0.9], dtype=np.float32)
        assert np.array_equal(
            result.blendshape_values[:, eye_blink_l_idx], expected_eye_blink_l)

        # Other values should remain unchanged
        eye_blink_r_idx = face_clip.blendshape_names.index('eye_blink_R')
        assert np.array_equal(
            result.blendshape_values[:, eye_blink_r_idx],
            face_clip.blendshape_values[:, eye_blink_r_idx]
        )

    def test_call_preserves_original(
            self, blendshape_threshold_with_thresholds: BlendshapeThreshold,
            face_clip: FaceClip):
        """Test that __call__ method preserves the original face_clip."""
        original_values = face_clip.blendshape_values.copy()
        result = blendshape_threshold_with_thresholds(face_clip)

        # Original should remain unchanged
        assert np.array_equal(face_clip.blendshape_values, original_values)
        # Result should be different
        assert not np.array_equal(result.blendshape_values, original_values)

    def test_call_with_exact_threshold_values(self):
        """Test __call__ method with values exactly equal to thresholds."""
        blendshape_names = ['test_blendshape']
        # exactly at thresholds
        blendshape_values = np.array(
            [[0.15, 0.20, 0.25]], dtype=np.float32).reshape(3, 1)
        face_clip = FaceClip(
            blendshape_names=blendshape_names,
            blendshape_values=blendshape_values,
            dtype=np.float32
        )

        threshold = BlendshapeThreshold(
            name='test_threshold_exact',
            thresholds={'test_blendshape': 0.15},
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = threshold(face_clip)

        # Values equal to threshold should be set to 0
        expected_values = np.array([[0.0, 0.20, 0.25]], dtype=np.float32).reshape(3, 1)
        assert np.array_equal(result.blendshape_values, expected_values)

    def test_call_with_negative_thresholds(self, face_clip: FaceClip):
        """Test __call__ method with negative thresholds."""
        threshold = BlendshapeThreshold(
            name='test_threshold_negative',
            thresholds={'eye_blink_L': -0.1},
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = threshold(face_clip)

        # Values below negative threshold should be set to 0
        eye_blink_l_idx = face_clip.blendshape_names.index('eye_blink_L')
        # All values [0.1, 0.5, 0.9] are above -0.1, so none should be set to 0
        assert np.array_equal(
            result.blendshape_values[:, eye_blink_l_idx],
            face_clip.blendshape_values[:, eye_blink_l_idx]
        )

    def test_call_with_high_thresholds(self, face_clip: FaceClip):
        """Test __call__ method with high thresholds."""
        threshold = BlendshapeThreshold(
            name='test_threshold_high',
            thresholds={'eye_blink_L': 2.0},
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = threshold(face_clip)

        # All values [0.1, 0.5, 0.9] are below 2.0, so all should be set to 0
        eye_blink_l_idx = face_clip.blendshape_names.index('eye_blink_L')
        expected_values = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        assert np.array_equal(
            result.blendshape_values[:, eye_blink_l_idx], expected_values)

    def test_call_with_single_frame(self):
        """Test __call__ method with single frame data."""
        blendshape_names = ['test_blendshape']
        blendshape_values = np.array([[0.1, 0.3, 0.5]], dtype=np.float32).reshape(3, 1)
        face_clip = FaceClip(
            blendshape_names=blendshape_names,
            blendshape_values=blendshape_values,
            dtype=np.float32
        )

        threshold = BlendshapeThreshold(
            name='test_threshold_single_frame',
            thresholds={'test_blendshape': 0.2},
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = threshold(face_clip)

        # Values [0.1, 0.3, 0.5] with threshold 0.2 -> [0.0, 0.3, 0.5]
        expected_values = np.array([[0.0, 0.3, 0.5]], dtype=np.float32).reshape(3, 1)
        assert np.array_equal(result.blendshape_values, expected_values)

    def test_call_with_zero_frames(self):
        """Test __call__ method with zero frames data."""
        blendshape_names = ['test_blendshape']
        blendshape_values = np.array([], dtype=np.float32).reshape(0, 1)
        face_clip = FaceClip(
            blendshape_names=blendshape_names,
            blendshape_values=blendshape_values,
            dtype=np.float32
        )

        threshold = BlendshapeThreshold(
            name='test_threshold_zero_frames',
            thresholds={'test_blendshape': 0.5},
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = threshold(face_clip)

        # Should return empty array without error
        assert result.blendshape_values.shape == (0, 1)
        assert result.blendshape_names == blendshape_names

    def test_call_with_multiple_blendshapes_same_threshold(self, face_clip: FaceClip):
        """Test __call__ method with multiple blendshapes having the same threshold."""
        threshold = BlendshapeThreshold(
            name='test_threshold_multiple',
            thresholds={
                'eye_blink_L': 0.15,
                'eye_blink_R': 0.15,
                'mouth_smile': 0.15
            },
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = threshold(face_clip)

        # All three blendshapes should be processed with threshold 0.15
        eye_blink_l_idx = face_clip.blendshape_names.index('eye_blink_L')
        eye_blink_r_idx = face_clip.blendshape_names.index('eye_blink_R')
        mouth_smile_idx = face_clip.blendshape_names.index('mouth_smile')

        # eye_blink_L: [0.1, 0.5, 0.9] -> [0.0, 0.5, 0.9]
        expected_eye_blink_l = np.array([0.0, 0.5, 0.9], dtype=np.float32)
        assert np.array_equal(
            result.blendshape_values[:, eye_blink_l_idx], expected_eye_blink_l)

        # eye_blink_R: [0.2, 0.6, 1.0] -> [0.2, 0.6, 1.0]
        expected_eye_blink_r = np.array([0.2, 0.6, 1.0], dtype=np.float32)
        assert np.array_equal(
            result.blendshape_values[:, eye_blink_r_idx], expected_eye_blink_r)

        # mouth_smile: [0.3, 0.7, 1.1] -> [0.3, 0.7, 1.1]
        expected_mouth_smile = np.array([0.3, 0.7, 1.1], dtype=np.float32)
        assert np.array_equal(
            result.blendshape_values[:, mouth_smile_idx], expected_mouth_smile)

