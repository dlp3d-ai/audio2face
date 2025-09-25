
import numpy as np
import pytest

from audio2face.data_structures.face_clip import FaceClip
from audio2face.postprocess.blendshape_clip import BlendshapeClip


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
    ], dtype=np.float16)


@pytest.fixture
def face_clip(sample_blendshape_names: list[str],
              sample_blendshape_values: np.ndarray) -> FaceClip:
    """Fixture for creating a FaceClip instance."""
    return FaceClip(
        blendshape_names=sample_blendshape_names,
        blendshape_values=sample_blendshape_values,
        dtype=np.float16
    )


@pytest.fixture
def lowerbounds() -> dict[str, float]:
    """Fixture for creating lower bounds."""
    return {
        'eye_blink_L': 0.0,
        'mouth_smile': 0.1
    }


@pytest.fixture
def upperbounds() -> dict[str, float]:
    """Fixture for creating upper bounds."""
    return {
        'eye_blink_R': 0.8,
        'brow_up': 1.0
    }


@pytest.fixture
def blendshape_clip_with_bounds(lowerbounds: dict[str, float],
                               upperbounds: dict[str, float]) -> BlendshapeClip:
    """Fixture for creating a BlendshapeClip instance with bounds."""
    return BlendshapeClip(
        name='test_clip',
        lowerbounds=lowerbounds,
        upperbounds=upperbounds,
        logger_cfg={'logger_name': 'test_logger'}
    )


@pytest.fixture
def blendshape_clip_no_bounds() -> BlendshapeClip:
    """Fixture for creating a BlendshapeClip instance without bounds."""
    return BlendshapeClip(
        name='test_clip_no_bounds',
        logger_cfg={'logger_name': 'test_logger'}
    )


class TestBlendshapeClip:
    """Test class for BlendshapeClip."""

    def test_init_with_bounds(self, lowerbounds: dict[str, float],
                             upperbounds: dict[str, float]):
        """Test successful initialization with bounds."""
        clip = BlendshapeClip(
            name='test_clip',
            lowerbounds=lowerbounds,
            upperbounds=upperbounds,
            logger_cfg={'logger_name': 'test_logger'}
        )

        assert clip.name == 'test_clip'
        assert clip.lowerbounds == lowerbounds
        assert clip.upperbounds == upperbounds
        assert clip.logger_cfg['logger_name'] == 'test_clip'

    def test_init_without_bounds(self):
        """Test successful initialization without bounds."""
        clip = BlendshapeClip(
            name='test_clip_no_bounds',
            logger_cfg={'logger_name': 'test_logger'}
        )

        assert clip.name == 'test_clip_no_bounds'
        assert clip.lowerbounds == {}
        assert clip.upperbounds == {}
        assert clip.logger_cfg['logger_name'] == 'test_clip_no_bounds'

    def test_init_with_none_bounds(self):
        """Test initialization with None bounds."""
        clip = BlendshapeClip(
            name='test_clip_none',
            lowerbounds=None,
            upperbounds=None,
            logger_cfg={'logger_name': 'test_logger'}
        )

        assert clip.name == 'test_clip_none'
        assert clip.lowerbounds == {}
        assert clip.upperbounds == {}

    def test_call_no_bounds(self, blendshape_clip_no_bounds: BlendshapeClip,
                           face_clip: FaceClip):
        """Test __call__ method with no bounds."""
        result = blendshape_clip_no_bounds(face_clip)

        # Should return a clone without any modifications
        assert result is not face_clip
        assert result.blendshape_names == face_clip.blendshape_names
        assert np.array_equal(result.blendshape_values, face_clip.blendshape_values)
        assert result.dtype == face_clip.dtype

    def test_call_with_lowerbounds(self, blendshape_clip_with_bounds: BlendshapeClip,
                                  face_clip: FaceClip):
        """Test __call__ method with lower bounds."""
        result = blendshape_clip_with_bounds(face_clip)

        # Check that values are clipped to lower bounds
        # eye_blink_L should be clipped to minimum 0.0
        eye_blink_l_idx = face_clip.blendshape_names.index('eye_blink_L')
        assert np.all(result.blendshape_values[:, eye_blink_l_idx] >= 0.0)

        # mouth_smile should be clipped to minimum 0.1
        mouth_smile_idx = face_clip.blendshape_names.index('mouth_smile')
        assert np.all(result.blendshape_values[:, mouth_smile_idx] >= 0.1)

        # Other values should remain unchanged
        mouth_shrug_upper_idx = face_clip.blendshape_names.index('mouth_shrug_upper')
        assert np.array_equal(
            result.blendshape_values[:, mouth_shrug_upper_idx],
            face_clip.blendshape_values[:, mouth_shrug_upper_idx]
        )

    def test_call_with_upperbounds(self, blendshape_clip_with_bounds: BlendshapeClip,
                                  face_clip: FaceClip):
        """Test __call__ method with upper bounds."""
        result = blendshape_clip_with_bounds(face_clip)

        # Check that values are clipped to upper bounds
        # eye_blink_R should be clipped to maximum 0.8
        eye_blink_r_idx = face_clip.blendshape_names.index('eye_blink_R')
        assert np.all(result.blendshape_values[:, eye_blink_r_idx] <= 0.8)

        # brow_up should be clipped to maximum 1.0
        brow_up_idx = face_clip.blendshape_names.index('brow_up')
        assert np.all(result.blendshape_values[:, brow_up_idx] <= 1.0)

        # Other values should remain unchanged
        eye_blink_l_idx = face_clip.blendshape_names.index('eye_blink_L')
        mouth_smile_idx = face_clip.blendshape_names.index('mouth_smile')
        assert np.array_equal(
            result.blendshape_values[:, eye_blink_l_idx],
            face_clip.blendshape_values[:, eye_blink_l_idx]
        )
        assert np.array_equal(
            result.blendshape_values[:, mouth_smile_idx],
            face_clip.blendshape_values[:, mouth_smile_idx]
        )

    def test_call_with_both_bounds(self, blendshape_clip_with_bounds: BlendshapeClip,
                                  face_clip: FaceClip):
        """Test __call__ method with both lower and upper bounds."""
        result = blendshape_clip_with_bounds(face_clip)

        # Check that values are properly clipped
        # eye_blink_L: should be >= 0.0
        eye_blink_l_idx = face_clip.blendshape_names.index('eye_blink_L')
        assert np.all(result.blendshape_values[:, eye_blink_l_idx] >= 0.0)

        # mouth_smile: should be >= 0.1
        mouth_smile_idx = face_clip.blendshape_names.index('mouth_smile')
        assert np.all(result.blendshape_values[:, mouth_smile_idx] >= 0.1)

        # eye_blink_R: should be <= 0.8
        eye_blink_r_idx = face_clip.blendshape_names.index('eye_blink_R')
        assert np.all(result.blendshape_values[:, eye_blink_r_idx] <= 0.8)

        # brow_up: should be <= 1.0
        brow_up_idx = face_clip.blendshape_names.index('brow_up')
        assert np.all(result.blendshape_values[:, brow_up_idx] <= 1.0)

    def test_call_with_nonexistent_blendshape(self, face_clip: FaceClip):
        """Test __call__ method with bounds for non-existent blendshapes."""
        clip = BlendshapeClip(
            name='test_clip',
            lowerbounds={'nonexistent_blendshape': 0.0},
            upperbounds={'another_nonexistent': 1.0},
            logger_cfg={'logger_name': 'test_logger'}
        )

        # Should not raise an error, just log warnings
        result = clip(face_clip)

        # Should return a clone without any modifications
        assert result is not face_clip
        assert np.array_equal(result.blendshape_values, face_clip.blendshape_values)

    def test_call_with_mixed_existent_nonexistent(self, face_clip: FaceClip):
        """Test __call__ method with mix of existent and non-existent blendshapes."""
        clip = BlendshapeClip(
            name='test_clip',
            lowerbounds={
                'eye_blink_L': 0.0,  # exists
                'nonexistent': 0.1   # doesn't exist
            },
            upperbounds={
                'brow_up': 0.9,      # exists
                'another_nonexistent': 1.0  # doesn't exist
            },
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = clip(face_clip)

        # Should apply bounds to existing blendshapes
        eye_blink_l_idx = face_clip.blendshape_names.index('eye_blink_L')
        brow_up_idx = face_clip.blendshape_names.index('brow_up')

        assert np.all(result.blendshape_values[:, eye_blink_l_idx] >= 0.0)
        assert np.all(result.blendshape_values[:, brow_up_idx] <= 0.9)

        # Should ignore non-existent blendshapes
        eye_blink_r_idx = face_clip.blendshape_names.index('eye_blink_R')
        mouth_smile_idx = face_clip.blendshape_names.index('mouth_smile')
        assert np.array_equal(
            result.blendshape_values[:, eye_blink_r_idx],
            face_clip.blendshape_values[:, eye_blink_r_idx]
        )
        assert np.array_equal(
            result.blendshape_values[:, mouth_smile_idx],
            face_clip.blendshape_values[:, mouth_smile_idx]
        )

    def test_call_preserves_original(self, blendshape_clip_with_bounds: BlendshapeClip,
                                   face_clip: FaceClip):
        """Test that __call__ method preserves the original face_clip."""
        original_values = face_clip.blendshape_values.copy()
        result = blendshape_clip_with_bounds(face_clip)

        # Original should remain unchanged
        assert np.array_equal(face_clip.blendshape_values, original_values)

        # Result should be different (clipped)
        assert not np.array_equal(result.blendshape_values, original_values)

    def test_call_with_empty_bounds(self, face_clip: FaceClip):
        """Test __call__ method with empty bounds dictionaries."""
        clip = BlendshapeClip(
            name='test_clip',
            lowerbounds={},
            upperbounds={},
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = clip(face_clip)

        # Should return a clone without any modifications
        assert result is not face_clip
        assert np.array_equal(result.blendshape_values, face_clip.blendshape_values)

    def test_call_with_extreme_bounds(self, face_clip: FaceClip):
        """Test __call__ method with extreme bound values."""
        clip = BlendshapeClip(
            name='test_clip',
            lowerbounds={'eye_blink_L': 0.5},  # Very high lower bound
            upperbounds={'brow_up': 0.2},      # Very low upper bound
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = clip(face_clip)

        # Check extreme clipping behavior
        eye_blink_l_idx = face_clip.blendshape_names.index('eye_blink_L')
        brow_up_idx = face_clip.blendshape_names.index('brow_up')

        # All eye_blink_L values should be >= 0.5
        assert np.all(result.blendshape_values[:, eye_blink_l_idx] >= 0.5)

        # All brow_up values should be <= 0.2
        assert np.all(result.blendshape_values[:, brow_up_idx] <= 0.2)

    def test_call_with_negative_bounds(self, face_clip: FaceClip):
        """Test __call__ method with negative bound values."""
        clip = BlendshapeClip(
            name='test_clip',
            lowerbounds={'eye_blink_L': -0.1},  # Negative lower bound
            upperbounds={'brow_up': -0.05},     # Negative upper bound
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = clip(face_clip)

        # Check negative clipping behavior
        eye_blink_l_idx = face_clip.blendshape_names.index('eye_blink_L')
        brow_up_idx = face_clip.blendshape_names.index('brow_up')

        # All eye_blink_L values should be >= -0.1
        assert np.all(result.blendshape_values[:, eye_blink_l_idx] >= -0.1)

        # All brow_up values should be <= -0.05
        assert np.all(result.blendshape_values[:, brow_up_idx] <= -0.05)

    def test_call_with_single_frame(self):
        """Test __call__ method with single frame face clip."""
        blendshape_names = ['eye_blink_L', 'mouth_smile']
        blendshape_values = np.array([[0.3, 0.7]], dtype=np.float16)

        face_clip = FaceClip(
            blendshape_names=blendshape_names,
            blendshape_values=blendshape_values,
            dtype=np.float16
        )

        clip = BlendshapeClip(
            name='test_clip',
            lowerbounds={'eye_blink_L': 0.5},
            upperbounds={'mouth_smile': 0.5},
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = clip(face_clip)

        # Check single frame clipping
        assert result.blendshape_values[0, 0] >= 0.5  # eye_blink_L
        assert result.blendshape_values[0, 1] <= 0.5  # mouth_smile

    def test_call_with_zero_frames(self):
        """Test __call__ method with zero frames face clip."""
        blendshape_names = ['eye_blink_L', 'mouth_smile']
        blendshape_values = np.zeros((0, 2), dtype=np.float16)

        face_clip = FaceClip(
            blendshape_names=blendshape_names,
            blendshape_values=blendshape_values,
            dtype=np.float16
        )

        clip = BlendshapeClip(
            name='test_clip',
            lowerbounds={'eye_blink_L': 0.0},
            upperbounds={'mouth_smile': 1.0},
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = clip(face_clip)

        # Should handle empty array gracefully
        assert result.blendshape_values.shape == (0, 2)
        assert result.blendshape_names == blendshape_names
