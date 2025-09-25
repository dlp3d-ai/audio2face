import numpy as np
import pytest

from audio2face.data_structures.face_clip import FaceClip
from audio2face.postprocess.blendshape_scale import BlendshapeScale


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
        dtype=np.float32
    )


@pytest.fixture
def scaling_factors() -> dict[str, float]:
    """Fixture for creating scaling factors dictionary."""
    return {
        'eye_blink_L': 2.0,
        'mouth_smile': 1.5,
        'brow_up': 0.8
    }


@pytest.fixture
def blendshape_scale(scaling_factors: dict[str, float]) -> BlendshapeScale:
    """Fixture for creating a BlendshapeScale instance."""
    return BlendshapeScale(
        name='test_scale',
        scaling_factors=scaling_factors,
        logger_cfg={'logger_name': 'test_logger'}
    )


class TestBlendshapeScale:
    """Test class for BlendshapeScale."""

    def test_init(self, scaling_factors: dict[str, float]):
        """Test successful initialization."""
        scale = BlendshapeScale(
            name='test_scale',
            scaling_factors=scaling_factors,
            logger_cfg={'logger_name': 'test_logger'}
        )

        assert scale.name == 'test_scale'
        assert scale.scaling_factors == scaling_factors
        assert scale.logger_cfg['logger_name'] == 'test_scale'

    def test_call_with_matching_blendshapes(
            self,
            blendshape_scale: BlendshapeScale,
            face_clip: FaceClip):
        """Test __call__ method with matching blendshapes."""
        result = blendshape_scale(face_clip)

        # Should return a clone
        assert result is not face_clip
        assert result.blendshape_names == face_clip.blendshape_names
        assert result.dtype == face_clip.dtype

        # Check that specified blendshapes are scaled
        eye_blink_l_idx = face_clip.blendshape_names.index('eye_blink_L')
        mouth_smile_idx = face_clip.blendshape_names.index('mouth_smile')
        brow_up_idx = face_clip.blendshape_names.index('brow_up')

        # Values should be linearly scaled
        original_eye_blink_l = face_clip.blendshape_values[:, eye_blink_l_idx]
        original_mouth_smile = face_clip.blendshape_values[:, mouth_smile_idx]
        original_brow_up = face_clip.blendshape_values[:, brow_up_idx]

        expected_eye_blink_l = original_eye_blink_l * 2.0
        expected_mouth_smile = original_mouth_smile * 1.5
        expected_brow_up = original_brow_up * 0.8

        assert np.allclose(
            result.blendshape_values[:, eye_blink_l_idx], expected_eye_blink_l)
        assert np.allclose(
            result.blendshape_values[:, mouth_smile_idx], expected_mouth_smile)
        assert np.allclose(
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

    def test_call_with_no_matching_blendshapes(self, face_clip: FaceClip):
        """Test __call__ method with no matching blendshapes."""
        scale = BlendshapeScale(
            name='test_scale',
            scaling_factors={'nonexistent_blendshape': 1.0},
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = scale(face_clip)

        # Should return a clone without any modifications
        assert result is not face_clip
        assert np.array_equal(result.blendshape_values, face_clip.blendshape_values)

    def test_call_with_empty_scaling_factors(self, face_clip: FaceClip):
        """Test __call__ method with empty scaling_factors dictionary."""
        scale = BlendshapeScale(
            name='test_scale',
            scaling_factors={},
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = scale(face_clip)

        # Should return a clone without any modifications
        assert result is not face_clip
        assert np.array_equal(result.blendshape_values, face_clip.blendshape_values)

    def test_call_preserves_original(self, blendshape_scale: BlendshapeScale,
                                   face_clip: FaceClip):
        """Test that __call__ method preserves the original face_clip."""
        original_values = face_clip.blendshape_values.copy()
        result = blendshape_scale(face_clip)

        # Original should remain unchanged
        assert np.array_equal(face_clip.blendshape_values, original_values)

        # Result should be different (scaled)
        assert not np.array_equal(result.blendshape_values, original_values)

    def test_call_with_different_scaling_factors(self, face_clip: FaceClip):
        """Test __call__ method with different scaling factors."""
        eye_blink_l_idx = face_clip.blendshape_names.index('eye_blink_L')
        original_values = face_clip.blendshape_values[:, eye_blink_l_idx]

        # Test with scaling factor 1.0 (no change)
        scale_1 = BlendshapeScale(
            name='test_scale_1',
            scaling_factors={'eye_blink_L': 1.0},
            logger_cfg={'logger_name': 'test_logger'}
        )
        result_1 = scale_1(face_clip)
        expected_1 = original_values * 1.0
        assert np.allclose(result_1.blendshape_values[:, eye_blink_l_idx], expected_1)

        # Test with scaling factor 5.0 (amplify)
        scale_5 = BlendshapeScale(
            name='test_scale_5',
            scaling_factors={'eye_blink_L': 5.0},
            logger_cfg={'logger_name': 'test_logger'}
        )
        result_5 = scale_5(face_clip)
        expected_5 = original_values * 5.0
        assert np.allclose(result_5.blendshape_values[:, eye_blink_l_idx], expected_5)

        # Test with scaling factor 0.5 (reduce)
        scale_05 = BlendshapeScale(
            name='test_scale_05',
            scaling_factors={'eye_blink_L': 0.5},
            logger_cfg={'logger_name': 'test_logger'}
        )
        result_05 = scale_05(face_clip)
        expected_05 = original_values * 0.5
        assert np.allclose(result_05.blendshape_values[:, eye_blink_l_idx], expected_05)

        # Results should be different
        assert not np.array_equal(
            result_1.blendshape_values[:, eye_blink_l_idx],
            result_5.blendshape_values[:, eye_blink_l_idx]
        )
        assert not np.array_equal(
            result_1.blendshape_values[:, eye_blink_l_idx],
            result_05.blendshape_values[:, eye_blink_l_idx]
        )

    def test_call_with_single_frame(self):
        """Test __call__ method with single frame face clip."""
        blendshape_names = ['eye_blink_L', 'mouth_smile']
        blendshape_values = np.array([[0.3, 0.7]], dtype=np.float16)

        face_clip = FaceClip(
            blendshape_names=blendshape_names,
            blendshape_values=blendshape_values,
            dtype=np.float16
        )

        scale = BlendshapeScale(
            name='test_scale',
            scaling_factors={'eye_blink_L': 2.0},
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = scale(face_clip)

        # Check single frame scaling
        expected_eye_blink_l = 0.3 * 2.0
        assert np.allclose(result.blendshape_values[0, 0], expected_eye_blink_l)

        # mouth_smile should remain unchanged
        assert result.blendshape_values[0, 1] == 0.7

    def test_call_with_zero_frames(self):
        """Test __call__ method with zero frames face clip."""
        blendshape_names = ['eye_blink_L', 'mouth_smile']
        blendshape_values = np.zeros((0, 2), dtype=np.float16)

        face_clip = FaceClip(
            blendshape_names=blendshape_names,
            blendshape_values=blendshape_values,
            dtype=np.float16
        )

        scale = BlendshapeScale(
            name='test_scale',
            scaling_factors={'eye_blink_L': 2.0},
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = scale(face_clip)

        # Should handle empty array gracefully
        assert result.blendshape_values.shape == (0, 2)
        assert result.blendshape_names == blendshape_names

    def test_call_with_zero_scaling_factor(self, face_clip: FaceClip):
        """Test __call__ method with zero scaling factor."""
        scale = BlendshapeScale(
            name='test_scale',
            scaling_factors={'eye_blink_L': 0.0},
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = scale(face_clip)

        # Should handle zero scaling factor (all values become 0)
        eye_blink_l_idx = face_clip.blendshape_names.index('eye_blink_L')
        assert np.allclose(result.blendshape_values[:, eye_blink_l_idx], 0.0)

    def test_call_with_negative_scaling_factor(self, face_clip: FaceClip):
        """Test __call__ method with negative scaling factor."""
        scale = BlendshapeScale(
            name='test_scale',
            scaling_factors={'eye_blink_L': -1.0},
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = scale(face_clip)

        # Should handle negative scaling factor (values become negative)
        eye_blink_l_idx = face_clip.blendshape_names.index('eye_blink_L')
        original_values = face_clip.blendshape_values[:, eye_blink_l_idx]
        expected_values = original_values * (-1.0)
        assert np.allclose(
            result.blendshape_values[:, eye_blink_l_idx], expected_values)

    def test_call_with_multiple_blendshapes_different_factors(
            self, face_clip: FaceClip):
        """Test __call__ method with multiple blendshapes
        and different scaling factors."""
        scaling_factors = {
            'eye_blink_L': 2.0,
            'mouth_smile': 0.5,
            'brow_up': 1.5,
            'eye_blink_R': 0.8
        }

        scale = BlendshapeScale(
            name='test_scale',
            scaling_factors=scaling_factors,
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = scale(face_clip)

        # Check each blendshape is scaled correctly
        for bs_name, factor in scaling_factors.items():
            if bs_name in face_clip.blendshape_names:
                idx = face_clip.blendshape_names.index(bs_name)
                original_values = face_clip.blendshape_values[:, idx]
                expected_values = original_values * factor
                assert np.allclose(
                    result.blendshape_values[:, idx], expected_values)

    def test_call_with_partial_matching_blendshapes(self, face_clip: FaceClip):
        """Test __call__ method when only some blendshapes in scaling_factors exist."""
        scaling_factors = {
            'eye_blink_L': 2.0,
            'nonexistent_blendshape': 1.5,
            'mouth_smile': 0.8
        }

        scale = BlendshapeScale(
            name='test_scale',
            scaling_factors=scaling_factors,
            logger_cfg={'logger_name': 'test_logger'}
        )

        result = scale(face_clip)

        # Only existing blendshapes should be scaled
        eye_blink_l_idx = face_clip.blendshape_names.index('eye_blink_L')
        mouth_smile_idx = face_clip.blendshape_names.index('mouth_smile')

        original_eye_blink_l = face_clip.blendshape_values[:, eye_blink_l_idx]
        original_mouth_smile = face_clip.blendshape_values[:, mouth_smile_idx]

        assert np.allclose(
            result.blendshape_values[:, eye_blink_l_idx],
            original_eye_blink_l * 2.0)
        assert np.allclose(
            result.blendshape_values[:, mouth_smile_idx],
            original_mouth_smile * 0.8)

        # Other blendshapes should remain unchanged
        eye_blink_r_idx = face_clip.blendshape_names.index('eye_blink_R')
        assert np.array_equal(
            result.blendshape_values[:, eye_blink_r_idx],
            face_clip.blendshape_values[:, eye_blink_r_idx]
        )
