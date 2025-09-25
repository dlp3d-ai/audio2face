from typing import Any

import numpy as np

from ..data_structures.face_clip import FaceClip
from .base_postprocess import BasePostprocess


class UnitalkerClip(BasePostprocess):
    """Unitalker-specific blendshape clipping postprocessor.

    Performs specific clipping processing on Unitalker model's 51 blendshape
    outputs, including scaling of eye-related blendshapes and value limiting
    for specific blendshapes.
    """

    def __init__(self, logger_cfg: None | dict[str, Any] = None):
        """Initialize the UnitalkerClip postprocessor.

        Args:
            logger_cfg (None | dict[str, Any], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        BasePostprocess.__init__(self, logger_cfg)

    def __call__(self, face_clip: FaceClip, **kwargs) -> FaceClip:
        """Perform specialized clipping processing on Unitalker output FaceClip.

        Validates that blendshape count is 51 and clips blendshapes at specific
        positions:
        - Eye-related blendshapes (indices 2-13): clipped to [0,1] range and
          multiplied by 0.8
        - MouthShrugUpper (index 45): clipped to [0,0.01] range
        - EyeBlinkLeft (index 0): clipped to [0,0.01] range
        - EyeBlinkRight (index 1): clipped to [0,0.01] range

        Args:
            face_clip (FaceClip):
                Input facial expression animation clip, must contain 51 blendshapes.
            **kwargs:
                Additional keyword arguments.

        Returns:
            FaceClip:
                Processed facial expression animation clip. Original data
                is not modified.

        Raises:
            ValueError:
                When blendshape count is not 51 or MouthShrugUpper is missing.
        """
        blendshape_names = face_clip.blendshape_names
        if len(blendshape_names) != 51:
            msg = ('blendshape_names length is incorrect, expected 51, ' +
                   f'actual: {len(blendshape_names)}, ' +
                   f'details: {blendshape_names}')
            self.logger.error(msg)
            raise ValueError(msg)
        if 'MouthShrugUpper' not in blendshape_names:
            msg = ('MouthShrugUpper not in blendshape_names, ' +
                   'input FaceClip blendshape_names is incorrect, ' +
                   f'details: {blendshape_names}')
            self.logger.error(msg)
            raise ValueError(msg)
        ret_face_clip = face_clip.clone()
        ret_face_clip.blendshape_values[:,2:14] = \
            np.clip(
                ret_face_clip.blendshape_values[:, 2:14], 0, 1
            ) * 0.8  # eye (more small perturbation)
        ret_face_clip.blendshape_values[:, 45] = \
            np.clip(
                ret_face_clip.blendshape_values[:, 45], 0, 0.01
            )  # MouthShrugUpper

        ret_face_clip.blendshape_values[:, 0] = \
            np.clip(ret_face_clip.blendshape_values[:, 0], 0, 0.01)  # EyeBlinkLeft
        ret_face_clip.blendshape_values[:, 1] = \
            np.clip(ret_face_clip.blendshape_values[:, 1], 0, 0.01)  # EyeBlinkRight
        return ret_face_clip
