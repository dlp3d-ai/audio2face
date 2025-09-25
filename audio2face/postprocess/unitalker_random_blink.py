import random
from typing import Any

import numpy as np

from ..data_structures.face_clip import FaceClip
from .base_postprocess import BasePostprocess


class UnitalkerRandomBlink(BasePostprocess):
    """Unitalker-specific random blink postprocessor.

    Adds random blink effects to facial expression animations output by Unitalker,
    making facial expressions more natural and lively. Blink intervals and timing
    are randomized.
    """

    def __init__(
            self,
            fps: float,
            logger_cfg: None | dict[str, Any] = None):
        """Initialize the UnitalkerRandomBlink postprocessor.

        Args:
            fps (float):
                Animation frame rate for calculating blink time intervals.
            logger_cfg (None | dict[str, Any], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        BasePostprocess.__init__(self, logger_cfg)
        self.fps = fps

    def __call__(self, face_clip: FaceClip, **kwargs) -> FaceClip:
        """Add random blink effects to FaceClip.

        Validates that blendshape count is 51 and adds random blink effects to
        the animation:
        - Blink intervals randomly between 3-4 seconds
        - Blink process includes 3-frame gradient effects
        - Blink intensity gradient from 0.4 to 1.0
        - Avoids starting blink at frame 0

        Args:
            face_clip (FaceClip):
                Input facial expression animation clip, must contain 51 blendshapes.
            **kwargs:
                Additional keyword arguments.

        Returns:
            FaceClip:
                Facial expression animation clip with blink effects added.
                Original data is not modified.

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
        set_blink = True
        # avoid frame=0 eye close, loop starts from 1
        for idx in range(1, len(face_clip)):
            if set_blink:
                random_int = random.randint(3, 4)
                set_blink = False
            if (idx / self.fps) % random_int == 0:
                ret_face_clip.blendshape_values[idx - 2:idx + 2, 0:2] = 0.4
                ret_face_clip.blendshape_values[idx - 1:idx + 2, 0:2] = 0.6
                ret_face_clip.blendshape_values[idx, 0:2] = \
                    np.ones_like(ret_face_clip.blendshape_values[idx, 0:2])
                set_blink = True
        return ret_face_clip
