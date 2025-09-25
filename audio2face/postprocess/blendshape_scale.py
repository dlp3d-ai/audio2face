from typing import Any

import numpy as np

from ..data_structures.face_clip import FaceClip
from ..utils.log import setup_logger
from .base_postprocess import BasePostprocess


class BlendshapeScale(BasePostprocess):
    """Blendshape value scaling postprocessor.

    Performs linear scaling on specified blendshape values to adjust
    facial expression intensity. Supports configuring different scaling
    factors for different blendshapes through dictionary configuration.
    """

    def __init__(
            self,
            name: str,
            scaling_factors: dict[str, float],
            logger_cfg: None | dict[str, Any] = None):
        """Initialize the BlendshapeScale postprocessor.

        Args:
            name (str):
                Name of the postprocessor.
            scaling_factors (dict[str, float]):
                Scaling factors dictionary, where keys are blendshape names
                and values are corresponding scaling factors. A scaling factor
                of 1.0 means no scaling, >1.0 means amplification, <1.0 means reduction.
            logger_cfg (None | dict[str, Any], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.

        Raises:
            FileNotFoundError:
                When bs_names is a file path but the file does not exist.
        """
        BasePostprocess.__init__(self, logger_cfg)
        self.name = name
        self.logger_cfg["logger_name"] = name
        self.logger = setup_logger(**self.logger_cfg)
        self.scaling_factors = scaling_factors

    def __call__(self, face_clip: FaceClip, **kwargs) -> FaceClip:
        """Perform scaling processing on specified blendshape values in FaceClip.

        Uses linear scaling to process specified blendshape values for
        adjusting facial expression intensity. Warnings are logged for
        non-existent blendshape names.

        Args:
            face_clip (FaceClip):
                Input facial expression animation clip.
            **kwargs:
                Additional keyword arguments.

        Returns:
            FaceClip:
                Processed facial expression animation clip. Original data
                is not modified.
        """
        src_bs_names = face_clip.blendshape_names
        sensitive_idxs = list()
        scaling_factors = list()
        for idx, bs_name in enumerate(src_bs_names):
            if bs_name in self.scaling_factors:
                sensitive_idxs.append(idx)
                scaling_factors.append(self.scaling_factors[bs_name])
        ret_face_clip = face_clip.clone()
        if len(sensitive_idxs) == 0:
            self.logger.warning(
                'No blendshapes found for processing, ' +
                'please check if bs_names is correct. ' +
                f'Expected to process {self.scaling_factors.keys()}, ' +
                f'but received blendshape_names: {src_bs_names}')
            return ret_face_clip
        sensitive_idxs = np.array(sensitive_idxs, dtype=np.int32)
        # src_values: (n_frames, n_blendshapes_to_process)
        src_values = ret_face_clip.blendshape_values[:, sensitive_idxs]
        # scaling_factors: (n_blendshapes_to_process,)
        scaling_factors = np.array(scaling_factors, dtype=np.float32)
        # scaled_values: (n_frames, n_blendshapes_to_process)
        scaled_values = src_values * scaling_factors[None, :]
        ret_face_clip.blendshape_values[:, sensitive_idxs] = scaled_values
        return ret_face_clip
