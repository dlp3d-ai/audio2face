import json
import os
from typing import Any

import numpy as np

from ..data_structures.face_clip import FaceClip
from ..utils.log import setup_logger
from .base_postprocess import BasePostprocess


class LinearExpBlend(BasePostprocess):
    """Linear exponential blending postprocessor.

    Performs linear and exponential function blending on specified blendshape
    values, optimizing facial expression effects by adjusting offset,
    normalization reference, exponential strength, and blend weight.
    """

    def __init__(
            self,
            name: str,
            offset: float,
            normalize_reference: float,
            exponential_strength: float,
            blend_weight : float,
            bs_names: list[str] | str,
            logger_cfg: None | dict[str, Any] = None):
        """Initialize the linear exponential blending postprocessor.

        Args:
            name (str):
                Processor name for logging purposes.
            offset (float):
                Value offset for adjusting input value baseline.
            normalize_reference (float):
                Normalization reference value for standardization.
            exponential_strength (float):
                Exponential strength parameter controlling the steepness
                of the exponential function.
            blend_weight (float):
                Blend weight controlling the mixing ratio between linear
                and exponential functions.
            bs_names (list[str] | str):
                List of blendshape names to process. Can be a string list
                or a JSON file path containing the list.
            logger_cfg (None | dict[str, Any], optional):
                Logger configuration dictionary. Defaults to None.
        """
        BasePostprocess.__init__(self, logger_cfg)
        self.name = name
        self.logger_cfg["logger_name"] = name
        self.logger = setup_logger(**self.logger_cfg)
        self.offset = offset
        self.normalize_reference = normalize_reference
        self.exponential_strength = exponential_strength
        self.blend_weight = blend_weight
        if isinstance(bs_names, str):
            if not os.path.exists(bs_names):
                msg = f'Failed to read bs_names, file {bs_names} does not exist.'
                self.logger.error(msg)
                raise FileNotFoundError(msg)
            with open(bs_names, encoding='utf-8') as f:
                self.bs_names = json.load(f)
        else:
            self.bs_names = bs_names

    def __call__(self, face_clip: FaceClip, **kwargs) -> FaceClip:
        """Perform linear exponential blending on specified blendshape values.

        Applies weighted blending transformation of linear and exponential functions
        to specified blendshape values, using offset to adjust baseline and
        normalization reference for standardization to optimize facial expression
        effects.
        Warnings are logged for non-existent blendshape names.

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
        for idx, bs_name in enumerate(src_bs_names):
            if bs_name in self.bs_names:
                sensitive_idxs.append(idx)
        ret_face_clip = face_clip.clone()
        if len(sensitive_idxs) == 0:
            self.logger.warning(
                'No blendshapes found for processing, please check bs_names. ' +
                f'Expected to process {self.bs_names}, but received blendshape_names: '
                f'{src_bs_names}')
            return ret_face_clip
        sensitive_idxs = np.array(sensitive_idxs, dtype=np.int32)
        src_values = ret_face_clip.blendshape_values[:, sensitive_idxs]
        scaled_values = _linear_exp_blend_batch(
            src_values,
            self.offset,
            self.normalize_reference,
            self.exponential_strength,
            self.blend_weight)
        ret_face_clip.blendshape_values[:, sensitive_idxs] = scaled_values
        return ret_face_clip

def _linear_exp_blend_batch(
        x: np.ndarray,
        offset: float,
        normalize_reference: float,
        exponential_strength: float,
        blend_weight: float) -> np.ndarray:
    """Perform linear exponential blending transformation on batch data.

    Transforms input data using weighted blending of linear and exponential
    functions, adjusting baseline with offset and standardizing with
    normalization reference.

    Args:
        x (np.ndarray):
            Input array with shape (n_frames, n_blendshapes).
        offset (float):
            Value offset for adjusting input value baseline.
        normalize_reference (float):
            Normalization reference value for standardization.
        exponential_strength (float):
            Exponential strength parameter controlling the steepness
            of the exponential function.
        blend_weight (float):
            Blend weight controlling the mixing ratio between linear
            and exponential functions.

    Returns:
        np.ndarray:
            Transformed array with same shape as input, maintaining
            consistent data type.
    """
    shifted_x = np.clip(x + offset, 0, 1.0)
    normalized_x = shifted_x / normalize_reference

    linear = normalized_x
    expo = 1 - np.exp(-exponential_strength * normalized_x)

    result = (1 - blend_weight) * linear + blend_weight * expo
    return np.clip(result, 0.0, 1.0)
