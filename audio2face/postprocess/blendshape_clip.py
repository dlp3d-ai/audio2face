from typing import Any

import numpy as np

from ..data_structures.face_clip import FaceClip
from ..utils.log import setup_logger
from .base_postprocess import BasePostprocess


class BlendshapeClip(BasePostprocess):
    """Blendshape value clipping postprocessor.

    Performs upper and lower bound clipping on blendshape values in FaceClip,
    ensuring values are within specified ranges. Supports setting different
    upper and lower bound thresholds for different blendshapes.
    """

    def __init__(
            self,
            name: str,
            lowerbounds: dict[str, float] | None = None,
            upperbounds: dict[str, float] | None = None,
            logger_cfg: None | dict[str, Any] = None):
        """Initialize the BlendshapeClip postprocessor.

        Args:
            name (str):
                Name of the postprocessor.
            lowerbounds (dict[str, float] | None, optional):
                Mapping dictionary from blendshape names to lower bound values.
                Defaults to None (empty dictionary).
            upperbounds (dict[str, float] | None, optional):
                Mapping dictionary from blendshape names to upper bound values.
                Defaults to None (empty dictionary).
            logger_cfg (None | dict[str, Any], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        BasePostprocess.__init__(self, logger_cfg)
        self.name = name
        self.logger_cfg["logger_name"] = name
        self.logger = setup_logger(**self.logger_cfg)
        if lowerbounds is None:
            self.lowerbounds = dict()
        else:
            self.lowerbounds = lowerbounds
        if upperbounds is None:
            self.upperbounds = dict()
        else:
            self.upperbounds = upperbounds

    def __call__(self, face_clip: FaceClip, **kwargs) -> FaceClip:
        """Perform clipping processing on blendshape values in FaceClip.

        Clips specified blendshape values according to configured upper and
        lower bounds, ensuring values are within reasonable ranges. For
        non-existent blendshape names, warnings are logged and processing is skipped.

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
        ret_face_clip = face_clip.clone()
        if len(self.lowerbounds) + len(self.upperbounds) > 0:
            name_idx_mapping = {
                name: idx for idx, name in enumerate(face_clip.blendshape_names)}
            for name, lowerbound in self.lowerbounds.items():
                if name not in name_idx_mapping:
                    self.logger.warning(
                        f'No blendshape named {name} found in input blendshape_names')
                    continue
                ret_face_clip.blendshape_values[:, name_idx_mapping[name]] = \
                    np.clip(
                        ret_face_clip.blendshape_values[:, name_idx_mapping[name]],
                        a_min=lowerbound, a_max=None)
            for name, upperbound in self.upperbounds.items():
                if name not in name_idx_mapping:
                    self.logger.warning(
                        f'No blendshape named {name} found in input blendshape_names')
                    continue
                ret_face_clip.blendshape_values[:, name_idx_mapping[name]] = \
                    np.clip(
                        ret_face_clip.blendshape_values[:, name_idx_mapping[name]],
                        a_min=None, a_max=upperbound)
        return ret_face_clip
