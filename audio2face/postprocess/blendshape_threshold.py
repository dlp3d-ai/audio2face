from typing import Any

from ..data_structures.face_clip import FaceClip
from ..utils.log import setup_logger
from .base_postprocess import BasePostprocess


class BlendshapeThreshold(BasePostprocess):
    """Blendshape threshold filtering postprocessor.

    Performs threshold filtering on blendshape values in FaceClip, setting
    values to 0 when they are below or equal to the specified threshold.
    Used to remove minor facial expression changes.
    """

    def __init__(
            self,
            name: str,
            thresholds: dict[str, float],
            logger_cfg: None | dict[str, Any] = None):
        """Initialize the BlendshapeThreshold postprocessor.

        Args:
            name (str):
                Name of the postprocessor.
            thresholds (dict[str, float]):
                Mapping dictionary from blendshape names to threshold values.
                Values <= threshold will be set to 0.
            logger_cfg (None | dict[str, Any], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        BasePostprocess.__init__(self, logger_cfg)
        self.name = name
        self.logger_cfg["logger_name"] = name
        self.logger = setup_logger(**self.logger_cfg)
        self.thresholds = thresholds

    def __call__(self, face_clip: FaceClip, **kwargs) -> FaceClip:
        """Perform threshold filtering processing on blendshape values in FaceClip.

        Filters specified blendshape values according to configured thresholds,
        setting values to 0 when they are below or equal to the threshold.
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
        ret_face_clip = face_clip.clone()
        if len(self.thresholds) > 0:
            name_idx_mapping = {
                name: idx for idx, name in enumerate(face_clip.blendshape_names)}
            for name, threshold in self.thresholds.items():
                if name not in name_idx_mapping:
                    self.logger.warning(
                        f'No blendshape named {name} found in input blendshape_names')
                    continue
                # Get all frame data for this blendshape
                bs_idx = name_idx_mapping[name]
                blendshape_values = ret_face_clip.blendshape_values[:, bs_idx]
                # Set values below or equal to threshold to 0
                blendshape_values[blendshape_values <= threshold] = 0
                ret_face_clip.blendshape_values[:, bs_idx] = blendshape_values
        return ret_face_clip
