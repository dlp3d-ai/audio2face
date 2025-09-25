import json
from typing import Any

from ..data_structures.face_clip import FaceClip
from .base_postprocess import BasePostprocess


class Offset(BasePostprocess):
    """Blendshape offset postprocessor.

    Performs offset adjustment on blendshape values in FaceClip based on
    predefined offset configurations. Supports multiple offset configurations
    with different offset schemes selectable via offset_name.
    """

    def __init__(
            self,
            offset_json_paths: dict[str, str],
            logger_cfg: None | dict[str, Any] = None):
        """Initialize the Offset postprocessor.

        Args:
            offset_json_paths (dict[str, str]):
                Mapping dictionary from offset configuration names to JSON file paths.
            logger_cfg (None | dict[str, Any], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.

        Raises:
            FileNotFoundError:
                When the specified JSON file does not exist.
            json.JSONDecodeError:
                When the JSON file format is incorrect.
        """
        BasePostprocess.__init__(self, logger_cfg)
        self.offset_json_paths = offset_json_paths
        self.offset_dicts = dict()
        for name, offset_json_path in self.offset_json_paths.items():
            with open(offset_json_path) as f:
                self.offset_dicts[name] = json.load(f)

    def __call__(self, face_clip: FaceClip, offset_name: str, **kwargs) -> FaceClip:
        """Perform offset processing on blendshape values in FaceClip.

        Performs offset adjustment on blendshape values according to specified
        offset configuration, ensuring result values are within [0, 1] range.
        Warnings are logged for non-existent offset configurations.

        Args:
            face_clip (FaceClip):
                Input facial expression animation clip.
            offset_name (str):
                Name of the offset configuration to use.
            **kwargs:
                Additional keyword arguments.

        Returns:
            FaceClip:
                Processed facial expression animation clip. Original data
                is not modified.
        """
        ret_face_clip = face_clip.clone()
        if offset_name not in self.offset_dicts:
            msg = f'offset_name {offset_name} does not exist, no edits performed.'
            self.logger.warning(msg)
            return ret_face_clip
        selected_offset = self.offset_dicts[offset_name]
        edit_count = 0
        for bs_idx, bs_name in enumerate(face_clip.blendshape_names):
            if bs_name in selected_offset:
                ret_face_clip.blendshape_values[:, bs_idx] += selected_offset[bs_name]
                edit_count += 1
        if edit_count == 0:
            self.logger.warning(
                'No matching blendshape names found, no edits performed. ' +
                f'Input blendshape_names: {face_clip.blendshape_names}.')
        else:
            self.logger.debug(
                f'Edited {edit_count} blendshapes based on {offset_name}.')
        return ret_face_clip

