import json
import os
from typing import Any

import numpy as np

from ..data_structures.face_clip import FaceClip
from ..utils.log import setup_logger
from .base_postprocess import BasePostprocess


class Rename(BasePostprocess):
    """Blendshape renaming postprocessor.

    Reorganizes blendshape names and data in FaceClip based on mapping
    relationships. Supports loading mapping configurations from files or
    dictionaries for blendshape name conversion between different systems.
    """

    def __init__(
            self,
            name: str,
            bs_names_mapping: dict[str, str] | str,
            logger_cfg: None | dict[str, Any] = None):
        """Initialize the Rename postprocessor.

        Args:
            name (str):
                Name of the postprocessor.
            bs_names_mapping (dict[str, str] | str):
                Blendshape name mapping configuration, can be a dictionary
                or JSON file path. Dictionary format is {target_name: source_name}.
            logger_cfg (None | dict[str, Any], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.

        Raises:
            FileNotFoundError:
                When bs_names_mapping is a file path but the file does not exist.
            ValueError:
                When bs_names_mapping is empty.
        """
        BasePostprocess.__init__(self, logger_cfg)
        self.name = name
        self.logger_cfg["logger_name"] = name
        self.logger = setup_logger(**self.logger_cfg)
        if isinstance(bs_names_mapping, str):
            if not os.path.exists(bs_names_mapping):
                msg = (f'Failed to read bs_names_mapping, file {bs_names_mapping} '
                       'does not exist.')
                self.logger.error(msg)
                raise FileNotFoundError(msg)
            with open(bs_names_mapping, encoding='utf-8') as f:
                self.bs_names_mapping = json.load(f)
        else:
            self.bs_names_mapping = bs_names_mapping
        if len(self.bs_names_mapping) == 0:
            msg = ('bs_names_mapping is empty, please check if the provided '
                   'bs_names_mapping is correct.')
            self.logger.error(msg)
            raise ValueError(msg)
        # mapping from src name to dst index
        self.name_idx_mapping = {
            name: idx for idx, name in enumerate(self.bs_names_mapping.keys())}

    def __call__(self, face_clip: FaceClip, **kwargs) -> FaceClip:
        """Rename and reorganize blendshapes in FaceClip.

        Reorganizes blendshape names and data according to configured mapping
        relationships, creating a new FaceClip instance. For source blendshapes
        not present in the mapping, corresponding positions are filled with 0.

        Args:
            face_clip (FaceClip):
                Input facial expression animation clip.
            **kwargs:
                Additional keyword arguments.

        Returns:
            FaceClip:
                Renamed facial expression animation clip containing new
                blendshape names and data.
        """
        n_frames = face_clip.blendshape_values.shape[0]
        dst_n_bs_names = len(self.bs_names_mapping)
        dst_bs_values = np.zeros((n_frames, dst_n_bs_names), dtype=face_clip.dtype)
        src_name_idx_mapping = {
            name: idx for idx, name in enumerate(face_clip.blendshape_names)}
        rename_count = 0
        for dst_idx, dst_name in enumerate(self.bs_names_mapping.keys()):
            src_name = self.bs_names_mapping[dst_name]
            if src_name is not None:
                if src_name in face_clip.blendshape_names:
                    src_idx = src_name_idx_mapping[src_name]
                    dst_bs_values[:, dst_idx] = \
                        face_clip.blendshape_values[:, src_idx]
                    rename_count += 1
        if rename_count == 0:
            msg = ('No blendshapes found for renaming, ' +
                   'please check if bs_names_mapping is correct. ' +
                   f'Expected mapping relationships: {self.bs_names_mapping}')
            self.logger.warning(msg)
        ret_face_clip = FaceClip(
            blendshape_names=list(self.bs_names_mapping.keys()),
            blendshape_values=dst_bs_values,
            dtype=face_clip.dtype,
            timeline_start_idx=face_clip.timeline_start_idx,
            logger_cfg=face_clip.logger_cfg
        )
        return ret_face_clip
