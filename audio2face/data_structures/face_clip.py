import io
from typing import Any

import numpy as np

from ..utils.io import load_npz
from ..utils.log import get_logger
from ..utils.super import Super


class FaceClip(Super):
    """Class for storing and passing values of each blendshape in animation.

    Used for managing facial expression animation data, containing blendshape
    name list and corresponding value matrix. Supports cloning, slicing,
    format conversion and other operations.
    """
    XRMOGEN_NPZ_KEYS_TO_EXCLUDE = ('Timecode', 'BlendShapeCount', 'n_frames')

    def __init__(self,
                 blendshape_names: list[str],
                 blendshape_values: np.ndarray,
                 dtype: np.dtype = np.float16,
                 timeline_start_idx: int | None = None,
                 logger_cfg: None | dict[str, Any] = None) -> None:
        """Initialize FaceClip instance.

        Args:
            blendshape_names (list[str]):
                List of facial expression shapekey names.
            blendshape_values (np.ndarray):
                Facial expression value array with shape (n_frames, n_blendshapes).
            dtype (np.dtype, optional):
                Data type for facial expression values. Defaults to np.float16.
            timeline_start_idx (int | None, optional):
                Starting index of animation clip in timeline, only used when
                alignment with other channels on timeline is needed. In most
                cases, pass None. Defaults to None.
            logger_cfg (None | dict[str, Any], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        super().__init__(logger_cfg=logger_cfg)
        self.blendshape_names = blendshape_names
        self.blendshape_values = blendshape_values
        self.dtype = dtype
        self.blendshape_values = self.blendshape_values.astype(self.dtype)
        self._check_n_bs()
        self.timeline_start_idx = timeline_start_idx

    def _check_n_bs(self) -> None:
        """Check if blendshape_names length matches blendshape_values column count.

        Raises ValueError if they do not match.
        """
        n_bs_np = self.blendshape_values.shape[1]
        n_bs_names = len(self.blendshape_names)
        if n_bs_np != n_bs_names:
            msg = (f'Length of blendshape_names {n_bs_names} != '
                   f'blendshape_values.shape[1] value {n_bs_np}')
            self.logger.error(msg)
            raise ValueError(msg)

    def __len__(self) -> int:
        """Return the number of frames in the animation.

        Returns:
            int: Number of frames in the animation.
        """
        return len(self.blendshape_values)

    def set_timeline_start_idx(self, timeline_start_idx: int | None) -> None:
        """Set timeline starting index.

        Args:
            timeline_start_idx (int | None):
                New timeline starting index.
        """
        self.timeline_start_idx = timeline_start_idx

    def set_blendshape_values(self, blendshape_values: np.ndarray) -> None:
        """Set facial expression value array.

        Args:
            blendshape_values (np.ndarray):
                New facial expression value array with shape (n_frames, n_blendshapes).
        """
        self.blendshape_values = blendshape_values
        self._check_n_bs()

    def clone(self) -> 'FaceClip':
        """Create a deep clone of the current FaceClip.

        Returns:
            FaceClip: Clone of the current facial expression clip.
        """
        return FaceClip(
            blendshape_names=self.blendshape_names.copy(),
            blendshape_values=self.blendshape_values.copy(),
            dtype=self.dtype,
            timeline_start_idx=self.timeline_start_idx,
            logger_cfg=self.logger_cfg)

    def slice(self, start_frame: int, end_frame: int) -> 'FaceClip':
        """Create a slice of the facial expression clip.

        Args:
            start_frame (int):
                Starting frame of the slice (inclusive).
            end_frame (int):
                Ending frame of the slice (exclusive).

        Returns:
            FaceClip: Slice of the current facial expression clip with range
                [start_frame, end_frame).
        """
        if self.timeline_start_idx is not None and start_frame == 0:
            timeline_start_idx = self.timeline_start_idx + start_frame
        else:
            timeline_start_idx = None
        return FaceClip(
            blendshape_names=self.blendshape_names.copy(),
            blendshape_values=self.blendshape_values[start_frame:end_frame].copy(),
            dtype=self.dtype,
            timeline_start_idx=timeline_start_idx,
            logger_cfg=self.logger_cfg)

    def to_xrmogen_dict(self) -> dict[str, Any]:
        """Convert current facial expression clip to XRMogen format dictionary.

        Returns:
            dict[str, Any]:
                Dictionary representation of the current facial expression clip,
                containing n_frames, BlendShapeCount,
                Timecode and data for each blendshape.
        """
        ret_dict = dict()
        ret_dict['n_frames'] = len(self)
        for bs_idx, bs_name in enumerate(self.blendshape_names):
            ret_dict[bs_name] = self.blendshape_values[:, bs_idx]
        ret_dict['BlendShapeCount'] = len(self.blendshape_names)
        ret_dict['Timecode'] = list(range(len(self)))
        return ret_dict

    def to_xrmogen_npz(self) -> io.BytesIO:
        """Convert current facial expression clip to compressed npz file.

        Returns:
            io.BytesIO: Compressed npz file containing facial expression data.
        """
        npz_dict = self.to_xrmogen_dict()
        ret_io = io.BytesIO()
        np.savez_compressed(ret_io, **npz_dict)
        ret_io.seek(0)
        return ret_io

    @classmethod
    def from_xrmogen_dict(
            cls,
            face_dict: dict[str, Any],
            dtype: np.dtype = np.float16,
            logger_cfg: None | dict[str, Any] = None) -> 'FaceClip':
        """Construct a facial expression clip from XRMogen format dictionary.

        Args:
            face_dict (dict[str, Any]):
                Dictionary representation of the facial expression clip.
            dtype (np.dtype, optional):
                Data type for facial expression values. Defaults to np.float16.
            logger_cfg (None | dict[str, Any], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.

        Returns:
            FaceClip: Facial expression clip constructed from dictionary.
        """
        blendshape_names = [key for key in face_dict.keys()
                           if key not in cls.XRMOGEN_NPZ_KEYS_TO_EXCLUDE]
        n_frames = face_dict['n_frames']
        blendshape_values = np.zeros((n_frames, len(blendshape_names)),
                                     dtype=dtype)
        for bs_idx, bs_name in enumerate(blendshape_names):
            blendshape_values[:, bs_idx] = face_dict[bs_name]
        return FaceClip(
            blendshape_names=blendshape_names,
            blendshape_values=blendshape_values,
            dtype=dtype,
            logger_cfg=logger_cfg)

    @classmethod
    def from_xrmogen_npz(
            cls,
            npz_file: io.BytesIO | str,
            dtype: np.dtype = np.float16,
            logger_cfg: None | dict[str, Any] = None) -> 'FaceClip':
        """Construct a facial expression clip from compressed npz file.

        Args:
            npz_file (io.BytesIO | str):
                Path to compressed npz file or file object.
            dtype (np.dtype, optional):
                Data type for facial expression values. Defaults to np.float16.
            logger_cfg (None | dict[str, Any], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.

        Returns:
            FaceClip: Facial expression clip constructed from compressed npz file.
        """
        npz_dict = load_npz(npz_file)
        return cls.from_xrmogen_dict(npz_dict, dtype=dtype, logger_cfg=logger_cfg)

    @classmethod
    def concat(cls,
               face_clips: list['FaceClip']) -> 'FaceClip':
        """Concatenate multiple facial expression clip lists.

        Concatenates multiple FaceClips in chronological order into a new FaceClip.
        All clips must have the same blendshape_names.

        Args:
            face_clips (list[FaceClip]):
                List of facial expression clips to concatenate.

        Returns:
            FaceClip: Concatenated facial expression clip.

        Raises:
            ValueError:
                When face_clips is empty list or contains different blendshape_names.
        """
        logger = get_logger(cls.__name__)
        if len(face_clips) <= 0:
            msg = 'FaceClip.concat() parameter face_clips is empty list.'
            logger.error(msg)
            raise ValueError(msg)
        logger = face_clips[0].logger
        n_frames = sum([len(fc) for fc in face_clips])
        blendshape_names = face_clips[0].blendshape_names
        blendshape_values = face_clips[0].blendshape_values
        timeline_start_idx = face_clips[0].timeline_start_idx
        # Check restpose name and app name
        for idx in range(1, len(face_clips)):
            cur_blendshape_names = face_clips[idx].blendshape_names
            if cur_blendshape_names != blendshape_names:
                msg = (f'FaceClip.concat() parameter face_clips contains '
                       f'different blendshape_names.\n'
                       f'Clip0: {blendshape_names}\n'
                       f'Clip{idx}: {cur_blendshape_names}')
                logger.error(msg)
                raise ValueError(msg)
        blendshape_values = np.zeros((n_frames, len(blendshape_names)))
        clip_start_frame = 0
        for idx in range(0, len(face_clips)):
            cur_blendshape_values = face_clips[idx].blendshape_values
            cur_n_frames = len(face_clips[idx])
            blendshape_values[clip_start_frame:clip_start_frame +
                              cur_n_frames, :] = cur_blendshape_values
            clip_start_frame += cur_n_frames
        return FaceClip(
            blendshape_names=blendshape_names,
            blendshape_values=blendshape_values,
            dtype=face_clips[0].dtype,
            timeline_start_idx=timeline_start_idx,
            logger_cfg=face_clips[0].logger_cfg)

