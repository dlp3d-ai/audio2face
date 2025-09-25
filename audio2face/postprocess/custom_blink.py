import json
import time
from typing import Any

import numpy as np

from ..data_structures.face_clip import FaceClip
from .base_postprocess import BasePostprocess


class CustomBlink(BasePostprocess):
    """Custom blink animation postprocessor.

    Automatically adds blink effects to facial expression animations based on
    specified blink animation JSON files. Supports multiple blink animation
    styles, configurable blink interval ranges, and random blink timing generation.
    Supports streaming processing with stream_id to maintain blink state across
    different streams.
    """

    def __init__(
            self,
            default_blink_json_path: str,
            blink_json_paths: dict[str, str],
            blink_interval_lowerbound: int = 60,
            blink_interval_upperbound: int = 120,
            maintain_check_interval: float = 30.0,
            expire_time: float = 120.0,
            logger_cfg: None | dict[str, Any] = None):
        """Initialize the custom blink animation postprocessor.

        Args:
            default_blink_json_path (str):
                Default blink animation JSON file path.
            blink_json_paths (dict[str, str]):
                Custom blink animation JSON file paths dictionary, where keys
                are blink animation names and values are corresponding JSON file paths.
            blink_interval_lowerbound (int, optional):
                Minimum frame count for blink intervals. Defaults to 60.
            blink_interval_upperbound (int, optional):
                Maximum frame count for blink intervals. Defaults to 120.
            maintain_check_interval (float, optional):
                Interval time in seconds for checking if cache needs updating.
                Defaults to 30.0.
            expire_time (float, optional):
                Expiration time in seconds for stream cache. Defaults to 120.0.
            logger_cfg (None | dict[str, Any], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        BasePostprocess.__init__(self, logger_cfg)
        self.blink_json_paths = blink_json_paths
        if 'default' in self.blink_json_paths:
            msg = ('default keyword already exists in blink_json_paths, ' +
                   'will override the corresponding value.')
            self.logger.warning(msg)
        self.blink_json_paths['default'] = default_blink_json_path
        self.blink_animation = dict()
        for name, blink_json_path in self.blink_json_paths.items():
            self.blink_animation[name] = dict()
            with open(blink_json_path, encoding='utf-8') as f:
                json_dict = json.load(f)
            bs_values = np.array(json_dict['blendshape_values'])
            n_frames = bs_values.shape[0]
            self.blink_animation[name]['n_frames'] = n_frames
            for bs_idx, bs_name in enumerate(json_dict['blendshape_names']):
                self.blink_animation[name][bs_name] = bs_values[:, bs_idx]
        self.blink_interval_lowerbound = blink_interval_lowerbound
        self.blink_interval_upperbound = blink_interval_upperbound
        self.stream_cache = dict()
        self.maintain_check_interval = maintain_check_interval
        self.expire_time = expire_time
        self.last_maintain_check_time = 0.0

    def __call__(
            self,
            face_clip: FaceClip,
            blink_name: str,
            stream_id: str | None = None,
            **kwargs) -> FaceClip:
        """Add blink effects to facial expression animation.

        Adds blink effects at random positions in facial expression animation
        based on specified blink animation name. If the specified blink animation
        does not exist, the default blink animation will be used. Supports
        streaming processing with stream_id to maintain blink state continuity
        across different streams.

        Args:
            face_clip (FaceClip):
                Input facial expression animation clip.
            blink_name (str):
                Blink animation name, must exist in blink_json_paths provided
                during initialization. If not found, default blink animation
                will be used.
            stream_id (str | None, optional):
                Stream ID for maintaining blink state continuity in streaming
                processing. Defaults to None for non-streaming processing.
            **kwargs:
                Additional keyword arguments, currently unused.

        Returns:
            FaceClip:
                Facial expression animation clip with blink effects added.
        """
        cur_time = time.time()
        self._maintain_stream_cache(cur_time)
        ret_face_clip = face_clip.clone()
        if blink_name not in self.blink_animation:
            msg = (f'blink_name {blink_name} does not exist, '
                   f'will use default blink animation.')
            self.logger.warning(msg)
            blink_animation = self.blink_animation['default']
        else:
            blink_animation = self.blink_animation[blink_name]
        blink_n_frames = blink_animation['n_frames']
        if stream_id is None:
            base_frame_idx = 0
        else:
            if stream_id not in self.stream_cache:
                self.stream_cache[stream_id] = dict(
                    last_update_time=cur_time,
                    last_blink_end_frame=0,
                )
            else:
                self.stream_cache[stream_id]['last_update_time'] = cur_time
            base_frame_idx = self.stream_cache[stream_id]['last_blink_end_frame']
        blink_start_frames = list()
        cursor = base_frame_idx
        while cursor < len(face_clip):
            if cursor + self.blink_interval_upperbound < 0:
                blink_start_frame = 0
            else:
                blink_interval = np.random.randint(
                    self.blink_interval_lowerbound,
                    self.blink_interval_upperbound)
                blink_start_frame = cursor + blink_interval
            blink_end_frame = blink_start_frame + blink_n_frames
            if blink_start_frame >= 0 and blink_end_frame < len(face_clip):
                blink_start_frames.append(blink_start_frame)
            cursor = blink_end_frame
        if stream_id is not None:
            if len(blink_start_frames) > 0:
                last_blink_end_frame = blink_start_frames[-1] + blink_n_frames
                last_blink_end_frame = last_blink_end_frame - len(face_clip)
                self.stream_cache[stream_id]['last_blink_end_frame'] = \
                    last_blink_end_frame
            else:
                self.stream_cache[stream_id]['last_blink_end_frame'] -= \
                    len(face_clip)
        edit_count = 0
        for bs_idx, bs_name in enumerate(face_clip.blendshape_names):
            if bs_name in blink_animation:
                for blink_start_frame in blink_start_frames:
                    ret_face_clip.blendshape_values[
                        blink_start_frame:blink_start_frame + blink_n_frames,
                        bs_idx] = blink_animation[bs_name]
                edit_count += 1
        if edit_count == 0:
            self.logger.warning(
                'No matching blendshape names found, no edits performed. ' +
                f'Input blendshape_names: {face_clip.blendshape_names}.')
        else:
            self.logger.debug(
                f'Edited {edit_count} blendshapes based on {blink_name}, ' +
                f'total {len(blink_start_frames)} blinks, '
                f'each blink lasts {blink_n_frames} frames.')
        return ret_face_clip

    def _maintain_stream_cache(self, cur_time: float) -> None:
        """Maintain stream cache and clean up expired stream data.

        Checks and cleans up stream cache data that exceeds expiration time
        to prevent memory leaks.

        Args:
            cur_time (float):
                Current timestamp.
        """
        if cur_time - self.last_maintain_check_time <= self.maintain_check_interval:
            return
        self.last_maintain_check_time = cur_time
        expired_stream_ids = list()
        for stream_id, stream_info in self.stream_cache.items():
            if cur_time - stream_info['last_update_time'] > self.expire_time:
                expired_stream_ids.append(stream_id)
        if len(expired_stream_ids) > 0:
            self.logger.warning(
                f'Found {len(expired_stream_ids)} expired streams: ' +
                f'{expired_stream_ids}, removed from cache.'
            )
            for stream_id in expired_stream_ids:
                self.clean_stream(stream_id)

    def clean_stream(self, stream_id: str) -> None:
        """Clean up cache data for specified stream.

        Removes specified stream ID and its related data from stream cache.

        Args:
            stream_id (str):
                Stream ID to clean up.
        """
        if stream_id in self.stream_cache:
            self.stream_cache.pop(stream_id)
