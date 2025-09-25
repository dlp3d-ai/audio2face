import asyncio
import json
import os
from typing import Any

import numpy as np

from ..data_structures.face_clip import FaceClip
from ..utils.super import Super


class DummyGeneratorInputTooShort(Exception):
    """Audio input is too short to generate facial expression animation."""
    pass

class DummyGeneratorInputInvalid(Exception):
    """Audio input is invalid and cannot generate facial expression animation."""
    pass

class DummyGenerator(Super):
    """Dummy facial expression generator.

    A dummy facial expression generator for testing and debugging purposes.
    Does not perform actual inference, but returns zero-filled blendshape values.
    Supports asynchronous processing and delay simulation.
    """

    def __init__(self,
                 blendshape_names: list[str] | str,
                 delay_time: float = 0.0,
                 logger_cfg: None | dict[str, Any] = None,
                 **kwargs) -> None:
        """Initialize the dummy facial expression generator.

        Args:
            blendshape_names (list[str] | str):
                List of blendshape names or path to JSON file containing
                blendshape names.
            delay_time (float, optional):
                Time in seconds to simulate processing delay. Defaults to 0.0.
            logger_cfg (None | dict[str, Any], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.

        Raises:
            FileNotFoundError: When blendshape_names file does not exist.
            TypeError: When blendshape_names type is incorrect.
        """
        super().__init__(logger_cfg=logger_cfg)
        self.delay_time = delay_time
        if isinstance(blendshape_names, str):
            if not os.path.exists(blendshape_names):
                msg = (f'JSON file storing blendshape_names does not exist, '
                       f'path: {blendshape_names}')
                self.logger.error(msg)
                raise FileNotFoundError(msg)
            with open(blendshape_names) as f:
                self.blendshape_names = json.load(f)
        elif isinstance(blendshape_names, list):
            self.blendshape_names = blendshape_names
        else:
            msg = f'blendshape_names: {blendshape_names} type error'
            self.logger.error(msg)
            raise TypeError(msg)

    async def warmup(self) -> None:
        """Warm up the model.

        Dummy warmup method that performs no actual operations.
        """
        pass


    async def infer(
            self,
            audio_feature: np.ndarray,
            sample_rate: int,
            fps: float,
            **kwargs
            ) -> FaceClip:
        """Asynchronously infer facial expressions from audio features.

        Dummy facial expression generation method that generates zero-filled
        blendshape values for corresponding frame count based on input audio features.

        Args:
            audio_feature (np.ndarray):
                Audio feature array with shape (batch_size, time_steps, feature_dim).
            sample_rate (int):
                Audio sample rate, typically 16000.
            fps (float):
                Output animation frame rate, typically 30.
            **kwargs:
                Additional optional parameters, currently unused.

        Returns:
            FaceClip:
                Generated facial expression animation clip containing blendshape
                names and corresponding zero-filled values.

        Raises:
            DummyGeneratorInputInvalid: When audio feature array dimensions
                are incorrect.
            DummyGeneratorInputTooShort: When calculated frame count is less than 1.
        """
        if len(audio_feature.shape) != 2:
            msg = f'Audio input is invalid, shape: {audio_feature.shape}'
            self.logger.error(msg)
            raise DummyGeneratorInputInvalid(msg)
        n_frames = int(audio_feature.shape[1] * fps / sample_rate)
        if n_frames < 1:
            msg = (f'Audio input is too short to generate facial expression animation, '
                   f'n_frames: {n_frames}')
            self.logger.error(msg)
            raise DummyGeneratorInputTooShort(msg)
        bs_values = np.zeros((n_frames, len(self.blendshape_names)))
        ret_face_clip = FaceClip(
            blendshape_names=self.blendshape_names,
            blendshape_values=bs_values,
            logger_cfg=self.logger_cfg
        )
        if self.delay_time > 0:
            await asyncio.sleep(self.delay_time)
        return ret_face_clip
