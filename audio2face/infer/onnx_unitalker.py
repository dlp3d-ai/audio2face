import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import numpy as np
import onnxruntime

from ..data_structures.face_clip import FaceClip
from ..utils.super import Super


class UnitalkerInputTooShort(Exception):
    """Audio input is too short to generate facial expression animation."""
    pass

class UnitalkerInputInvalid(Exception):
    """Audio input is invalid and cannot generate facial expression animation."""
    pass

class OnnxUnitalker(Super):
    """ONNX-based Unitalker inference engine.

    Uses ONNX Runtime for facial expression inference with support for
    multi-threaded concurrent processing. Supports running multiple inference
    sessions simultaneously to improve throughput.
    """

    def __init__(self,
                 model_path: str,
                 blendshape_names: list[str] | str,
                 max_workers: int = 1,
                 thread_pool_executor: ThreadPoolExecutor | None = None,
                 onnx_providers: Literal[
                     "CUDAExecutionProvider",
                     "CPUExecutionProvider"] = "CUDAExecutionProvider",
                 sleep_time: float = 0.01,
                 logger_cfg: None | dict[str, Any] = None) -> None:
        """Initialize the OnnxUnitalker inference engine.

        Args:
            model_path (str):
                Path to the ONNX model file.
            blendshape_names (list[str] | str):
                List of blendshape names or path to JSON file containing
                blendshape names.
            max_workers (int, optional):
                Maximum number of worker threads. Defaults to 1.
            thread_pool_executor (ThreadPoolExecutor | None, optional):
                Thread pool executor. If None, creates a new thread pool
                based on max_workers. Defaults to None.
            onnx_providers (Literal[
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider"], optional):
                ONNX execution provider. Defaults to CUDAExecutionProvider.
            sleep_time (float, optional):
                Wait time for each loop iteration in seconds.
                Defaults to 0.01.
            logger_cfg (None | dict[str, Any], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.

        Raises:
            FileNotFoundError: When model file or blendshape_names file does not exist.
            TypeError: When blendshape_names type is incorrect.
        """
        super().__init__(logger_cfg=logger_cfg)
        if not os.path.exists(model_path):
            msg = f'model_path: {model_path} does not exist'
            self.logger.error(msg)
            raise FileNotFoundError(msg)
        sess_options = onnxruntime.SessionOptions()
        sess_options.log_severity_level = 3
        if thread_pool_executor is None:
            self.thread_pool_executor = ThreadPoolExecutor(max_workers=max_workers)
            self.max_workers = max_workers
        else:
            self.thread_pool_executor = thread_pool_executor
            self.max_workers = 1
        # Create ONNX sessions according to max_workers
        self.unitalker_sessions = [
            onnxruntime.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=[onnx_providers]
            ) for _ in range(self.max_workers)
        ]
        self.session_busy = [False] * self.max_workers
        self.sleep_time = sleep_time
        if isinstance(blendshape_names, str):
            if not os.path.exists(blendshape_names):
                msg = (f'JSON file storing blendshape_names does not exist, '
                       f'path: {blendshape_names}')
                self.logger.error(msg)
                raise FileNotFoundError(msg)
            with open(blendshape_names, encoding='utf-8') as f:
                self.blendshape_names = json.load(f)
        elif isinstance(blendshape_names, list):
            self.blendshape_names = blendshape_names
        else:
            msg = f'blendshape_names: {blendshape_names} type error'
            self.logger.error(msg)
            raise TypeError(msg)

    async def warmup(self) -> None:
        """Warm up the model by processing random audio features to
        initialize ONNX session state.

        Uses randomly generated audio feature data for inference to ensure all
        ONNX sessions are loaded into memory before first use, avoiding delays
        in subsequent inference. Number of warmup coroutines is
        max(5, max_workers * 2).
        """
        coroutines = list()
        n_coroutines = max(5, self.max_workers * 2)
        for _ in range(n_coroutines):
            audio_feature = np.random.randn(1, 8512).astype(np.float32)
            coroutine = self.infer(
                audio_feature,
                emotion_id=10,
                fps=30,
                sample_rate=16000)
            coroutines.append(coroutine)
        await asyncio.gather(*coroutines)


    async def infer(
            self,
            audio_feature: np.ndarray,
            emotion_id: int,
            fps: float,
            sample_rate: int,
            timeout: float = 30) -> FaceClip:
        """Asynchronously infer facial expressions from audio features.

        Uses ONNX model to convert audio features to facial expression animation.
        Supports multi-session concurrent processing with automatic session
        resource management.

        Args:
            audio_feature (np.ndarray):
                Audio feature array with shape (batch_size, time_steps, feature_dim).
            emotion_id (int):
                Emotion ID, 10 represents neutral emotion.
            fps (float):
                Output animation frame rate, typically 30.
            sample_rate (int):
                Audio sample rate, typically 16000.
            timeout (float, optional):
                Timeout for waiting for an idle session in seconds.
                Defaults to 30.

        Returns:
            FaceClip:
                Generated facial expression animation clip containing blendshape
                names and corresponding values.

        Raises:
            TimeoutError: When unable to acquire an idle inference session
                within timeout.
        """
        if len(audio_feature.shape) != 2:
            msg = f'Audio input is invalid, shape: {audio_feature.shape}'
            self.logger.error(msg)
            raise UnitalkerInputInvalid(msg)
        n_frames = int(audio_feature.shape[1] * fps / sample_rate)
        if n_frames < 1:
            msg = (f'Audio input is too short to generate facial expression animation, '
                   f'n_frames: {n_frames}')
            self.logger.error(msg)
            raise UnitalkerInputTooShort(msg)
        session_idx = None
        start_time = time.time()
        while session_idx is None:
            cur_time = time.time()
            if cur_time - start_time > timeout:
                msg = f'No idle ONNX session found within timeout {timeout}s'
                self.logger.error(msg)
                raise TimeoutError(msg)
            for i, busy in enumerate(self.session_busy):
                if not busy:
                    session_idx = i
                    self.session_busy[i] = True
                    break
            await asyncio.sleep(self.sleep_time)
        self.logger.debug(f'Found idle ONNX session: {session_idx}')
        unitalker_session = self.unitalker_sessions[session_idx]
        try:
            time_steps = audio_feature.shape[1] * fps // sample_rate
            ort_inputs = {
                'audio': audio_feature.astype(np.float32),
                'emo_id': np.array([emotion_id], dtype=np.int32),
                'time_steps': np.array([time_steps], dtype=np.int32)
            }
            loop = asyncio.get_event_loop()
            onnx_output = await loop.run_in_executor(
                self.thread_pool_executor,
                unitalker_session.run,
                None,
                ort_inputs
            )
            bs_values = onnx_output[0][0]
            ret_face_clip = FaceClip(
                blendshape_names=self.blendshape_names,
                blendshape_values=bs_values,
                logger_cfg=self.logger_cfg
            )
            return ret_face_clip
        except Exception as e:
            msg = (f'ONNX session: {session_idx} inference failed, '
                   f'using all-zero values, error: {e}')
            self.logger.error(msg)
            bs_values = np.zeros((n_frames, len(self.blendshape_names)))
            ret_face_clip = FaceClip(
                blendshape_names=self.blendshape_names,
                blendshape_values=bs_values,
                logger_cfg=self.logger_cfg
            )
            return ret_face_clip
        finally:
            self.session_busy[session_idx] = False
