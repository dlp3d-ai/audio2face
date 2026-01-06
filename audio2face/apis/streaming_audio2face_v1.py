import asyncio
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ..data_structures.face_clip import FaceClip
from ..infer.builder import build_infer
from ..infer.dummy_generator import DummyGeneratorInputTooShort
from ..infer.onnx_unitalker import UnitalkerInputTooShort
from ..postprocess.builder import build_postprocess
from ..postprocess.custom_blink import CustomBlink
from ..postprocess.offset import Offset
from ..split.builder import build_split
from ..utils.super import Super

if TYPE_CHECKING:
    from ..infer.onnx_unitalker import OnnxUnitalker
    from ..infer.torch_feature_extractor import TorchFeatureExtractor


class StreamingAudio2FaceV1ChunkStart(BaseModel):
    """Start chunk for streaming audio to face animation.

    Contains metadata for starting a new streaming audio to face animation request.
    """
    request_id: str
    sample_rate: int
    sample_width: int
    n_channels: int
    callback: Callable[[FaceClip | None], None]
    profile_name: str

class StreamingAudio2FaceV1ChunkBody(BaseModel):
    """Body chunk for streaming audio to face animation.

    Contains audio data chunk for processing in streaming audio to face animation.
    """
    request_id: str
    pcm_bytes: bytes
    offset_name: str | None = None

class StreamingAudio2FaceV1ChunkEnd(BaseModel):
    """End chunk for streaming audio to face animation.

    Signals the end of a streaming audio to face animation request.
    """
    request_id: str


class StreamingAudio2FaceV1(Super):
    """Streaming audio to face animation processor.

    Processes streaming audio data and generates facial expression animations
    in real-time using feature extraction, inference, and post-processing pipelines.
    """

    def __init__(
            self,
            profiles: dict[str, list[str]],
            feature_extractor_cfg: dict[str, Any],
            unitalker_cfg: dict[str, Any],
            split_cfg: dict[str, Any],
            postprocess_cfgs: dict[str, dict[str, Any]],
            split_interval: float = 0.5,
            fps: float = 30,
            max_workers: int = 4,
            thread_pool_executor: ThreadPoolExecutor | None = None,
            maintain_check_interval: float = 30.0,
            request_expire_time: float = 10.0,
            logger_cfg: None | dict[str, Any] = None):
        """Initialize the streaming audio to face animation processor.

        Args:
            profiles (dict[str, list[str]]):
                Mapping of profile names to postprocessing pipeline names.
            feature_extractor_cfg (dict[str, Any]):
                Configuration for the feature extractor.
            unitalker_cfg (dict[str, Any]):
                Configuration for the Unitalker inference model.
            split_cfg (dict[str, Any]):
                Configuration for the audio splitter.
            postprocess_cfgs (dict[str, dict[str, Any]]):
                Configuration for postprocessing pipelines.
            split_interval (float, optional):
                Minimum interval for PCM silence detection in seconds.
                Defaults to 0.5.
            fps (float, optional):
                Output animation frame rate. Defaults to 30.
            max_workers (int, optional):
                Maximum number of worker threads. Defaults to 4.
            thread_pool_executor (ThreadPoolExecutor | None, optional):
                Custom thread pool executor. If None, creates a new one.
                Defaults to None.
            maintain_check_interval (float, optional):
                Interval for checking cache updates in seconds.
                Defaults to 30.0.
            request_expire_time (float, optional):
                Request expiration time in seconds. Defaults to 10.0.
            logger_cfg (None | dict[str, Any], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        Super.__init__(self, logger_cfg)
        self.max_workers = max_workers
        if thread_pool_executor is None:
            self.thread_pool_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        else:
            self.thread_pool_executor = thread_pool_executor
        self.feature_extractor_cfg = feature_extractor_cfg.copy()
        self.feature_extractor_cfg['logger_cfg'] = self.logger_cfg
        self.feature_extractor_cfg['thread_pool_executor'] = self.thread_pool_executor
        self.feature_extractor: TorchFeatureExtractor = build_infer(
            self.feature_extractor_cfg)
        self.unitalker_cfg = unitalker_cfg.copy()
        self.unitalker_cfg['logger_cfg'] = self.logger_cfg
        self.unitalker_cfg['thread_pool_executor'] = self.thread_pool_executor
        self.unitalker: OnnxUnitalker = build_infer(self.unitalker_cfg)
        self.split_cfg = split_cfg.copy()
        self.split_cfg['logger_cfg'] = self.logger_cfg
        self.split = build_split(self.split_cfg)
        self.postprocess_cfgs = postprocess_cfgs.copy()
        for postprocess_cfg in self.postprocess_cfgs.values():
            postprocess_cfg['logger_cfg'] = self.logger_cfg
        self.postprocessors = dict()
        for postprocess_name, postprocess_cfg in self.postprocess_cfgs.items():
            self.postprocessors[postprocess_name] = build_postprocess(postprocess_cfg)
        # Space for recording generation requests
        self.request_space: dict[str, Any] = dict()
        # Minimum interval for PCM silence detection
        self.split_interval = split_interval
        # Frame rate
        self.fps = fps
        # Interval for checking if cache needs updating
        self.maintain_check_interval = maintain_check_interval
        # Request expiration time
        self.request_expire_time = request_expire_time
        # Last time cache update check was performed
        self.last_maintain_check_time = 0.0
        # Mapping of profile names to corresponding postprocess pipelines
        self.profiles = profiles

    async def handle_chunk_start(
            self,
            chunk_start: StreamingAudio2FaceV1ChunkStart) -> None:
        """Handle the start of a streaming audio to face animation request.

        Args:
            chunk_start (StreamingAudio2FaceV1ChunkStart):
                Start chunk containing request metadata and callback function.
        """
        cur_time = time.time()
        await self._maintain_check(cur_time)
        self.request_space[chunk_start.request_id] = dict(
            sample_rate=chunk_start.sample_rate,
            sample_width=chunk_start.sample_width,
            n_channels=chunk_start.n_channels,
            callback=chunk_start.callback,
            profile_name=chunk_start.profile_name,
            last_split_time=0.0,
            last_update_time=cur_time,
            offset_name=None,
            pcm_bytes=b'',
            pcm_duration=0.0
        )

    async def handle_chunk_body(
            self,
            chunk_body: StreamingAudio2FaceV1ChunkBody) -> FaceClip | None:
        """Handle audio data chunk for streaming processing.

        Processes audio data chunks, performs feature extraction, inference,
        and post-processing to generate facial expression animations.

        Args:
            chunk_body (StreamingAudio2FaceV1ChunkBody):
                Body chunk containing audio data and optional offset name.

        Returns:
            FaceClip | None:
                Generated facial expression animation clip, or None if no
                processing occurred (e.g., insufficient audio data).
        """
        cur_time = time.time()
        self.request_space[chunk_body.request_id]['last_update_time'] = cur_time
        duration_before = self.request_space[chunk_body.request_id]['pcm_duration']
        if duration_before > self.split_interval:
            not_before = 0
        else:
            not_before = self.split_interval - duration_before
        new_bytes = chunk_body.pcm_bytes
        sample_rate = self.request_space[chunk_body.request_id]['sample_rate']
        sample_width = self.request_space[chunk_body.request_id]['sample_width']
        n_channels = self.request_space[chunk_body.request_id]['n_channels']
        loop = asyncio.get_event_loop()
        split_points = await loop.run_in_executor(
            self.thread_pool_executor,
            self.split,
            new_bytes,
            sample_rate,
            sample_width,
            n_channels,
            not_before,
            self.split_interval
        )
        if len(split_points) == 0:
            new_duration = duration_before + \
                len(new_bytes) / (sample_rate * sample_width * n_channels)
            self.request_space[chunk_body.request_id]['pcm_bytes'] += new_bytes
            self.request_space[chunk_body.request_id]['pcm_duration'] = new_duration
            return None
        else:
            offset_name = chunk_body.offset_name
            profile_name = \
                self.request_space[chunk_body.request_id]['profile_name']
            if profile_name in self.profiles:
                postprocess_names = self.profiles[profile_name]
            else:
                if len(self.profiles) == 0:
                    msg = 'No profiles found, cannot ' +\
                        'generate facial expression animation.'
                    self.logger.error(msg)
                    raise ValueError(msg)
                default_key = next(iter(self.profiles))
                postprocess_names = self.profiles[default_key]
                self.logger.warning(
                    f'Profile {profile_name} not found, ' +
                    f'using default profile {default_key}')
            old_pcm_bytes = \
                self.request_space[chunk_body.request_id]['pcm_bytes']
            last_split_time = \
                self.request_space[chunk_body.request_id]['last_split_time']
            total_bytes = \
                old_pcm_bytes + new_bytes
            start_idx = 0
            for split_point in split_points:
                end_idx = split_point + len(old_pcm_bytes)
                split_pcm_bytes = total_bytes[start_idx:end_idx]
                split_duration = len(split_pcm_bytes) / \
                    (sample_rate * sample_width * n_channels)
                abs_split_time = last_split_time + split_duration
                try:
                    infer_start_time = time.time()
                    audio_feature = await self.feature_extractor.infer(
                        split_pcm_bytes,
                        sample_rate=sample_rate,
                        sample_width=sample_width,
                        n_channels=n_channels)
                    face_clip = await self.unitalker.infer(
                        audio_feature,
                        emotion_id=10,
                        fps=self.fps,
                        sample_rate=sample_rate)
                    infer_end_time = time.time()
                    infer_duration = infer_end_time - infer_start_time
                    self.logger.debug(
                        f'Request {chunk_body.request_id} progress '
                        f'{abs_split_time:.2f}s, ' +
                        f'segment duration {split_duration:.2f}s, '
                        f'inference time: {infer_duration:.2f}s')
                except (UnitalkerInputTooShort, DummyGeneratorInputTooShort):
                    continue
                for postprocess_name in postprocess_names:
                    postprocess_instance = self.postprocessors[postprocess_name]
                    if isinstance(postprocess_instance, Offset):
                        if offset_name is None:
                            continue
                        else:
                            args = (face_clip, offset_name)
                    elif isinstance(postprocess_instance, CustomBlink):
                        if offset_name is None:
                            blink_name = 'default'
                        else:
                            blink_name = offset_name
                        args = (face_clip, blink_name, chunk_body.request_id)
                    else:
                        args = (face_clip, )
                    face_clip = await loop.run_in_executor(
                        self.thread_pool_executor,
                        postprocess_instance,
                        *args
                    )
                await self.request_space[chunk_body.request_id]['callback'](face_clip)
                start_idx = end_idx
                last_split_time = abs_split_time
            self.request_space[chunk_body.request_id]['last_split_time'] = \
                last_split_time
            self.request_space[chunk_body.request_id]['offset_name'] = offset_name
            if start_idx < len(total_bytes):
                self.request_space[chunk_body.request_id]['pcm_bytes'] = \
                    total_bytes[start_idx:]
                self.request_space[chunk_body.request_id]['pcm_duration'] = \
                    len(self.request_space[chunk_body.request_id]['pcm_bytes']) / \
                    (sample_rate * sample_width * n_channels)
            else:
                self.request_space[chunk_body.request_id]['pcm_bytes'] = b''
                self.request_space[chunk_body.request_id]['pcm_duration'] = 0.0

    async def handle_chunk_end(
            self,
            chunk_end: StreamingAudio2FaceV1ChunkEnd) -> None:
        """Handle the end of a streaming audio to face animation request.

        Processes any remaining audio data and cleans up request resources.

        Args:
            chunk_end (StreamingAudio2FaceV1ChunkEnd):
                End chunk containing request ID for cleanup.
        """
        cur_time = time.time()
        self.request_space[chunk_end.request_id]['last_update_time'] = cur_time
        old_pcm_bytes = \
            self.request_space[chunk_end.request_id]['pcm_bytes']
        if len(old_pcm_bytes) > 0:
            loop = asyncio.get_event_loop()
            profile_name = \
                self.request_space[chunk_end.request_id]['profile_name']
            postprocess_names = self.profiles[profile_name]
            offset_name = \
                self.request_space[chunk_end.request_id]['offset_name']
            last_split_time = \
                self.request_space[chunk_end.request_id]['last_split_time']
            sample_rate = \
                self.request_space[chunk_end.request_id]['sample_rate']
            sample_width = \
                self.request_space[chunk_end.request_id]['sample_width']
            n_channels = \
                self.request_space[chunk_end.request_id]['n_channels']
            split_pcm_bytes = old_pcm_bytes
            split_duration = len(split_pcm_bytes) / \
                (sample_rate * sample_width * n_channels)
            abs_split_time = last_split_time + split_duration
            try:
                infer_start_time = time.time()
                audio_feature = await self.feature_extractor.infer(
                    split_pcm_bytes,
                    sample_rate=sample_rate,
                    sample_width=sample_width,
                    n_channels=n_channels)
                face_clip = await self.unitalker.infer(
                    audio_feature,
                    emotion_id=10,
                    fps=self.fps,
                    sample_rate=sample_rate)
                infer_end_time = time.time()
                infer_duration = infer_end_time - infer_start_time
                self.logger.debug(
                    f'At end, request {chunk_end.request_id} progress '
                    f'{abs_split_time:.2f}s, segment duration {split_duration:.2f}s, '
                    f'inference time: {infer_duration:.2f}s')
                for postprocess_name in postprocess_names:
                    postprocess_instance = self.postprocessors[postprocess_name]
                    if isinstance(postprocess_instance, Offset):
                        if offset_name is None:
                            continue
                        else:
                            args = (face_clip, offset_name)
                    elif isinstance(postprocess_instance, CustomBlink):
                        if offset_name is None:
                            blink_name = 'default'
                        else:
                            blink_name = offset_name
                        args = (face_clip, blink_name, chunk_end.request_id)
                    else:
                        args = (face_clip,)
                    face_clip = await loop.run_in_executor(
                        self.thread_pool_executor,
                        postprocess_instance,
                        *args
                    )
                await self.request_space[chunk_end.request_id]['callback'](face_clip)
            except (UnitalkerInputTooShort, DummyGeneratorInputTooShort):
                msg = (f'Request {chunk_end.request_id} at end, '
                       f'audio data is too short, '
                       'cannot generate facial expression animation.')
                self.logger.warning(msg)
        else:
            self.logger.warning(
                f'Request {chunk_end.request_id} at end has no unprocessed audio data.')
        await self.request_space[chunk_end.request_id]['callback'](None)
        profile_name = \
            self.request_space[chunk_end.request_id]['profile_name']
        postprocess_names = self.profiles[profile_name]
        for postprocess_name in postprocess_names:
            self.postprocessors[postprocess_name].clean_stream(chunk_end.request_id)
        self.request_space.pop(chunk_end.request_id)

    async def _maintain_check(self, cur_time: float | None) -> None:
        """Check if cache needs updating and if there are expired requests.

        Args:
            cur_time (float | None):
                Current time in seconds. If None, uses current system time.
        """
        if cur_time is None:
            cur_time = time.time()
        if cur_time - self.last_maintain_check_time > self.maintain_check_interval:
            self.last_maintain_check_time = cur_time
            # Check if there are expired requests in request_space
            expired_request_ids = []
            for request_id, request_dict in self.request_space.items():
                if cur_time - request_dict['last_update_time'] > \
                        self.request_expire_time:
                    expired_request_ids.append(request_id)
            if len(expired_request_ids) > 0:
                msg = (f'Found {len(expired_request_ids)} expired requests, '
                       f'deleted these requests: {expired_request_ids}')
                self.logger.warning(msg)
                for request_id in expired_request_ids:
                    self.request_space.pop(request_id)
