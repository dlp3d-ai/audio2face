import logging
import os
import uuid

import numpy as np
import pytest

from audio2face.apis.builder import build_api
from audio2face.apis.streaming_audio2face_v1 import (
    StreamingAudio2FaceV1ChunkBody,
    StreamingAudio2FaceV1ChunkEnd,
    StreamingAudio2FaceV1ChunkStart,
)
from audio2face.data_structures.face_clip import FaceClip
from audio2face.utils.log import setup_logger

LOGGER_CFG = dict(
    logger_name='test_streaming_audio2face_v1',
    file_level=logging.DEBUG,
    console_level=logging.INFO,
    logger_path='logs/pytest.log',
)
FEATURE_EXTRACTOR_PRETRAINED_PATH = 'configs/wavlm-base-plus/config.json'
ONNX_UNITALKER_MODEL_PATH = 'weights/unitalker_v0.4.0_base.onnx'
UNITALKER_BLENDSHAPE_NAMES_PATH = 'configs/unitalker_output_names.json'
FPS = 30
MAX_WORKERS = 4

async def log_callback(face_clip: FaceClip | None):
    """Log callback function for streaming audio2face processing.

    This callback function is called during streaming audio2face processing
    to log information about generated face clips or completion status.

    Args:
        face_clip (FaceClip | None): Generated face clip data, or None
            if processing is finished.
    """
    logger = setup_logger(**LOGGER_CFG)
    if face_clip is None:
        logger.info('Streaming audio2face v1 finished')
        return
    n_frames = len(face_clip)
    n_blendshapes = len(face_clip.blendshape_names)
    logger.info(
        f'Generated a face clip with {n_frames} frames, {n_blendshapes} blendshapes')

@pytest.mark.asyncio
async def test_streaming_audio2face_v1():
    """Test streaming audio2face v1 API functionality.

    This test verifies the complete streaming audio2face pipeline by:
    - Building the API with various postprocessing configurations
    - Generating random audio data with different durations
    - Simulating streaming audio chunks with silent periods
    - Testing emotion offset and blink functionality
    - Verifying the complete processing pipeline
    """
    if not os.path.exists('logs'):
        os.makedirs('logs')
    api_cfg = dict(
        type='StreamingAudio2FaceV1',
        profiles={
            'KQ-default': [
                'unitalker_clip',
                'arkit_to_mmd',
                'linear_exp_blend',
                'emotion_offset',
                'emotion_blink']
        },
        feature_extractor_cfg=dict(
            type='TorchFeatureExtractor',
            pretrained_path=FEATURE_EXTRACTOR_PRETRAINED_PATH
        ),
        unitalker_cfg=dict(
            type='OnnxUnitalker',
            model_path=ONNX_UNITALKER_MODEL_PATH,
            blendshape_names=UNITALKER_BLENDSHAPE_NAMES_PATH,
            onnx_providers='CPUExecutionProvider',
            logger_cfg=LOGGER_CFG
        ),
        split_cfg=dict(
            type='EnergySplit',
            window_duration=0.01,
        ),
        postprocess_cfgs=dict(
            unitalker_clip=dict(
                type='UnitalkerClip',
            ),
            unitalker_random_blink=dict(
                type='UnitalkerRandomBlink',
                fps=FPS,
            ),
            arkit_to_mmd=dict(
                type='Rename',
                name='rename_arkit_to_mmd',
                bs_names_mapping='configs/mmd_arkit_mapping.json',
            ),
            linear_exp_blend=dict(
                type='LinearExpBlend',
                name='mmd_jaw_open_blend',
                offset=-0.106,
                normalize_reference=0.35,
                exponential_strength=4,
                blend_weight=0.3,
                bs_names=['ワ'],
            ),
            emotion_offset=dict(
                type='Offset',
                offset_json_paths=dict(
                    anger_1='configs/blendshapes_offset/anger_1.json',
                    disgust_1='configs/blendshapes_offset/disgust_1.json',
                ),
            ),
            emotion_blink=dict(
                type='CustomBlink',
                default_blink_json_path='configs/blink_anim/neutral.json',
                blink_json_paths=dict(
                    anger_1='configs/blink_anim/anger_1.json',
                    disgust_1='configs/blink_anim/disgust_1.json',
                ),
            )
        ),
        fps=FPS,
        max_workers=MAX_WORKERS,
        logger_cfg=LOGGER_CFG
    )
    api = build_api(api_cfg)
    valid_durations = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    sample_rate = 16000
    sample_width = 2
    n_channels = 1
    for duration in valid_durations:
        # Generate a random int16 PCM audio with duration
        audio = np.random.randint(
            -32768, 32767, int(duration * sample_rate), dtype=np.int16)
        # Set silent every 1s for 0.02s (320 samples at 16kHz)
        silent_duration_samples = int(0.02 * sample_rate)  # 320 samples
        for i in range(sample_rate, len(audio), sample_rate):
            if (i + silent_duration_samples) < len(audio):
                audio[i:i + silent_duration_samples] = 0
        # Convert audio to bytes
        audio_bytes = audio.tobytes()
        # Generate a random request_id
        request_id = str(uuid.uuid4())
        chunk_start = StreamingAudio2FaceV1ChunkStart(
            request_id=request_id,
            sample_rate=sample_rate,
            sample_width=sample_width,
            n_channels=n_channels,
            callback=log_callback,
            profile_name='KQ-default',
        )
        await api.handle_chunk_start(chunk_start)
        # Feed chunk body every 0.04s
        step = int(0.04 * sample_rate * sample_width * n_channels)
        for start_idx in range(
                0, len(audio_bytes), step):
            end_idx = start_idx + step
            end_idx = min(end_idx, len(audio_bytes))
            # Randomly select offset_name from ('anger_1', 'disgust_1', None)
            offset_name = np.random.choice(['anger_1', 'disgust_1', None])
            chunk_body = StreamingAudio2FaceV1ChunkBody(
                request_id=request_id,
                pcm_bytes=audio_bytes[start_idx:end_idx],
                offset_name=offset_name
            )
            await api.handle_chunk_body(chunk_body)
        # Feed chunk end
        chunk_end = StreamingAudio2FaceV1ChunkEnd(
            request_id=request_id
        )
        await api.handle_chunk_end(chunk_end)
