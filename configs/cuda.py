type = 'FastAPIServer'
# CRITICAL = 50
# ERROR = 40
# WARNING = 30
# INFO = 20
# DEBUG = 10
# NOTSET = 0
__logger_cfg__ = dict(
    logger_name="root",
    file_level=10,
    console_level=20,
    logger_path="logs/server.log"
)
__fps__ = 30
__default_profile__ = [
    'unitalker_clip',
    'arkit_to_mmd',
    'mmd_jaw_open_blend',
    'mouth_scale',
    'mmd_offset',
    'mmd_emotional_blink'
]
response_chunk_n_frames = 10
max_workers = 4
enable_cors = True
host = '0.0.0.0'
port = 18083
logger_cfg = __logger_cfg__
python_api_cfg = dict(
    type='StreamingAudio2FaceV1',
    profiles={
        'KQ-default': __default_profile__,
        'Ani-default': __default_profile__,
        'HT-default': __default_profile__,
        'FNN-default': __default_profile__,
        'KL-default': __default_profile__,
        'NXD-default': __default_profile__
    },
    feature_extractor_cfg=dict(
        type='TorchFeatureExtractor',
        pretrained_path='configs/wavlm-base-plus/config.json'
    ),
    unitalker_cfg=dict(
        type='OnnxUnitalker',
        model_path='weights/unitalker_v0.4.0_base.onnx',
        blendshape_names='configs/unitalker_output_names.json',
        onnx_providers='CUDAExecutionProvider',
        logger_cfg=__logger_cfg__
    ),
    split_cfg=dict(
        type='EnergySplit',
        window_duration=0.01,
    ),
    postprocess_cfgs=dict(
        unitalker_clip=dict(
            type='UnitalkerClip',
        ),
        arkit_to_mmd=dict(
            type='Rename',
            name='rename_arkit_to_mmd',
            bs_names_mapping='configs/mmd_arkit_mapping.json',
        ),
        mmd_jaw_open_blend=dict(
            type='LinearExpBlend',
            name='mmd_jaw_open_blend',
            offset=-0.106,
            normalize_reference=0.75,
            exponential_strength=8,
            blend_weight=0.4,
            bs_names=['ワ'],
        ),
        mouth_scale=dict(
            type='BlendshapeScale',
            name='mouth_scale',
            scaling_factors={
                'あ': 0.2,
                'お': 0.2,
                'あ２': 0.25,
                'ワ': 0.33,
            },
        ),
        mmd_offset=dict(
            type='Offset',
            offset_json_paths=dict(
                anger_1='configs/blendshapes_offset/anger_1.json',
                anger_2='configs/blendshapes_offset/anger_2.json',
                anger_3='configs/blendshapes_offset/anger_3.json',
                anger_4='configs/blendshapes_offset/anger_4.json',
                disgust_1='configs/blendshapes_offset/disgust_1.json',
                disgust_2='configs/blendshapes_offset/disgust_2.json',
                disgust_3='configs/blendshapes_offset/disgust_3.json',
                disgust_4='configs/blendshapes_offset/disgust_4.json',
                fear_1='configs/blendshapes_offset/fear_1.json',
                fear_2='configs/blendshapes_offset/fear_2.json',
                fear_3='configs/blendshapes_offset/fear_3.json',
                fear_4='configs/blendshapes_offset/fear_4.json',
                happiness_1='configs/blendshapes_offset/happiness_1.json',
                happiness_2='configs/blendshapes_offset/happiness_2.json',
                happiness_3='configs/blendshapes_offset/happiness_3.json',
                happiness_4='configs/blendshapes_offset/happiness_4.json',
                sadness_1='configs/blendshapes_offset/sadness_1.json',
                sadness_2='configs/blendshapes_offset/sadness_2.json',
                sadness_3='configs/blendshapes_offset/sadness_3.json',
                sadness_4='configs/blendshapes_offset/sadness_4.json',
                shyness_1='configs/blendshapes_offset/shyness_1.json',
                shyness_2='configs/blendshapes_offset/shyness_2.json',
                shyness_3='configs/blendshapes_offset/shyness_3.json',
                shyness_4='configs/blendshapes_offset/shyness_4.json',
                surprise_1='configs/blendshapes_offset/surprise_1.json',
                surprise_2='configs/blendshapes_offset/surprise_2.json',
                surprise_3='configs/blendshapes_offset/surprise_3.json',
                surprise_4='configs/blendshapes_offset/surprise_4.json',
            ),
        ),
        mmd_emotional_blink=dict(
            type='CustomBlink',
            blink_interval_lowerbound=30,
            blink_interval_upperbound=90,
            default_blink_json_path='configs/blink_anim/neutral.json',
            blink_json_paths=dict(
                anger_1='configs/blink_anim/anger_1.json',
                anger_2='configs/blink_anim/anger_2.json',
                anger_3='configs/blink_anim/anger_3.json',
                anger_4='configs/blink_anim/anger_4.json',
                disgust_1='configs/blink_anim/disgust_1.json',
                disgust_2='configs/blink_anim/disgust_2.json',
                disgust_3='configs/blink_anim/disgust_3.json',
                disgust_4='configs/blink_anim/disgust_4.json',
                fear_1='configs/blink_anim/fear_1.json',
                fear_2='configs/blink_anim/fear_2.json',
                fear_3='configs/blink_anim/fear_3.json',
                fear_4='configs/blink_anim/fear_4.json',
                happiness_1='configs/blink_anim/happiness_1.json',
                happiness_2='configs/blink_anim/happiness_2.json',
                happiness_3='configs/blink_anim/happiness_3.json',
                happiness_4='configs/blink_anim/happiness_4.json',
                sadness_1='configs/blink_anim/sadness_1.json',
                sadness_2='configs/blink_anim/sadness_2.json',
                sadness_3='configs/blink_anim/sadness_3.json',
                sadness_4='configs/blink_anim/sadness_4.json',
                shyness_1='configs/blink_anim/shyness_1.json',
                shyness_2='configs/blink_anim/shyness_2.json',
                shyness_3='configs/blink_anim/shyness_3.json',
                shyness_4='configs/blink_anim/shyness_4.json',
                surprise_1='configs/blink_anim/surprise_1.json',
                surprise_2='configs/blink_anim/surprise_2.json',
                surprise_3='configs/blink_anim/surprise_3.json',
                surprise_4='configs/blink_anim/surprise_4.json',
            ),
        )
    ),
    fps=__fps__,
)
