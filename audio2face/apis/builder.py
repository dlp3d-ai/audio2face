from .streaming_audio2face_v1 import StreamingAudio2FaceV1

_APIS = dict(
    StreamingAudio2FaceV1=StreamingAudio2FaceV1,
)


def build_api(cfg: dict) -> StreamingAudio2FaceV1:
    """Build an API instance from configuration dictionary.

    Selects the corresponding API class for instantiation based on the type field
    in the configuration dictionary. Currently supports StreamingAudio2FaceV1.

    Args:
        cfg (dict):
            API configuration dictionary, must contain type field to specify
            API type. Other fields will be passed as parameters to the
            corresponding API constructor.

    Returns:
        StreamingAudio2FaceV1:
            API instance built according to configuration.

    Raises:
        TypeError: When the specified API type is not in the supported list.
    """
    cfg = cfg.copy()
    cls_name = cfg.pop('type')
    if cls_name not in _APIS:
        msg = f'Unknown api type: {cls_name}'
        raise TypeError(msg)
    ret_inst = _APIS[cls_name](**cfg)
    return ret_inst

