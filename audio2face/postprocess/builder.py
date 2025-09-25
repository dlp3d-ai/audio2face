from .base_postprocess import BasePostprocess
from .blendshape_clip import BlendshapeClip
from .blendshape_scale import BlendshapeScale
from .blendshape_threshold import BlendshapeThreshold
from .custom_blink import CustomBlink
from .linear_exp_blend import LinearExpBlend
from .offset import Offset
from .rename import Rename
from .unitalker_clip import UnitalkerClip
from .unitalker_random_blink import UnitalkerRandomBlink

_POSTPROCESS_CLASSES = dict(
    UnitalkerClip=UnitalkerClip,
    UnitalkerRandomBlink=UnitalkerRandomBlink,
    BlendshapeClip=BlendshapeClip,
    BlendshapeScale=BlendshapeScale,
    BlendshapeThreshold=BlendshapeThreshold,
    Offset=Offset,
    Rename=Rename,
    LinearExpBlend=LinearExpBlend,
    CustomBlink=CustomBlink,
)


def build_postprocess(cfg: dict) -> BasePostprocess:
    """Build a postprocessor instance from configuration dictionary.

    Selects and instantiates the corresponding postprocessor class based on
    the type field in the configuration dictionary. Supported postprocessor
    types include UnitalkerClip, UnitalkerRandomBlink, BlendshapeClip,
    BlendshapeScale, BlendshapeThreshold, Offset, Rename, LinearExpBlend,
    and CustomBlink.

    Args:
        cfg (dict):
            Postprocessor configuration dictionary. Must contain a type field
            specifying the postprocessor type. Other fields will be passed as
            parameters to the corresponding postprocessor constructor.

    Returns:
        BasePostprocess:
            Postprocessor instance built according to the configuration.

    Raises:
        TypeError:
            When the specified postprocessor type is not in the supported list.
    """
    cfg = cfg.copy()
    cls_name = cfg.pop('type')
    if cls_name not in _POSTPROCESS_CLASSES:
        msg = f'Unknown postprocess type: {cls_name}'
        raise TypeError(msg)
    ret_inst = _POSTPROCESS_CLASSES[cls_name](**cfg)
    return ret_inst

