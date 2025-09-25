from .dummy_feature_extractor import DummyFeatureExtractor
from .dummy_generator import DummyGenerator
from .onnx_unitalker import OnnxUnitalker
from .torch_feature_extractor import TorchFeatureExtractor

_INFER_CLASSES = dict(
    OnnxUnitalker=OnnxUnitalker,
    DummyGenerator=DummyGenerator,
    TorchFeatureExtractor=TorchFeatureExtractor,
    DummyFeatureExtractor=DummyFeatureExtractor,
)


def build_infer(
    cfg: dict
) -> OnnxUnitalker | TorchFeatureExtractor | DummyGenerator | DummyFeatureExtractor:
    """Build an inference engine instance from configuration dictionary.

    Selects the corresponding inference engine class for instantiation based on
    the type field in the configuration dictionary. Supported inference engine
    types include OnnxUnitalker, TorchFeatureExtractor, DummyGenerator, and
    DummyFeatureExtractor.

    Args:
        cfg (dict):
            Inference engine configuration dictionary, must contain type field
            to specify inference engine type. Other fields will be passed as
            parameters to the corresponding inference engine constructor.

    Returns:
        OnnxUnitalker | TorchFeatureExtractor | DummyGenerator | DummyFeatureExtractor:
            Inference engine instance built according to configuration.

    Raises:
        TypeError: When the specified inference engine type is not in the
            supported list.
    """
    cfg = cfg.copy()
    cls_name = cfg.pop('type')
    if cls_name not in _INFER_CLASSES:
        msg = f'Unknown infer type: {cls_name}'
        raise TypeError(msg)
    ret_inst = _INFER_CLASSES[cls_name](**cfg)
    return ret_inst

