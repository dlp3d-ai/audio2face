from .base_split import BaseSplit
from .dummy_split import DummySplit
from .energy_split import EnergySplit

_SPLIT_CLASSES = dict(
    EnergySplit=EnergySplit,
    DummySplit=DummySplit,
)


def build_split(cfg: dict) -> BaseSplit:
    """Build an audio splitter instance from configuration dictionary.

    Selects and instantiates the corresponding audio splitter class based on
    the type field in the configuration dictionary. Supported audio splitter
    types include EnergySplit and DummySplit.

    Args:
        cfg (dict):
            Audio splitter configuration dictionary. Must contain a type field
            specifying the splitter type. Other fields will be passed as
            parameters to the corresponding splitter constructor.

    Returns:
        BaseSplit:
            Audio splitter instance built according to the configuration.

    Raises:
        TypeError:
            When the specified audio splitter type is not in the supported list.
    """
    cfg = cfg.copy()
    cls_name = cfg.pop('type')
    if cls_name not in _SPLIT_CLASSES:
        msg = f'Unknown split type: {cls_name}'
        raise TypeError(msg)
    ret_inst = _SPLIT_CLASSES[cls_name](**cfg)
    return ret_inst

