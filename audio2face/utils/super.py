from typing import Any

from .log import setup_logger


class Super:
    """Super class for all classes in the project.

    Args:
        logger_cfg (Union[None, Dict[str, Any]], optional):
            Logger configuration. Defaults to None.
    """

    def __init__(self, logger_cfg: None | dict[str, Any] = None):
        """
        Args:
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration, see `setup_logger` for more details.
                Logger name will be the class name.
                Default to None.
        """
        logger_name = self.__class__.__name__
        if logger_cfg is None:
            logger_cfg = dict(logger_name=logger_name)
        else:
            logger_cfg = logger_cfg.copy()
            logger_cfg["logger_name"] = logger_name
        self.logger_cfg = logger_cfg
        self.logger = setup_logger(**logger_cfg)
