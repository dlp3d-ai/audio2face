from abc import ABC, abstractmethod
from typing import Any

from ..data_structures.face_clip import FaceClip
from ..utils.super import Super


class BasePostprocess(Super, ABC):
    """Base class for postprocessing operations.

    Defines the standard interface for facial expression animation postprocessing.
    All concrete postprocessing classes should inherit from this class and
    implement the __call__ method.
    """

    def __init__(self, logger_cfg: None | dict[str, Any] = None):
        """Initialize the postprocessing base class.

        Args:
            logger_cfg (None | dict[str, Any], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        ABC.__init__(self)
        Super.__init__(self, logger_cfg)

    @abstractmethod
    def __call__(
            self,
            face_clip: FaceClip,
            **kwargs) -> FaceClip:
        """Execute postprocessing operation.

        Process the input facial expression animation clip and return the
        processed result. This is an abstract method that must be implemented
        by subclasses with specific processing logic.

        Args:
            face_clip (FaceClip):
                Input facial expression animation clip.
            **kwargs:
                Additional keyword arguments, specific meaning defined by subclasses.

        Returns:
            FaceClip:
                Processed facial expression animation clip.
        """
        pass

    def clean_stream(self, stream_id: str) -> None:
        """Clean stream.

        Clean the stream, used for cleaning cached data in the stream.

        Args:
            stream_id (str):
                Stream ID.
        """
        pass
