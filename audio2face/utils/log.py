import atexit
import logging
import socket

import boto3
from watchtower import CloudWatchLogHandler

# Get hostname
_HOSTNAME_FULL = socket.gethostname()
_HOSTNAME_SPLIT_MARKS = (".", "_", "-")
_HOSTNAME = _HOSTNAME_FULL
for mark in _HOSTNAME_SPLIT_MARKS:
    if mark in _HOSTNAME_FULL:
        _HOSTNAME = _HOSTNAME_FULL.rsplit(mark)[-1]


# Custom LogRecord factory, add hostname field
def custom_log_record_factory(*args, **kwargs):
    """Create a custom LogRecord with hostname field.

    Args:
        *args: Variable length argument list for LogRecord.
        **kwargs: Arbitrary keyword arguments for LogRecord.

    Returns:
        logging.LogRecord:
            LogRecord instance with hostname field added.
    """
    record = logging.LogRecord(*args, **kwargs)
    record.hostname = _HOSTNAME
    return record


logging.setLogRecordFactory(custom_log_record_factory)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(hostname)s - %(name)s - %(levelname)s - %(thread)d - %(message)s"  # noqa: E501
)

_cloudwatch_handlers = dict()


def setup_logger(
    logger_name: str = "root",
    aws_level: int = logging.INFO,
    file_level: int = logging.INFO,
    console_level: int = logging.INFO,
    logger_level: int | None = None,
    logger_path: str | None = None,
    aws_group_name: str | None = None,
    aws_stream_name: str | None = None,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_region: str | None = None,
    logger_format: str | None = None,
    aws_send_interval: int = 5,
    aws_use_queues: bool = True,
) -> logging.Logger:
    """Setup and configure logger.

    Create a logger with specified configuration, supporting simultaneous output
    to console, file, and AWS CloudWatch, with separate log levels for each
    output stream.

    Args:
        logger_name (str, optional):
            Logger name. Defaults to 'root'.
        aws_level (int, optional):
            Log level for AWS CloudWatch output stream. Defaults to logging.INFO.
        file_level (int, optional):
            Log level for file output stream. Defaults to logging.INFO.
        console_level (int, optional):
            Log level for console output stream. Defaults to logging.INFO.
        logger_level (int | None, optional):
            Global logger level, deprecated. Use file_level/console_level
            instead. Defaults to None.
        logger_path (str | None, optional):
            Log file path. If None, no file output. Defaults to None.
        aws_group_name (str | None, optional):
            AWS CloudWatch log group name. If None, no AWS output.
            Defaults to None.
        aws_stream_name (str | None, optional):
            AWS CloudWatch log stream name. If None, no AWS output.
            Defaults to None.
        aws_access_key_id (str | None, optional):
            AWS access key ID. Defaults to None.
        aws_secret_access_key (str | None, optional):
            AWS secret access key. Defaults to None.
        aws_region (str | None, optional):
            AWS region name. Defaults to None.
        logger_format (str | None, optional):
            Log format string. If None, uses default format. Defaults to None.
        aws_send_interval (int, optional):
            AWS CloudWatch log send interval in seconds. Recommended 5-30
            seconds to reduce API calls. Defaults to 5.
        aws_use_queues (bool, optional):
            Whether to use queues for async AWS CloudWatch log sending. True
            improves performance but may generate warnings on program exit.
            Defaults to True.

    Returns:
        logging.Logger:
            Configured logger instance.

    Note:
        logger_level parameter is deprecated. Use file_level, console_level
        and aws_level to set log levels for different output streams separately.
        Program will automatically gracefully shutdown all CloudWatch log
        handlers on exit to avoid warnings.
    """
    logger = logging.getLogger(logger_name)
    level_candidates = [file_level, console_level, aws_level]
    if logger_level is not None:
        level_candidates = [
            logger_level,
        ]
        console_level = logger_level
        file_level = logger_level
        aws_level = logger_level
    min_level = min(level_candidates)
    logger.setLevel(level=min_level)
    # prevent logging twice in stdout
    logger.propagate = False
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(console_level)
    handlers = [stream_handler]
    if logger_path is not None:
        handler = logging.FileHandler(logger_path, encoding="utf-8")
        handler.setLevel(file_level)
        handlers.append(handler)
    if aws_group_name is not None and aws_stream_name is not None:
        aws_logger_key = f"{aws_group_name}-{aws_stream_name}-" + \
            f"{aws_access_key_id}-{aws_region}-{aws_level}"
        if aws_logger_key in _cloudwatch_handlers:
            handler = _cloudwatch_handlers[aws_logger_key]
        else:
            client = boto3.client(
                service_name="logs",
                region_name=aws_region,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
            handler = CloudWatchLogHandler(
                boto3_client=client,
                log_group_name=aws_group_name,
                log_stream_name=aws_stream_name,
                send_interval=aws_send_interval,
                use_queues=aws_use_queues,
            )
            handler.setLevel(aws_level)
            _cloudwatch_handlers[aws_logger_key] = handler
        handlers.append(handler)
    if logger_format is not None:
        formatter = logging.Formatter(logger_format)
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(hostname)s - %(name)s - %(levelname)s - %(thread)d - %(message)s"  # noqa: E501
        )
    # assure handlers are not double
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if logger_level is not None:
        logger.warning(
            "UserWarning: logger_level is now deprecated, " +
            "please specify file_level/console_level instead."
        )
    return logger


def get_logger(logger: None | str | logging.Logger = None) -> logging.Logger:
    """Get logger instance.

    Get corresponding logger instance based on input parameter,
    supporting multiple input types.

    Args:
        logger (None | str | logging.Logger, optional):
            Logger identifier. None means get root logger, string means
            logger name. Defaults to None.

    Returns:
        logging.Logger:
            Corresponding logger instance.
    """
    if logger is None or isinstance(logger, str):
        ret_logger = logging.getLogger(logger)
    else:
        ret_logger = logger
    return ret_logger


def shutdown_cloudwatch_handlers():
    """Gracefully shutdown all AWS CloudWatch log handlers.

    Ensure all pending log messages are sent to AWS CloudWatch,
    avoiding "Received message after logging system shutdown" warnings
    on program exit. This function is automatically called on program exit.

    Note:
        This function is registered through the atexit module and will
        automatically execute on normal program exit. Any exceptions during
        shutdown are silently ignored to ensure normal program exit.
    """
    try:
        for handler in _cloudwatch_handlers.values():
            handler.flush()
            handler.close()
    except Exception:
        pass


atexit.register(shutdown_cloudwatch_handlers)
