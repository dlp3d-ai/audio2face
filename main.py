import argparse
import os
import sys

from audio2face.service.server import FastAPIServer
from audio2face.utils.config import file2dict


def main(args) -> int:
    """Start the backend server with the given configuration.

    Args:
        args: Parsed command line arguments containing config_path.

    Returns:
        int: Exit code, 0 for success.
    """
    if not os.path.exists('logs'):
        os.makedirs('logs')
    startup_config = file2dict(args.config_path)
    # init server
    cls_name = startup_config.pop('type')
    if cls_name == 'FastAPIServer':
        server = FastAPIServer(**startup_config)
    else:
        raise ValueError(f'Invalid server type: {cls_name}')
    server.run()
    return 0


def setup_parser():
    """Set up command line argument parser for the backend program.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser('Start the backed program.')
    # server args
    parser.add_argument(
        '--config_path',
        type=str,
        help='Path to the config file, which contains the server info.',
        default='configs/cpu.py')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_parser()
    ret_val = main(args)
    sys.exit(ret_val)
