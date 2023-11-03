"""
Logging utility. Adapted from
https://github.com/skypilot-org/skypilot/blob/master/sky/sky_logging.py
"""

import logging
import sys
import colorlog

# pylint: disable=line-too-long
_FORMAT = "%(log_color)s%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"


class ColoredFormatter(colorlog.ColoredFormatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self,
                 fmt,
                 datefmt=None,
                 log_colors=None,
                 reset=True,
                 style="%"):
        super().__init__(fmt,
                         datefmt=datefmt,
                         log_colors=log_colors,
                         reset=reset,
                         style=style)

    def format(self, record):
        msg = super().format(record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


_root_logger = logging.getLogger("aphrodite")
_default_handler = None


def _setup_logger():
    _root_logger.setLevel(logging.DEBUG)
    global _default_handler
    if _default_handler is None:
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.flush = sys.stdout.flush  # type: ignore
        _default_handler.setLevel(logging.INFO)
        _root_logger.addHandler(_default_handler)
    fmt = ColoredFormatter(_FORMAT,
                           datefmt=_DATE_FORMAT,
                           log_colors={
                               "DEBUG": "cyan",
                               "INFO": "green",
                               "WARNING": "yellow",
                               "ERROR": "red",
                               "CRITICAL": "red,bg_white",
                           },
                           reset=True)
    _default_handler.setFormatter(fmt)
    # Setting this will avoid the message
    # being propagated to the parent logger.
    _root_logger.propagate = False


# The logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
_setup_logger()


def init_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(_default_handler)
    logger.propagate = False
    return logger
