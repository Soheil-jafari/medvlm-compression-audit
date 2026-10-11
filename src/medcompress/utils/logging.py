"""Logging setup shared by scripts and library code (standard library only)."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str = "medcompress", level: int | str = logging.INFO) -> logging.Logger:
    """Return a logger that writes to stderr with a consistent format.

    Calling it repeatedly with the same name does not add duplicate handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger
