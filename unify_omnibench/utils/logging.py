"""Lightweight logger factory."""
import logging
import sys

_FORMAT = "%(asctime)s | %(levelname).1s | %(name)s | %(message)s"


def get_logger(name: str = "unify_omnibench", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
    logger.addHandler(h)
    logger.propagate = False
    return logger
