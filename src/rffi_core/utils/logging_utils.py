"""Centralized logging configuration primitives."""

from __future__ import annotations

import logging
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Typed logging setup values."""

    level: int = logging.INFO
    fmt: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def build_logger(name: str, cfg: LoggingConfig | None = None) -> logging.Logger:
    """Create or return a named logger with an attached stream handler."""
    logger = logging.getLogger(name)
    logger.propagate = False

    if not logger.handlers:
        active = cfg or LoggingConfig()
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(active.fmt))
        logger.addHandler(handler)
        logger.setLevel(active.level)

    return logger
