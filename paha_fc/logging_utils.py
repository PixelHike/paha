"""Shared logging helpers."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def _configure(level: int) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger("paha").setLevel(level)
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s",
                                           datefmt="%H:%M:%S"))
    root = logging.getLogger("paha")
    root.handlers = [handler]
    root.setLevel(level)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str = "paha", level: int = logging.INFO) -> logging.Logger:
    _configure(level)
    return logging.getLogger(name if name.startswith("paha") else f"paha.{name}")


def set_verbosity(verbose: bool, quiet: bool = False) -> None:
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    _configure(level)
