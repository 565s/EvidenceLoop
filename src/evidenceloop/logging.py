"""Logging is configured only by an explicit command or application."""

import logging

logger = logging.getLogger("evidenceloop")


def configure_logging(level="INFO", filename=None):
    handlers = [logging.StreamHandler()]
    if filename is not None:
        handlers.append(logging.FileHandler(filename, encoding="utf-8"))
    logging.basicConfig(level=level, handlers=handlers)
