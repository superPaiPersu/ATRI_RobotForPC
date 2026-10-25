"""Shared logging configuration for the desktop-pet server."""

from __future__ import annotations

import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


SERVER_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = SERVER_DIR / "server.log"
LOGGER_NAME = "ai_pet"


def _parse_level(value: str | None) -> int:
    name = str(value or "INFO").strip().upper()
    return getattr(logging, name, logging.INFO)


def configure_logging() -> logging.Logger:
    """Configure one console handler and one bounded UTF-8 file handler."""
    logger = logging.getLogger(LOGGER_NAME)
    if getattr(logger, "_ai_pet_configured", False):
        return logger

    level = _parse_level(os.getenv("AI_PET_LOG_LEVEL"))
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_path = Path(os.getenv("AI_PET_LOG_FILE", str(DEFAULT_LOG_PATH))).expanduser()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
    except OSError:
        file_handler = None
    if file_handler is not None:
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger._ai_pet_configured = True
    return logger


def get_logger(component: str | None = None) -> logging.Logger:
    configure_logging()
    if not component:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{component}")


def compact_value(value, max_chars: int = 180) -> str:
    """Keep log fields short and avoid dumping complete payloads."""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(value)
    text = " ".join(text.split())
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text

