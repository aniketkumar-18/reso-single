"""Centralised logger factory with ANSI color output.

Usage in every module:
    from src.infra.logger import setup_logger
    logger = setup_logger(__name__)

Per-call color override (optional):
    logger.info("msg", extra={"log_color": "\033[35m"})  # magenta
"""

from __future__ import annotations

import logging

# ── ANSI color codes ───────────────────────────────────────────────────────────

RESET_COLOR = "\033[0m"

# Default colors by log level
_LEVEL_COLORS: dict[str, str] = {
    "DEBUG":    "\033[36m",    # Cyan
    "INFO":     "\033[32m",    # Green
    "WARNING":  "\033[33m",    # Yellow
    "ERROR":    "\033[31m",    # Red
    "CRITICAL": "\033[1;31m",  # Bold Red
}

# Named colors for extra={"log_color": ...} overrides
COLORS = {
    "cyan":    "\033[36m",
    "green":   "\033[32m",
    "yellow":  "\033[33m",
    "red":     "\033[31m",
    "magenta": "\033[35m",
    "blue":    "\033[34m",
    "white":   "\033[37m",
    "bold":    "\033[1m",
}


# ── Formatter ─────────────────────────────────────────────────────────────────

class FunctionColorFormatter(logging.Formatter):
    """Custom formatter that adds color to log messages based on log_color attribute.

    Falls back to per-level color when log_color is not set on the record.
    """

    def format(self, record: logging.LogRecord) -> str:
        color = getattr(
            record,
            "log_color",
            _LEVEL_COLORS.get(record.levelname, RESET_COLOR),
        )
        msg = super().format(record)
        return f"{color}{msg}{RESET_COLOR}"


# ── Factory ───────────────────────────────────────────────────────────────────

def setup_logger(logger_name: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger with color formatting.

    Args:
        logger_name: Name of the logger (pass __name__ from calling module).
        log_level:   Logging level (default: logging.INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # Clear existing handlers to avoid duplicate logs on hot-reload
    logger.handlers = []
    logger.propagate = False

    handler = logging.StreamHandler()
    handler.setFormatter(
        FunctionColorFormatter(
            fmt="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logger.addHandler(handler)
    return logger
