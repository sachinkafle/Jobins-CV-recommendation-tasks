"""Logging configuration"""
import logging
import sys
from rich.logging import RichHandler

def setup_logger(name: str = "cv_matching", level: int = logging.INFO) -> logging.Logger:
    """Setup logger with rich formatting"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = RichHandler(rich_tracebacks=True)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    return logger

# Global logger instance
logger = setup_logger()
