"""
Shared helper functions used across the FastAPI app.
"""

import logging
import os
import sys
from pathlib import Path


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Set up structured logging for the app."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("healthcare_api")


def ensure_model_exists() -> bool:
    """
    Return True if the saved model file exists.
    Used on startup to give a clear error message if training hasn't run yet.
    """
    model_path = Path(__file__).resolve().parents[1] / "models" / "model.joblib"
    return model_path.exists()


def get_version() -> str:
    return "1.0.0"
