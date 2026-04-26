"""
Thin wrapper used by the FastAPI app to load / reload the trained model
without importing the full ml package at module level.
"""

from ml.predict import get_artefact, load_model, reload_model

__all__ = ["get_artefact", "load_model", "reload_model"]
