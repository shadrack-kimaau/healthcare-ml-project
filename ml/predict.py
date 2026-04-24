"""
Prediction logic — loads the saved model artefact and runs inference.

This module is imported by the FastAPI app.  It keeps a module-level
cache of the loaded model so the file is only read from disk once per
process lifetime (or after an explicit reload call).
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml.preprocess import prepare_single_record

# paths 

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"


# module-level cache

_artefact: dict[str, Any] | None = None


def load_model(path: Path | None = None) -> dict:
    """
    Load the model artefact from disk and cache it.
    Call with path=None to use the default MODEL_PATH.
    """
    global _artefact
    target_path = path or MODEL_PATH

    if not target_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {target_path}.\n"
            "Run 'uv run python ml/train.py' first to train and save a model."
        )

    _artefact = joblib.load(target_path)
    print(
        f"[predict] Model loaded: {_artefact.get('model_name', 'unknown')} "
        f"(trained at {_artefact.get('trained_at', '?')})"
    )
    return _artefact


def get_artefact() -> dict:
    """Return the cached artefact, loading it on first access."""
    global _artefact
    if _artefact is None:
        load_model()
    return _artefact


def reload_model() -> None:
    """Force a reload from disk (called after a retraining run)."""
    global _artefact
    _artefact = None
    load_model()
    print("[predict] Model reloaded successfully.")


# inference 

def predict_single(payload: dict) -> str:
    """
    Run inference for a single patient record.

    Parameters
    ----------
    payload : dict with the API input fields (original casing OK):
        Age, Gender, Blood Type, Medical Condition, Billing Amount,
        Admission Type, Insurance Provider, Medication

    Returns
    -------
    str — one of: "Normal", "Abnormal", "Inconclusive"
    """
    artefact = get_artefact()
    pipeline = artefact["pipeline"]
    label_encoder = artefact["label_encoder"]

    # Convert payload → single-row DataFrame
    X = prepare_single_record(payload)

    # Predict
    y_encoded = pipeline.predict(X)
    label = label_encoder.inverse_transform(y_encoded)[0]

    return str(label)
