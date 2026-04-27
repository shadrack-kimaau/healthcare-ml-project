"""
Unit / integration tests for the FastAPI endpoints.

Run with:
    uv run pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# Patch DB and model loading before importing the app
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="module")
def client():
    """Return a TestClient with DB + model mocked out."""
    with (
        patch("database.db_connection.get_engine"),
        patch("database.test_connection", return_value=True),
        patch("database.models.create_all_tables"),
        patch("app.utils.ensure_model_exists", return_value=True),
        patch("app.model_loader.load_model"),
    ):
        from app.main import create_app
        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ── /health ─

def test_health_returns_200(client):
    with (
        patch("app.routes.test_connection", return_value=True),
        patch("app.routes.get_artefact", return_value={"model_name": "XGBoost"}),
    ):
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "db_connected" in body
    assert "model_loaded" in body


# ── /predict ───

VALID_PAYLOAD = {
    "Age": 45,
    "Gender": "Male",
    "Blood Type": "O+",
    "Medical Condition": "Diabetes",
    "Billing Amount": 2000.50,
    "Admission Type": "Emergency",
    "Insurance Provider": "Cigna",
    "Medication": "Aspirin",
}


def test_predict_returns_valid_class(client):
    with patch("app.routes.predict_single", return_value="Abnormal"), \
         patch("app.routes.get_artefact", return_value={"model_name": "XGBoost"}):
        resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_test_result"] in {"Normal", "Abnormal", "Inconclusive"}


def test_predict_invalid_gender_returns_422(client):
    bad_payload = {**VALID_PAYLOAD, "Gender": "Robot"}
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_predict_invalid_age_returns_422(client):
    bad_payload = {**VALID_PAYLOAD, "Age": -5}
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_predict_invalid_admission_type_returns_422(client):
    bad_payload = {**VALID_PAYLOAD, "Admission Type": "Walk-in"}
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_predict_missing_field_returns_422(client):
    incomplete = {k: v for k, v in VALID_PAYLOAD.items() if k != "Medication"}
    resp = client.post("/predict", json=incomplete)
    assert resp.status_code == 422


def test_predict_model_not_found_returns_503(client):
    with patch("app.routes.predict_single", side_effect=FileNotFoundError("no model")):
        resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 503


# ── /model-info ─

def test_model_info_returns_name_and_date(client):
    fake_artefact = {"model_name": "XGBoost", "trained_at": "2025-01-01T12:00:00"}
    with patch("app.routes.get_artefact", return_value=fake_artefact):
        resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_name"] == "XGBoost"


# ── /train ──

def test_train_returns_202(client):
    resp = client.post("/train")
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "accepted"
