"""
test_api.py

Tests for the FastAPI endpoints.
MLflow is mocked so these tests run without any MLflow server or trained model.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# We patch mlflow.sklearn.load_model before importing the app
# so the lifespan startup doesn't fail in CI
@pytest.fixture
def client():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

    with patch("mlflow.sklearn.load_model", return_value=mock_model), \
         patch("mlflow.set_tracking_uri"), \
         patch("api.main._model", mock_model), \
         patch("api.main._model_loaded", True):

        from api.main import app
        with TestClient(app) as c:
            yield c


SAMPLE_PAYLOAD = {
    "Pregnancies": 2,
    "Glucose": 138,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 80,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 47,
}


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_name" in data


def test_predict_returns_valid_response(client):
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200

    data = response.json()
    assert data["prediction"] in [0, 1]
    assert data["prediction_label"] in ["Diabetes Risk Detected", "No Diabetes Risk"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "model_name" in data


def test_predict_invalid_glucose_rejected(client):
    """Glucose of 0 is not valid per our schema (gt=0)."""
    bad_payload = {**SAMPLE_PAYLOAD, "Glucose": 0}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_missing_field_rejected(client):
    """Missing a required field should return a 422 validation error."""
    incomplete = {k: v for k, v in SAMPLE_PAYLOAD.items() if k != "BMI"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_dashboard_get_returns_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Diabetes Risk Predictor" in response.text


def test_dashboard_post_returns_result(client):
    form_data = {k: str(v) for k, v in SAMPLE_PAYLOAD.items()}
    response = client.post("/dashboard-predict", data=form_data)
    assert response.status_code == 200
    # Result section should appear in the rendered HTML
    assert "confidence" in response.text.lower() or "risk" in response.text.lower()
