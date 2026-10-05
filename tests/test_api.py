"""
Integration Tests for FastAPI Endpoints, Metrics, and Drift Reporting.
"""

import pytest
from fastapi.testclient import TestClient

from src.app.main import app


@pytest.fixture(scope="module")
def client():
    """Context client that triggers startup/shutdown lifecycle events."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Ensure /health returns 200 with model and reference status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "model_version" in data
    assert "uptime_seconds" in data


def test_predict_endpoint_valid(client):
    """Test single valid inference request."""
    payload = {
        "features": {
            "age": 35,
            "income": 65000.0,
            "credit_score": 720.0,
            "debt_to_income": 0.22,
            "loan_amount": 12000.0,
            "employment_status": "employed",
            "loan_purpose": "debt_consolidation",
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]
    assert data["prediction_label"] in ["default", "non_default"]
    assert 0.0 <= data["probability"] <= 1.0
    assert "model_version" in data


def test_predict_endpoint_invalid_input(client):
    """Test validation failure for out-of-bounds input."""
    payload = {
        "features": {
            "age": 10,  # Invalid: age < 18
            "income": -500.0,  # Invalid: negative income
            "credit_score": 950.0,  # Invalid: > 850
            "debt_to_income": 0.2,
            "loan_amount": 1000.0,
            "employment_status": "invalid_status",
            "loan_purpose": "vacation",
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity


def test_batch_predict_endpoint(client):
    """Test batch prediction with multiple instances."""
    payload = {
        "instances": [
            {
                "age": 28,
                "income": 45000.0,
                "credit_score": 640.0,
                "debt_to_income": 0.35,
                "loan_amount": 10000.0,
                "employment_status": "employed",
                "loan_purpose": "education",
            },
            {
                "age": 52,
                "income": 120000.0,
                "credit_score": 790.0,
                "debt_to_income": 0.15,
                "loan_amount": 25000.0,
                "employment_status": "self_employed",
                "loan_purpose": "home_improvement",
            },
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2


def test_prometheus_metrics_endpoint(client):
    """Test /metrics exposes standard Prometheus format."""
    response = client.get("/metrics")
    assert response.status_code == 200
    content = response.text
    assert "model_prediction_requests_total" in content
    assert "model_prediction_latency_seconds" in content
    assert "model_dataset_drift_detected" in content


def test_drift_endpoints(client):
    """Test /drift/evaluate, /drift/metrics, and /drift/report."""
    # Test evaluation
    eval_resp = client.post("/drift/evaluate")
    assert eval_resp.status_code == 200
    eval_data = eval_resp.json()
    assert "dataset_drift_detected" in eval_data

    # Test JSON metrics
    metrics_resp = client.get("/drift/metrics")
    assert metrics_resp.status_code == 200

    # Test HTML report
    report_resp = client.get("/drift/report")
    assert report_resp.status_code == 200
    assert "text/html" in report_resp.headers["content-type"]
    assert "ML Model Drift Detection Report" in report_resp.text


def test_root_dashboard_endpoint(client):
    """Test / serves the interactive glassmorphic web dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "ML RiskOps Platform" in response.text
    assert "Real-Time Inference Playground" in response.text

