"""
Tests de integración para la API FastAPI.
Usa httpx.AsyncClient para pruebas sin levantar servidor real.
"""

import pytest
import os
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient


os.environ.setdefault("API_KEY_SECRET", "test-api-key-12345")
os.environ.setdefault("ENCRYPTION_KEY", "test-password-for-unit-tests!")
os.environ.setdefault("APP_PASSWORD", "test123")


@pytest.fixture(scope="module")
def test_client():
    """Cliente de prueba con ensemble mockeado."""
    from api.main import app

    with TestClient(app) as client:
        # Mockear el ensemble en el estado de la app
        mock_ensemble = MagicMock()
        mock_ensemble.get_ensemble_accuracy.return_value = 91.3
        mock_ensemble._rf_ready = True
        mock_ensemble._lstm_ready = True
        app.state.ensemble = mock_ensemble
        yield client


API_KEY = "test-api-key-12345"
HEADERS = {"X-API-Key": API_KEY}


class TestRootAndHealth:
    def test_root_returns_200(self, test_client):
        resp = test_client.get("/")
        assert resp.status_code == 200

    def test_root_has_required_fields(self, test_client):
        resp = test_client.get("/")
        data = resp.json()
        assert "sistema" in data
        assert "version" in data
        assert "endpoints_principales" in data

    def test_health_endpoint(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "ensemble_accuracy_pct" in data


class TestAuthentication:
    def test_no_api_key_returns_401(self, test_client):
        resp = test_client.get("/predict/Santiago")
        assert resp.status_code == 401

    def test_wrong_api_key_returns_403(self, test_client):
        resp = test_client.get("/predict/Santiago", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 403

    def test_valid_api_key_accepted(self, test_client):
        resp = test_client.get("/provinces/list", headers=HEADERS)
        assert resp.status_code == 200


class TestPredictionEndpoint:
    def test_valid_province(self, test_client):
        resp = test_client.get("/predict/Santiago", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["province"] == "Santiago"
        assert 0 <= data["current_risk_index"] <= 100
        assert "forecast_4_weeks" in data

    def test_invalid_province_returns_404(self, test_client):
        resp = test_client.get("/predict/CiudadFicticia", headers=HEADERS)
        assert resp.status_code == 404

    def test_forecast_has_4_weeks(self, test_client):
        resp = test_client.get("/predict/Azua", headers=HEADERS)
        assert resp.status_code == 200
        forecast = resp.json()["forecast_4_weeks"]
        assert "week_1" in forecast
        assert "week_2" in forecast
        assert "week_3" in forecast
        assert "week_4" in forecast

    def test_forecast_values_in_range(self, test_client):
        resp = test_client.get("/predict/Barahona", headers=HEADERS)
        data = resp.json()
        for week in ["week_1", "week_2", "week_3", "week_4"]:
            val = data["forecast_4_weeks"][week]
            assert 0 <= val <= 100, f"{week}: {val} fuera de rango"

    def test_response_has_risk_level(self, test_client):
        resp = test_client.get("/predict/La%20Vega", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["risk_level"] in ["Bajo", "Moderado", "Alto", "Epidemia", "Crítico"]

    def test_response_has_trend(self, test_client):
        resp = test_client.get("/predict/Duarte", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["trend"] in ["Ascendente", "Descendente", "Estable"]


class TestProvincesEndpoint:
    def test_province_list(self, test_client):
        resp = test_client.get("/provinces/list", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "provinces" in data
        assert data["total"] == 32
        assert len(data["provinces"]) == 32


class TestAlertsEndpoint:
    def test_alerts_endpoint_returns_200(self, test_client):
        resp = test_client.get("/alerts", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "total_alerts" in data
        assert "alerts" in data
        assert isinstance(data["alerts"], list)

    def test_critical_alerts_endpoint(self, test_client):
        resp = test_client.get("/alerts/critical", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "level" in data
        assert data["level"] == "Crítico"
        assert data["threshold"] == 80

    def test_custom_threshold(self, test_client):
        resp = test_client.get("/alerts?threshold=90", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        for alert in data["alerts"]:
            assert alert["current_risk_index"] >= 90


class TestUploadEndpoint:
    def test_upload_valid_csv(self, test_client, tmp_path):
        csv_content = "province,year,week,cases\nSantiago,2024,15,120\nAzua,2024,15,45\n"
        files = {"file": ("data.csv", csv_content.encode(), "text/csv")}
        resp = test_client.post("/upload", headers=HEADERS, files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["records_processed"] == 2

    def test_upload_non_csv_rejected(self, test_client):
        files = {"file": ("data.xlsx", b"binary content", "application/vnd.ms-excel")}
        resp = test_client.post("/upload", headers=HEADERS, files=files)
        assert resp.status_code == 422

    def test_upload_missing_columns_rejected(self, test_client):
        csv_bad = "nombre,valor\nSantiago,100\n"
        files = {"file": ("bad.csv", csv_bad.encode(), "text/csv")}
        resp = test_client.post("/upload", headers=HEADERS, files=files)
        assert resp.status_code == 422
