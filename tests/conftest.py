"""
Fixtures compartidos para todos los tests de pytest.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock

# Asegurar que el root está en el path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configurar variables de entorno para tests
os.environ.setdefault("ENCRYPTION_KEY", "test-password-for-unit-tests-only!")
os.environ.setdefault("APP_PASSWORD", "test123")
os.environ.setdefault("API_KEY_SECRET", "test-api-key-12345")
os.environ.setdefault("ENVIRONMENT", "test")


@pytest.fixture(scope="session")
def sample_provinces():
    return ["Santo Domingo", "Santiago", "La Vega", "San Pedro de Macorís"]


@pytest.fixture(scope="session")
def sample_climate_df(sample_provinces):
    """DataFrame de clima sintético para tests."""
    np.random.seed(42)
    records = []
    for prov in sample_provinces:
        for week in range(1, 53):
            records.append({
                "province": prov, "year": 2024, "week": week,
                "date": f"2024-W{week:02d}",
                "rainfall_mm": max(0, np.random.exponential(60)),
                "temp_max_c": 28 + np.random.normal(0, 2),
                "temp_min_c": 22 + np.random.normal(0, 1.5),
                "temp_avg_c": 25 + np.random.normal(0, 1.8),
                "humidity_pct": min(100, max(40, np.random.normal(75, 8))),
                "wind_speed_kmh": max(0, np.random.normal(18, 4)),
                "enso_index": np.random.normal(0, 0.5),
                "cases": np.random.poisson(15),
                "deaths": 0,
                "population": 1_000_000,
                "population_density_km2": 800.0,
                "urban_pct": 70.0,
                "poverty_index": 25.0,
                "sanitation_index": 65.0,
                "incidence_rate_100k": np.random.exponential(10),
                "outbreak_risk_index": np.random.uniform(10, 85),
            })
    return pd.DataFrame(records)


@pytest.fixture(scope="session")
def sample_rf_data(sample_climate_df):
    """Features tabulares para Random Forest."""
    numeric_cols = sample_climate_df.select_dtypes(include=[np.number]).columns
    X = sample_climate_df[numeric_cols].drop(columns=["outbreak_risk_index"], errors="ignore").fillna(0)
    y = pd.DataFrame({
        "target_week_1": sample_climate_df["outbreak_risk_index"],
        "target_week_2": sample_climate_df["outbreak_risk_index"].shift(-1).fillna(0),
        "target_week_3": sample_climate_df["outbreak_risk_index"].shift(-2).fillna(0),
        "target_week_4": sample_climate_df["outbreak_risk_index"].shift(-3).fillna(0),
    })
    return X, y


@pytest.fixture(scope="session")
def sample_lstm_sequences():
    """Secuencias LSTM de prueba."""
    np.random.seed(42)
    X = np.random.rand(50, 12, 15).astype(np.float32)
    y = np.random.rand(50, 4).astype(np.float32) * 100
    return X, y


@pytest.fixture(scope="function")
def mock_ensemble():
    """Ensemble mockeado para tests de API."""
    ensemble = MagicMock()
    ensemble.get_ensemble_accuracy.return_value = 91.3
    ensemble._rf_ready = True
    ensemble._lstm_ready = True
    return ensemble
