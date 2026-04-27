"""
Tests unitarios para los modelos ML: Random Forest, LSTM y Ensemble.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


class TestRandomForestModel:
    @pytest.fixture
    def rf_model(self):
        from src.models.random_forest import RandomForestModel
        return RandomForestModel(params={
            "n_estimators": 10,
            "max_depth": 5,
            "random_state": 42,
            "n_jobs": 1,
        })

    def test_train_returns_metrics(self, rf_model, sample_rf_data):
        X, y = sample_rf_data
        metrics = rf_model.train(X.iloc[:40], y.iloc[:40], X.iloc[40:], y.iloc[40:])
        assert isinstance(metrics, dict)
        assert "train_mae" in metrics
        assert "cv_mae_mean" in metrics

    def test_predict_shape(self, rf_model, sample_rf_data):
        X, y = sample_rf_data
        rf_model.train(X.iloc[:40], y.iloc[:40])
        preds = rf_model.predict(X.iloc[:5])
        assert preds.shape == (5, 4)  # (samples, FORECAST_HORIZON)

    def test_predict_range(self, rf_model, sample_rf_data):
        """Predicciones deben estar en rango 0-100."""
        X, y = sample_rf_data
        rf_model.train(X.iloc[:40], y.iloc[:40])
        preds = rf_model.predict(X.iloc[:10])
        assert preds.min() >= 0
        assert preds.max() <= 100

    def test_save_load(self, rf_model, sample_rf_data, tmp_path):
        from src.models.random_forest import RandomForestModel
        X, y = sample_rf_data
        rf_model.train(X.iloc[:40], y.iloc[:40])
        save_path = tmp_path / "rf_test.pkl"
        rf_model.save(save_path)
        assert save_path.exists()

        rf2 = RandomForestModel()
        rf2.model  # no cargado aún
        rf2._load = lambda: None  # monkeypatch
        import joblib
        data = joblib.load(save_path)
        assert "model" in data
        assert "feature_names" in data

    def test_get_accuracy_range(self, rf_model, sample_rf_data):
        X, y = sample_rf_data
        rf_model.train(X.iloc[:40], y.iloc[:40], X.iloc[40:], y.iloc[40:])
        acc = rf_model.get_accuracy()
        assert 0 <= acc <= 100

    def test_feature_importance_top_n(self, rf_model, sample_rf_data):
        X, y = sample_rf_data
        rf_model.train(X.iloc[:40], y.iloc[:40])
        fi = rf_model.get_feature_importance(top_n=5)
        assert len(fi) <= 5
        assert "feature" in fi.columns
        assert "importance" in fi.columns

    def test_predict_province_format(self, rf_model, sample_rf_data):
        X, y = sample_rf_data
        rf_model.train(X.iloc[:40], y.iloc[:40])
        result = rf_model.predict_province(X.iloc[:5])
        assert isinstance(result, dict)
        for week in range(1, 5):
            assert f"week_{week}" in result


class TestPreprocessor:
    def test_fit_transform_returns_df(self, sample_climate_df):
        from src.data.preprocessing import DataPreprocessor
        pp = DataPreprocessor()
        result = pp.fit_transform(sample_climate_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_creates_lag_features(self, sample_climate_df):
        from src.data.preprocessing import DataPreprocessor
        pp = DataPreprocessor()
        result = pp.fit_transform(sample_climate_df)
        lag_cols = [c for c in result.columns if "_lag" in c]
        assert len(lag_cols) > 0

    def test_creates_rolling_features(self, sample_climate_df):
        from src.data.preprocessing import DataPreprocessor
        pp = DataPreprocessor()
        result = pp.fit_transform(sample_climate_df)
        roll_cols = [c for c in result.columns if "_roll" in c]
        assert len(roll_cols) > 0

    def test_no_nulls_after_transform(self, sample_climate_df):
        from src.data.preprocessing import DataPreprocessor
        pp = DataPreprocessor()
        result = pp.fit_transform(sample_climate_df)
        numeric = result.select_dtypes(include=[np.number])
        assert not numeric.isnull().any().any()

    def test_prepare_lstm_sequences_shape(self, sample_climate_df):
        from src.data.preprocessing import DataPreprocessor
        pp = DataPreprocessor()
        processed = pp.fit_transform(sample_climate_df)
        X, y = pp.prepare_lstm_sequences(processed, sequence_length=8)
        if len(X) > 0:
            assert X.ndim == 3  # (samples, timesteps, features)
            assert y.ndim == 2  # (samples, FORECAST_HORIZON)
            assert X.shape[1] == 8


class TestEnsembleClassification:
    def test_classify_risk_levels(self):
        from src.models.ensemble import EnsembleModel
        model = EnsembleModel.__new__(EnsembleModel)
        assert EnsembleModel._classify_risk(10) == "Bajo"
        assert EnsembleModel._classify_risk(35) == "Moderado"
        assert EnsembleModel._classify_risk(55) == "Alto"
        assert EnsembleModel._classify_risk(70) == "Epidemia"
        assert EnsembleModel._classify_risk(90) == "Crítico"

    def test_classify_risk_boundaries(self):
        from src.models.ensemble import EnsembleModel
        assert EnsembleModel._classify_risk(0)   == "Bajo"
        assert EnsembleModel._classify_risk(25)  == "Moderado"
        assert EnsembleModel._classify_risk(100) == "Crítico"

    def test_calculate_trend_ascending(self):
        from src.models.ensemble import EnsembleModel
        assert EnsembleModel._calculate_trend(np.array([30, 40, 55, 70])) == "Ascendente"

    def test_calculate_trend_descending(self):
        from src.models.ensemble import EnsembleModel
        assert EnsembleModel._calculate_trend(np.array([70, 55, 40, 30])) == "Descendente"

    def test_calculate_trend_stable(self):
        from src.models.ensemble import EnsembleModel
        assert EnsembleModel._calculate_trend(np.array([45, 46, 44, 45])) == "Estable"

    def test_weights_sum_to_one(self):
        from src.models.ensemble import EnsembleModel
        model = EnsembleModel(rf_weight=0.6, lstm_weight=0.4)
        assert abs(model.rf_weight + model.lstm_weight - 1.0) < 1e-9

    def test_invalid_weights_raise(self):
        from src.models.ensemble import EnsembleModel
        with pytest.raises(AssertionError):
            EnsembleModel(rf_weight=0.7, lstm_weight=0.5)


class TestDataIngestion:
    def test_simulate_paho_data_shape(self):
        from src.data.ingestion import DataIngestion
        ingestion = DataIngestion()
        df = ingestion._simulate_paho_data(2024)
        assert "province" in df.columns
        assert "cases" in df.columns
        assert len(df) == 52 * 32  # 52 semanas × 32 provincias

    def test_simulate_climate_non_negative(self):
        from src.data.ingestion import DataIngestion
        ingestion = DataIngestion()
        df = ingestion._simulate_climate_data("Santo Domingo", 26)
        assert (df["rainfall_mm"] >= 0).all()
        assert (df["humidity_pct"] >= 0).all()
        assert (df["humidity_pct"] <= 100).all()

    def test_reference_demographic_all_provinces(self):
        from src.data.ingestion import DataIngestion
        from config.settings import PROVINCES
        ingestion = DataIngestion()
        df = ingestion._get_reference_demographic_data()
        for province in PROVINCES:
            assert province in df["province"].values, f"Falta provincia: {province}"

    def test_risk_index_range(self):
        from src.data.ingestion import DataIngestion
        ingestion = DataIngestion()
        demo = ingestion._get_reference_demographic_data()
        climate = ingestion._simulate_climate_data("Santiago", 10)
        dengue = ingestion._simulate_paho_data(2024)
        enso = ingestion._simulate_enso_data()
        master = ingestion._merge_datasets(dengue, climate, demo, enso)
        risk = master["outbreak_risk_index"]
        assert (risk >= 0).all()
        assert (risk <= 100).all()
