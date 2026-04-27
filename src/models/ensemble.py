"""
Modelo ensemble que combina Random Forest + LSTM para predicción final.
Usa pesos configurables para ponderar las predicciones de cada modelo.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
from config.settings import ENSEMBLE_WEIGHTS, FORECAST_HORIZON, EPIDEMIC_THRESHOLD, ALERT_THRESHOLD, PROVINCES, Paths
from .random_forest import RandomForestModel
from .lstm import LSTMModel


class EnsembleModel:
    """
    Ensemble ponderado: Risk_final = w_rf * RF_pred + w_lstm * LSTM_pred
    Pesos por defecto: RF=60%, LSTM=40%.
    """

    def __init__(
        self,
        rf_weight: float = ENSEMBLE_WEIGHTS["random_forest"],
        lstm_weight: float = ENSEMBLE_WEIGHTS["lstm"],
    ):
        assert abs(rf_weight + lstm_weight - 1.0) < 1e-6, "Los pesos deben sumar 1.0"
        self.rf_weight = rf_weight
        self.lstm_weight = lstm_weight
        self.rf_model = RandomForestModel()
        self.lstm_model = LSTMModel()
        self._rf_ready = False
        self._lstm_ready = False

    def load_models(self) -> None:
        """Carga ambos modelos desde disco."""
        try:
            self.rf_model._load()
            self._rf_ready = True
        except FileNotFoundError:
            logger.warning("Modelo RF no encontrado — usando solo LSTM")

        try:
            self.lstm_model._load()
            self._lstm_ready = True
        except FileNotFoundError:
            logger.warning("Modelo LSTM no encontrado — usando solo RF")

        if not self._rf_ready and not self._lstm_ready:
            raise RuntimeError("Ningún modelo entrenado encontrado. Ejecute el entrenamiento primero.")

    def predict(
        self,
        X_rf: pd.DataFrame,
        X_lstm: np.ndarray,
    ) -> np.ndarray:
        """
        Combina predicciones RF + LSTM.
        Retorna array (n_samples, FORECAST_HORIZON) con índices 0-100.
        """
        predictions = []

        if self._rf_ready:
            rf_pred = self.rf_model.predict(X_rf)
            predictions.append((self.rf_weight if self._lstm_ready else 1.0, rf_pred))

        if self._lstm_ready:
            lstm_pred = self.lstm_model.predict(X_lstm)
            predictions.append((self.lstm_weight if self._rf_ready else 1.0, lstm_pred))

        if len(predictions) == 1:
            return predictions[0][1]

        # Media ponderada
        total = sum(w * p for w, p in predictions)
        return np.clip(total, 0, 100)

    def predict_province(
        self,
        province: str,
        X_rf: pd.DataFrame,
        X_lstm: np.ndarray,
    ) -> Dict:
        """
        Predicción completa para una provincia: 4 semanas + metadatos.
        """
        combined = self.predict(X_rf, X_lstm)
        latest = combined[-1] if combined.ndim > 1 else combined

        forecast = {f"week_{i+1}": float(round(latest[i], 2)) for i in range(FORECAST_HORIZON)}
        current_risk = float(round(latest[0], 2))

        return {
            "province": province,
            "current_risk_index": current_risk,
            "risk_level": self._classify_risk(current_risk),
            "forecast_4_weeks": forecast,
            "peak_risk": float(round(max(latest), 2)),
            "peak_week": int(np.argmax(latest) + 1),
            "is_epidemic": current_risk >= EPIDEMIC_THRESHOLD,
            "is_alert": current_risk >= ALERT_THRESHOLD,
            "trend": self._calculate_trend(latest),
        }

    def predict_all_provinces(
        self,
        data_by_province: Dict[str, Tuple[pd.DataFrame, np.ndarray]],
    ) -> List[Dict]:
        """Genera predicciones para todas las provincias."""
        results = []
        for province, (X_rf, X_lstm) in data_by_province.items():
            try:
                pred = self.predict_province(province, X_rf, X_lstm)
                results.append(pred)
            except Exception as e:
                logger.error(f"Error prediciendo {province}: {e}")
                results.append(self._empty_prediction(province))

        results.sort(key=lambda x: x["current_risk_index"], reverse=True)
        logger.info(f"Predicciones generadas para {len(results)} provincias")
        return results

    def get_alerts(self, predictions: List[Dict]) -> List[Dict]:
        """Filtra provincias en nivel de alerta o epidemia."""
        return [p for p in predictions if p["is_epidemic"]]

    def get_ensemble_accuracy(self) -> float:
        """Accuracy combinada del ensemble."""
        accuracies = []
        weights = []
        if self._rf_ready:
            accuracies.append(self.rf_model.get_accuracy())
            weights.append(self.rf_weight)
        if self._lstm_ready:
            accuracies.append(self.lstm_model.get_accuracy())
            weights.append(self.lstm_weight)
        if not accuracies:
            return 0.0
        total_weight = sum(weights)
        return float(sum(a * w / total_weight for a, w in zip(accuracies, weights)))

    def export_predictions_csv(self, predictions: List[Dict], encrypt: bool = True) -> Path:
        """Exporta predicciones a CSV (opcionalmente cifrado)."""
        rows = []
        for p in predictions:
            row = {
                "province": p["province"],
                "risk_index_current": p["current_risk_index"],
                "risk_level": p["risk_level"],
                "week_1_forecast": p["forecast_4_weeks"].get("week_1"),
                "week_2_forecast": p["forecast_4_weeks"].get("week_2"),
                "week_3_forecast": p["forecast_4_weeks"].get("week_3"),
                "week_4_forecast": p["forecast_4_weeks"].get("week_4"),
                "peak_risk": p["peak_risk"],
                "peak_week": p["peak_week"],
                "is_epidemic": p["is_epidemic"],
                "trend": p["trend"],
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        Paths.OUTPUTS.mkdir(parents=True, exist_ok=True)

        if encrypt:
            try:
                from src.security import DataEncryptor
                enc = DataEncryptor()
                encrypted = enc.encrypt_dataframe(df)
                out_path = Paths.OUTPUTS / "predictions_latest.enc"
                out_path.write_bytes(encrypted)
                logger.info(f"Predicciones cifradas exportadas: {out_path}")
                return out_path
            except Exception as e:
                logger.warning(f"Cifrado fallido ({e}), exportando CSV plano")

        out_path = Paths.OUTPUTS / "predictions_latest.csv"
        df.to_csv(out_path, index=False)
        return out_path

    @staticmethod
    def _classify_risk(risk: float) -> str:
        if risk < 25:    return "Bajo"
        if risk < 50:    return "Moderado"
        if risk < 65:    return "Alto"
        if risk < 80:    return "Epidemia"
        return "Crítico"

    @staticmethod
    def _calculate_trend(forecast: np.ndarray) -> str:
        if len(forecast) < 2:
            return "Estable"
        slope = np.polyfit(range(len(forecast)), forecast, 1)[0]
        if slope > 2:   return "Ascendente"
        if slope < -2:  return "Descendente"
        return "Estable"

    @staticmethod
    def _empty_prediction(province: str) -> Dict:
        return {
            "province": province,
            "current_risk_index": 0.0,
            "risk_level": "Sin datos",
            "forecast_4_weeks": {f"week_{i+1}": 0.0 for i in range(FORECAST_HORIZON)},
            "peak_risk": 0.0, "peak_week": 1,
            "is_epidemic": False, "is_alert": False, "trend": "Sin datos",
        }
