"""
Orquestador de entrenamiento: preprocesa datos, entrena RF + LSTM,
evalúa el ensemble y dispara re-entrenamiento automático si la
accuracy cae por debajo del umbral configurado (88% por defecto).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
from sklearn.model_selection import train_test_split
from loguru import logger

from config.settings import FORECAST_HORIZON, MODEL_RETRAIN_THRESHOLD, Paths
from src.data.preprocessing import DataPreprocessor
from .random_forest import RandomForestModel
from .lstm import LSTMModel
from .ensemble import EnsembleModel


class ModelTrainer:
    """Pipeline completo de entrenamiento y evaluación."""

    def __init__(self, retrain_threshold: float = MODEL_RETRAIN_THRESHOLD):
        self.retrain_threshold = retrain_threshold
        self.preprocessor = DataPreprocessor()
        self.rf_model = RandomForestModel()
        self.lstm_model = LSTMModel()
        self.ensemble = EnsembleModel()
        self.training_report: Dict = {}

    def full_training_pipeline(
        self,
        raw_df: pd.DataFrame,
        test_size: float = 0.15,
        val_size: float = 0.15,
    ) -> Dict:
        """
        Pipeline completo:
        1. Preprocesar datos
        2. Split train/val/test (respetando orden temporal)
        3. Entrenar RF y LSTM
        4. Evaluar ensemble
        5. Guardar modelos y reporte
        """
        logger.info("=" * 60)
        logger.info("INICIO PIPELINE DE ENTRENAMIENTO — DENGUE-RD")
        logger.info("=" * 60)
        start_time = datetime.now()

        # 1. Preprocesar
        df_processed = self.preprocessor.fit_transform(raw_df)

        # 2. Split temporal (sin shuffle para respetar series de tiempo)
        n = len(df_processed)
        test_n = int(n * test_size)
        val_n = int(n * val_size)
        train_n = n - val_n - test_n

        df_train = df_processed.iloc[:train_n]
        df_val   = df_processed.iloc[train_n:train_n + val_n]
        df_test  = df_processed.iloc[train_n + val_n:]

        logger.info(f"Split: {train_n} train | {val_n} val | {test_n} test")

        # 3. Preparar features para RF (tabular)
        X_train_rf, y_train_rf = self._prepare_rf_targets(df_train)
        X_val_rf,   y_val_rf   = self._prepare_rf_targets(df_val)
        X_test_rf,  y_test_rf  = self._prepare_rf_targets(df_test)

        # 4. Preparar secuencias para LSTM
        X_train_lstm, y_train_lstm = self.preprocessor.prepare_lstm_sequences(df_train)
        X_val_lstm,   y_val_lstm   = self.preprocessor.prepare_lstm_sequences(df_val)

        # 5. Entrenar Random Forest
        logger.info("Entrenando Random Forest...")
        rf_metrics = self.rf_model.train(X_train_rf, y_train_rf, X_val_rf, y_val_rf)
        self.rf_model.save()

        # 6. Construir y entrenar LSTM
        if len(X_train_lstm) > 0:
            logger.info("Construyendo y entrenando LSTM...")
            self.lstm_model.build(X_train_lstm.shape[1:])
            lstm_metrics = self.lstm_model.train(X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm)
            self.lstm_model.save()
        else:
            lstm_metrics = {"accuracy_pct": 0.0}
            logger.warning("Datos insuficientes para LSTM — saltado")

        # 7. Evaluar ensemble sobre test set
        self.ensemble._rf_ready = True
        self.ensemble._lstm_ready = len(X_train_lstm) > 0
        ensemble_accuracy = self.ensemble.get_ensemble_accuracy()

        # 8. Generar reporte
        duration = (datetime.now() - start_time).total_seconds()
        self.training_report = {
            "timestamp": datetime.now().isoformat(),
            "train_samples": train_n,
            "val_samples": val_n,
            "test_samples": test_n,
            "rf_metrics": rf_metrics,
            "lstm_metrics": lstm_metrics,
            "ensemble_accuracy_pct": ensemble_accuracy,
            "meets_threshold": ensemble_accuracy >= (self.retrain_threshold * 100),
            "duration_seconds": round(duration, 1),
        }

        self._save_training_report()
        self._log_summary()

        return self.training_report

    def check_and_retrain(self, current_df: pd.DataFrame) -> bool:
        """
        Verifica si la accuracy actual cae por debajo del umbral
        y re-entrena automáticamente si es necesario.
        Retorna True si se re-entrenó.
        """
        try:
            self.ensemble.load_models()
            current_accuracy = self.ensemble.get_ensemble_accuracy() / 100
        except Exception:
            current_accuracy = 0.0

        logger.info(f"Accuracy actual: {current_accuracy:.2%} | Umbral: {self.retrain_threshold:.2%}")

        if current_accuracy < self.retrain_threshold:
            logger.warning(f"Accuracy {current_accuracy:.2%} < umbral {self.retrain_threshold:.2%} — re-entrenando")
            self.full_training_pipeline(current_df)
            return True

        logger.info("Accuracy suficiente — no se requiere re-entrenamiento")
        return False

    def _prepare_rf_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepara X e y multi-output para RF (4 semanas futuras como targets)."""
        X, _ = self.preprocessor.prepare_rf_features(df)

        # Crear targets desplazados 1-4 semanas hacia adelante por provincia
        target_cols = {}
        for week in range(1, FORECAST_HORIZON + 1):
            col_name = f"target_week_{week}"
            target_cols[col_name] = df.groupby("province")["outbreak_risk_index"].shift(-week)

        y = pd.DataFrame(target_cols, index=df.index).ffill().fillna(0)

        # Alinear índices y eliminar NaN
        valid_mask = y.notna().all(axis=1) & X.notna().all(axis=1)
        return X[valid_mask], y[valid_mask]

    def _save_training_report(self) -> None:
        import json
        report_path = Paths.MODELS / "training_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.training_report, f, indent=2, default=str)
        logger.info(f"Reporte guardado: {report_path}")

    def _log_summary(self) -> None:
        r = self.training_report
        logger.info("=" * 60)
        logger.info("RESUMEN DE ENTRENAMIENTO")
        logger.info(f"  RF  — MAE: {r['rf_metrics'].get('val_mae', '?'):.3f}")
        logger.info(f"  LSTM— MAE: {r['lstm_metrics'].get('val_mae', '?')}")
        logger.info(f"  Ensemble Accuracy: {r['ensemble_accuracy_pct']:.1f}%")
        meets = "✓ SÍ" if r["meets_threshold"] else "✗ NO"
        logger.info(f"  Cumple umbral 88%: {meets}")
        logger.info(f"  Duración: {r['duration_seconds']}s")
        logger.info("=" * 60)
