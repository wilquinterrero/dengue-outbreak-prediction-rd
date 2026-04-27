"""
Modelo Random Forest para predicción de brotes de dengue.
Predice el índice de riesgo 0-100 por provincia para 4 semanas adelante.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from loguru import logger
from config.settings import RANDOM_FOREST_PARAMS, FORECAST_HORIZON, Paths


class RandomForestModel:
    """
    Modelo Random Forest multi-output para predicción 4 semanas.
    Utiliza TimeSeriesSplit para validación cronológica.
    """

    MODEL_FILE = "random_forest_model.pkl"
    FEATURE_IMPORTANCE_FILE = "rf_feature_importance.csv"

    def __init__(self, params: Optional[Dict] = None):
        base_rf = RandomForestRegressor(**(params or RANDOM_FOREST_PARAMS))
        self.model = MultiOutputRegressor(base_rf, n_jobs=-1)
        self.feature_names: List[str] = []
        self.metrics: Dict = {}
        self._trained = False

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Entrena el modelo con validación cronológica.
        y_train debe tener FORECAST_HORIZON columnas (una por semana futura).
        """
        self.feature_names = list(X_train.columns)
        logger.info(f"Entrenando Random Forest: {X_train.shape[0]} muestras, {X_train.shape[1]} features")

        # Validación cruzada con series temporales
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
            self.model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            pred = self.model.predict(X_train.iloc[val_idx])
            mae = mean_absolute_error(y_train.iloc[val_idx], pred)
            cv_scores.append(mae)
            logger.debug(f"  Fold {fold+1}/5 MAE: {mae:.3f}")

        # Entrenamiento final sobre todos los datos
        self.model.fit(X_train, y_train)
        self._trained = True

        train_pred = self.model.predict(X_train)
        self.metrics = {
            "train_mae": mean_absolute_error(y_train, train_pred),
            "train_rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
            "train_r2": r2_score(y_train, train_pred),
            "cv_mae_mean": np.mean(cv_scores),
            "cv_mae_std": np.std(cv_scores),
        }

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            self.metrics.update({
                "val_mae": mean_absolute_error(y_val, val_pred),
                "val_rmse": np.sqrt(mean_squared_error(y_val, val_pred)),
                "val_r2": r2_score(y_val, val_pred),
                "accuracy_pct": max(0, 100 - mean_absolute_error(y_val, val_pred)),
            })

        self._save_feature_importance()
        logger.success(f"RF entrenado — CV MAE: {self.metrics['cv_mae_mean']:.3f} ± {self.metrics['cv_mae_std']:.3f}")
        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predice índice de riesgo para 4 semanas.
        Retorna array shape (n_samples, FORECAST_HORIZON).
        """
        if not self._trained:
            self._load()
        predictions = self.model.predict(X)
        return np.clip(predictions, 0, 100)

    def predict_province(self, X_province: pd.DataFrame) -> Dict:
        """Predicción formateada para una provincia específica."""
        pred = self.predict(X_province)
        latest = pred[-1] if len(pred) > 0 else pred[0]
        return {
            f"week_{i+1}": float(round(latest[i], 2))
            for i in range(FORECAST_HORIZON)
        }

    def get_accuracy(self) -> float:
        """Retorna accuracy estimada (100 - MAE normalizado)."""
        if "val_r2" in self.metrics:
            return float(max(0, self.metrics["val_r2"] * 100))
        return float(max(0, 100 - self.metrics.get("cv_mae_mean", 12)))

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Retorna las N features más importantes."""
        importances = []
        for estimator in self.model.estimators_:
            importances.append(estimator.feature_importances_)
        mean_importance = np.mean(importances, axis=0)
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": mean_importance
        }).sort_values("importance", ascending=False)
        return df.head(top_n)

    def save(self, path: Optional[Path] = None) -> Path:
        save_path = path or Paths.MODELS / self.MODEL_FILE
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
        }, save_path)
        logger.info(f"Modelo RF guardado: {save_path}")
        return save_path

    def _load(self) -> None:
        load_path = Paths.MODELS / self.MODEL_FILE
        if not load_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {load_path}. Entrene primero el modelo.")
        data = joblib.load(load_path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.metrics = data["metrics"]
        self._trained = True
        logger.info("Modelo RF cargado desde disco")

    def _save_feature_importance(self) -> None:
        try:
            fi_df = self.get_feature_importance(top_n=30)
            fi_path = Paths.MODELS / self.FEATURE_IMPORTANCE_FILE
            fi_df.to_csv(fi_path, index=False)
        except Exception as e:
            logger.warning(f"No se pudo guardar feature importance: {e}")
