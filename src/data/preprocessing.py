"""
Preprocesamiento y feature engineering para el modelo de predicción.
Incluye limpieza, normalización, creación de lag features y secuencias LSTM.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer
import joblib
from loguru import logger
from config.settings import ALL_FEATURES, TARGET_COLUMN, FORECAST_HORIZON, Paths


class DataPreprocessor:
    """Limpieza, transformación y feature engineering para dengue-RD."""

    LAG_WEEKS = [1, 2, 3, 4, 8, 12]
    ROLLING_WINDOWS = [4, 8, 12]

    def __init__(self):
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        self.feature_columns: List[str] = []
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajusta transformaciones y las aplica al DataFrame de entrenamiento."""
        logger.info("Iniciando preprocesamiento de datos")
        df = self._clean_data(df)
        df = self._create_lag_features(df)
        df = self._create_rolling_features(df)
        df = self._create_temporal_features(df)
        df = self._handle_missing_values(df, fit=True)
        df = self._scale_features(df, fit=True)
        self._fitted = True
        self._save_preprocessors()
        logger.success(f"Preprocesamiento completo: {df.shape}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica transformaciones previamente ajustadas a nuevos datos."""
        if not self._fitted:
            self._load_preprocessors()
        df = self._clean_data(df)
        df = self._create_lag_features(df)
        df = self._create_rolling_features(df)
        df = self._create_temporal_features(df)
        df = self._handle_missing_values(df, fit=False)
        df = self._scale_features(df, fit=False)
        return df

    def prepare_lstm_sequences(
        self, df: pd.DataFrame, sequence_length: int = 12
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera secuencias temporales para el modelo LSTM.
        Retorna (X, y) donde X.shape = (samples, timesteps, features).
        """
        feature_cols = [c for c in df.columns
                       if c not in [TARGET_COLUMN, "province", "date", "week", "year", "month"]]
        self.feature_columns = feature_cols

        X_seqs, y_seqs = [], []
        for province in df["province"].unique():
            prov_df = df[df["province"] == province].sort_values("date" if "date" in df.columns else "week")
            feat_arr = prov_df[feature_cols].values
            target_arr = prov_df[TARGET_COLUMN].values

            for i in range(len(feat_arr) - sequence_length - FORECAST_HORIZON + 1):
                X_seqs.append(feat_arr[i:i + sequence_length])
                y_seqs.append(target_arr[i + sequence_length:i + sequence_length + FORECAST_HORIZON])

        return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)

    def prepare_rf_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepara features tabulares para Random Forest."""
        feature_cols = [c for c in df.columns
                       if c not in [TARGET_COLUMN, "province", "date"]
                       and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
        X = df[feature_cols].fillna(0)
        y = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else None
        return X, y

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

    # --- MÉTODOS PRIVADOS ---

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Eliminar duplicados
        dup_cols = ["province", "year", "week"] if all(c in df.columns for c in ["province", "year", "week"]) else None
        if dup_cols:
            before = len(df)
            df = df.drop_duplicates(subset=dup_cols)
            logger.debug(f"Duplicados eliminados: {before - len(df)}")

        # Valores negativos → 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].clip(lower=0)

        # Outliers: winsorizing al 99.5%
        for col in ["cases", "rainfall_mm", "incidence_rate_100k"]:
            if col in df.columns:
                cap = df[col].quantile(0.995)
                df[col] = df[col].clip(upper=cap)

        # Crear fecha si no existe
        if "date" not in df.columns and "year" in df.columns and "week" in df.columns:
            df["date"] = pd.to_datetime(
                df["year"].astype(str) + df["week"].astype(str).str.zfill(2) + "1",
                format="%Y%W%w"
            )

        return df

    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features de lag por provincia para capturar autocorrelación."""
        df = df.sort_values(["province", "date"] if "date" in df.columns else ["province", "year", "week"])

        lag_source_cols = ["cases", "outbreak_risk_index", "rainfall_mm", "temp_avg_c"]
        for col in lag_source_cols:
            if col not in df.columns:
                continue
            for lag in self.LAG_WEEKS:
                df[f"{col}_lag{lag}w"] = df.groupby("province")[col].shift(lag)

        return df

    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea medias móviles y desviaciones estándar rolling."""
        roll_cols = ["cases", "rainfall_mm", "temp_avg_c", "humidity_pct"]
        for col in roll_cols:
            if col not in df.columns:
                continue
            for window in self.ROLLING_WINDOWS:
                grp = df.groupby("province")[col]
                df[f"{col}_roll{window}w_mean"] = grp.transform(lambda x: x.rolling(window, min_periods=1).mean())
                df[f"{col}_roll{window}w_std"] = grp.transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))

        return df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extrae features cíclicas de temporalidad."""
        if "week" in df.columns:
            df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
            df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)
            df["is_rainy_season"] = df["week"].between(18, 44).astype(int)

        if "date" in df.columns:
            dates = pd.to_datetime(df["date"])
            df["month"] = dates.dt.month
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        return df

    def _handle_missing_values(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return df
        if fit:
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.imputer.transform(df[numeric_cols])
        return df

    def _scale_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        exclude = ["province", "date", "year", "week", "month", TARGET_COLUMN]
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

        if fit:
            df[feature_cols] = self.feature_scaler.fit_transform(df[feature_cols].fillna(0))
            if TARGET_COLUMN in df.columns:
                df[[TARGET_COLUMN]] = self.target_scaler.fit_transform(df[[TARGET_COLUMN]])
        else:
            df[feature_cols] = self.feature_scaler.transform(df[feature_cols].fillna(0))
            if TARGET_COLUMN in df.columns:
                df[[TARGET_COLUMN]] = self.target_scaler.transform(df[[TARGET_COLUMN]])

        return df

    def _save_preprocessors(self) -> None:
        Paths.MODELS.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.feature_scaler, Paths.MODELS / "feature_scaler.pkl")
        joblib.dump(self.target_scaler, Paths.MODELS / "target_scaler.pkl")
        joblib.dump(self.imputer, Paths.MODELS / "imputer.pkl")
        logger.debug("Preprocesadores guardados")

    def _load_preprocessors(self) -> None:
        self.feature_scaler = joblib.load(Paths.MODELS / "feature_scaler.pkl")
        self.target_scaler = joblib.load(Paths.MODELS / "target_scaler.pkl")
        self.imputer = joblib.load(Paths.MODELS / "imputer.pkl")
        self._fitted = True
        logger.debug("Preprocesadores cargados desde disco")
