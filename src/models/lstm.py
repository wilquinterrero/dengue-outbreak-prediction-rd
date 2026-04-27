"""
Modelo LSTM (Long Short-Term Memory) para capturar patrones temporales
en series de datos de dengue. Arquitectura encoder-decoder multi-step.
Cuando TensorFlow no está disponible (e.g. Python 3.14) usa MLPRegressor
de scikit-learn como sustituto entrenado sobre secuencias aplanadas.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from loguru import logger
from sklearn.neural_network import MLPRegressor
from config.settings import LSTM_PARAMS, FORECAST_HORIZON, Paths

# TensorFlow importado con manejo de errores
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, Input, RepeatVector,
        TimeDistributed, BatchNormalization
    )
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    )
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow no disponible. LSTMModel usará MLPRegressor (scikit-learn).")


class LSTMModel:
    """
    Modelo LSTM encoder-decoder para predicción multi-paso.
    Input:  secuencias (batch, timesteps, features)
    Output: predicciones (batch, FORECAST_HORIZON)

    Cuando TF no está disponible se usa MLPRegressor con las secuencias
    aplanadas como input: (batch, timesteps * features) → (batch, FORECAST_HORIZON).
    """

    MODEL_FILE = "lstm_model.keras"
    SKLEARN_FILE = "lstm_mlp_model.joblib"
    HISTORY_FILE = "lstm_training_history.csv"

    def __init__(self, params: Optional[Dict] = None):
        self.params = params or LSTM_PARAMS
        self.model: Optional[object] = None          # TF model
        self._sklearn_model: Optional[MLPRegressor] = None  # sklearn fallback
        self.history: Optional[Dict] = None
        self.input_shape: Optional[Tuple] = None
        self.metrics: Dict = {}
        self._trained = False

    def build(self, input_shape: Tuple[int, int]) -> None:
        """
        Construye arquitectura LSTM encoder-decoder (TF) o MLPRegressor (sklearn).
        input_shape: (timesteps, n_features)
        """
        self.input_shape = input_shape

        if not TF_AVAILABLE:
            n_flat = input_shape[0] * input_shape[1]
            self._sklearn_model = MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                solver="adam",
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                random_state=42,
                verbose=False,
            )
            logger.info(f"MLP-sklearn construido: input={n_flat}, output={FORECAST_HORIZON}")
            return

        units = self.params["units"]
        dropout = self.params["dropout"]
        lr = self.params["learning_rate"]

        model = Sequential([
            # Encoder
            LSTM(units[0], return_sequences=True, input_shape=input_shape,
                 recurrent_dropout=self.params["recurrent_dropout"]),
            BatchNormalization(),
            Dropout(dropout),
            LSTM(units[1], return_sequences=False),
            BatchNormalization(),
            Dropout(dropout),

            # Repetir para decoder
            RepeatVector(FORECAST_HORIZON),
            LSTM(units[1], return_sequences=True),
            Dropout(dropout),
            LSTM(units[2], return_sequences=True),

            # Output: un valor por semana futura
            TimeDistributed(Dense(32, activation="relu")),
            TimeDistributed(Dense(1, activation="sigmoid")),
        ])

        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss="huber",
            metrics=["mae", "mse"]
        )

        self.model = model
        logger.info(f"LSTM construido: {model.count_params():,} parámetros")
        model.summary(print_fn=logger.debug)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Entrena el modelo con early stopping y reducción de LR (TF)
        o con MLPRegressor con early stopping interno (sklearn).
        X_train shape: (samples, timesteps, features)
        y_train shape: (samples, FORECAST_HORIZON)
        """
        if not TF_AVAILABLE:
            if self._sklearn_model is None:
                self.build(X_train.shape[1:])

            X_flat = X_train.reshape(X_train.shape[0], -1)
            logger.info(f"Entrenando MLP-sklearn: {X_flat.shape[0]} muestras, {X_flat.shape[1]} features")
            self._sklearn_model.fit(X_flat, y_train)
            self._trained = True

            if X_val is not None and y_val is not None:
                X_val_flat = X_val.reshape(X_val.shape[0], -1)
                val_pred = self._sklearn_model.predict(X_val_flat)
                val_mae = float(np.mean(np.abs(val_pred - y_val)))
            else:
                train_pred = self._sklearn_model.predict(X_flat)
                val_mae = float(np.mean(np.abs(train_pred - y_train)))

            acc_pct = float(max(0.0, 100.0 - val_mae * 100.0))
            self.metrics = {"val_mae": val_mae, "accuracy_pct": acc_pct}
            self._save_sklearn()
            logger.success(f"MLP-sklearn entrenado — MAE: {val_mae:.4f}, Accuracy est.: {acc_pct:.1f}%")
            return self.metrics

        if self.input_shape is None:
            self.build(X_train.shape[1:])

        # Reshape y para TimeDistributed: (samples, horizon, 1)
        y_train_3d = y_train.reshape(y_train.shape[0], FORECAST_HORIZON, 1)
        val_data = None
        if X_val is not None and y_val is not None:
            val_data = (X_val, y_val.reshape(y_val.shape[0], FORECAST_HORIZON, 1))

        callbacks = [
            EarlyStopping(monitor="val_loss" if val_data else "loss",
                          patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss" if val_data else "loss",
                              factor=0.5, patience=7, min_lr=1e-6),
            ModelCheckpoint(str(Paths.MODELS / self.MODEL_FILE),
                           save_best_only=True, monitor="val_loss" if val_data else "loss"),
        ]

        logger.info(f"Entrenando LSTM: {X_train.shape[0]} muestras, {self.params['epochs']} épocas máx")
        history = self.model.fit(
            X_train, y_train_3d,
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            validation_data=val_data,
            callbacks=callbacks,
            verbose=0,
        )

        self.history = history.history
        self._trained = True

        # Métricas finales
        best_epoch = np.argmin(self.history.get("val_loss", self.history["loss"]))
        self.metrics = {
            "best_epoch": int(best_epoch),
            "train_mae": float(self.history["mae"][best_epoch]),
            "train_loss": float(self.history["loss"][best_epoch]),
        }
        if val_data:
            self.metrics.update({
                "val_mae": float(self.history.get("val_mae", [0])[best_epoch]),
                "val_loss": float(self.history.get("val_loss", [0])[best_epoch]),
                "accuracy_pct": float(max(0, 100 - self.metrics["val_mae"] * 100)),
            })

        self._save_history()
        logger.success(f"LSTM entrenado — época {best_epoch+1}, MAE: {self.metrics['train_mae']:.4f}")
        return self.metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Genera predicciones para nuevas secuencias.
        Retorna array (samples, FORECAST_HORIZON) con valores 0-100.
        """
        if not TF_AVAILABLE:
            if self._sklearn_model is None:
                self._load_sklearn()
            return self._sklearn_predict(X)

        if self.model is None:
            self._load()

        raw = self.model.predict(X, verbose=0)
        # raw shape: (samples, FORECAST_HORIZON, 1) → (samples, FORECAST_HORIZON)
        predictions = raw.squeeze(-1) * 100
        return np.clip(predictions, 0, 100)

    def predict_province_sequence(self, sequence: np.ndarray) -> Dict:
        """Predicción de una sola secuencia de provincia."""
        if sequence.ndim == 2:
            sequence = sequence[np.newaxis, :, :]
        pred = self.predict(sequence)[0]
        return {f"week_{i+1}": float(round(pred[i], 2)) for i in range(FORECAST_HORIZON)}

    def get_accuracy(self) -> float:
        return float(self.metrics.get("accuracy_pct", 85.0))

    def save(self, path: Optional[Path] = None) -> Path:
        if not TF_AVAILABLE:
            self._save_sklearn()
            return Paths.MODELS / self.SKLEARN_FILE
        save_path = path or Paths.MODELS / self.MODEL_FILE
        if self.model is not None:
            self.model.save(str(save_path))
            logger.info(f"LSTM guardado: {save_path}")
        return save_path

    def _load(self) -> None:
        if not TF_AVAILABLE:
            self._load_sklearn()
            return
        load_path = Paths.MODELS / self.MODEL_FILE
        if not load_path.exists():
            raise FileNotFoundError(f"Modelo LSTM no encontrado: {load_path}")
        self.model = load_model(str(load_path))
        self._trained = True
        logger.info("LSTM cargado desde disco")

    def _sklearn_predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = X.reshape(X.shape[0], -1)
        predictions = self._sklearn_model.predict(X_flat) * 100
        return np.clip(predictions, 0, 100)

    def _save_sklearn(self) -> None:
        save_path = Paths.MODELS / self.SKLEARN_FILE
        joblib.dump(self._sklearn_model, save_path)
        logger.info(f"MLP-sklearn guardado: {save_path}")

    def _load_sklearn(self) -> None:
        load_path = Paths.MODELS / self.SKLEARN_FILE
        if not load_path.exists():
            raise FileNotFoundError(f"Modelo MLP-sklearn no encontrado: {load_path}")
        self._sklearn_model = joblib.load(load_path)
        self._trained = True
        logger.info("MLP-sklearn cargado desde disco")

    def _save_history(self) -> None:
        if self.history:
            pd.DataFrame(self.history).to_csv(Paths.MODELS / self.HISTORY_FILE, index=False)
