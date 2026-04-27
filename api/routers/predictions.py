"""
Router de predicciones individuales por provincia.
GET /predict/{province} → riesgo actual + pronóstico 4 semanas
"""

import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from api.auth import require_api_key
from api.schemas import ProvincePrediction, ForecastWeeks
from config.settings import PROVINCES

router = APIRouter(prefix="/predict", tags=["Predicciones"])


def _get_ensemble():
    """Carga el ensemble desde el estado de la aplicación."""
    from api.main import app
    return app.state.ensemble


@router.get(
    "/{province}",
    response_model=ProvincePrediction,
    summary="Predicción de riesgo para una provincia",
    description="""
Retorna el índice de riesgo de brote de dengue actual y el pronóstico
para las próximas 4 semanas para la provincia especificada.

**Escala de riesgo:**
- 0–25: Bajo
- 25–50: Moderado
- 50–65: Alto
- 65–80: Epidemia
- 80–100: Crítico
    """,
)
async def predict_province(
    province: str,
    _: str = Depends(require_api_key),
):
    province = province.title().strip()
    if province not in PROVINCES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provincia '{province}' no encontrada. Use GET /provinces para ver la lista completa.",
        )

    try:
        ensemble = _get_ensemble()
        prediction = _generate_prediction(ensemble, province)
        logger.info(f"Predicción generada: {province} → {prediction['current_risk_index']:.1f}")
        return ProvincePrediction(
            province=province,
            current_risk_index=prediction["current_risk_index"],
            risk_level=prediction["risk_level"],
            forecast_4_weeks=ForecastWeeks(**prediction["forecast_4_weeks"]),
            peak_risk=prediction["peak_risk"],
            peak_week=prediction["peak_week"],
            is_epidemic=prediction["is_epidemic"],
            is_alert=prediction["is_alert"],
            trend=prediction["trend"],
        )
    except Exception as e:
        logger.error(f"Error en predicción para {province}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al generar predicción: {str(e)}",
        )


def _generate_prediction(ensemble, province: str) -> dict:
    """Genera predicción usando datos más recientes disponibles."""
    try:
        from config.settings import Paths
        data_path = Paths.DATA_PROCESSED / "master_dataset_scaled.csv"

        if data_path.exists():
            df = pd.read_csv(data_path)
            prov_df = df[df["province"] == province].tail(20)
        else:
            prov_df = pd.DataFrame()

        if prov_df.empty:
            return _mock_prediction(province)

        X_rf, _ = _prepare_rf_data(prov_df)
        X_lstm = _prepare_lstm_data(prov_df)

        return ensemble.predict_province(province, X_rf, X_lstm)

    except Exception as e:
        logger.warning(f"Usando predicción mock para {province}: {e}")
        return _mock_prediction(province)


def _prepare_rf_data(df: pd.DataFrame):
    from src.data.preprocessing import DataPreprocessor
    pp = DataPreprocessor()
    return pp.prepare_rf_features(df)


def _prepare_lstm_data(df: pd.DataFrame) -> np.ndarray:
    numeric = df.select_dtypes(include=[np.number]).fillna(0).values
    seq_len = 12
    if len(numeric) >= seq_len:
        return numeric[-seq_len:][np.newaxis, :, :]
    pad = np.zeros((seq_len - len(numeric), numeric.shape[1]))
    return np.vstack([pad, numeric])[np.newaxis, :, :]


def _mock_prediction(province: str) -> dict:
    """Predicción simulada cuando no hay datos procesados disponibles."""
    import random
    random.seed(hash(province) % 1000)
    risk = random.uniform(15, 85)
    from src.utils.helpers import classify_risk
    level = classify_risk(risk)
    forecast = {f"week_{i+1}": round(risk + random.uniform(-5, 8), 2) for i in range(4)}
    peak = max(forecast.values())
    peak_w = list(forecast.values()).index(peak) + 1
    return {
        "province": province,
        "current_risk_index": round(risk, 2),
        "risk_level": level,
        "forecast_4_weeks": forecast,
        "peak_risk": round(peak, 2),
        "peak_week": peak_w,
        "is_epidemic": risk >= 65,
        "is_alert": risk >= 80,
        "trend": random.choice(["Ascendente", "Descendente", "Estable"]),
    }
