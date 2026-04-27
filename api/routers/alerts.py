"""
Router de alertas: provincias que superan el umbral epidémico.
GET /alerts → provincias con riesgo >= 65 (umbral epidemia)
"""

from fastapi import APIRouter, Depends, Query
from loguru import logger

from api.auth import require_api_key
from api.schemas import AlertResponse, ProvincePrediction, ForecastWeeks
from config.settings import PROVINCES, EPIDEMIC_THRESHOLD

router = APIRouter(prefix="/alerts", tags=["Alertas Epidemiológicas"])


@router.get(
    "",
    response_model=AlertResponse,
    summary="Provincias en alerta epidémica",
    description=f"""
Retorna todas las provincias cuyo índice de riesgo supera el umbral
epidémico ({EPIDEMIC_THRESHOLD} por defecto, configurable vía parámetro).

Útil para sistemas de alerta temprana y notificaciones automáticas.
    """,
)
async def get_epidemic_alerts(
    threshold: float = Query(
        default=EPIDEMIC_THRESHOLD,
        ge=0, le=100,
        description="Umbral de riesgo para considerar alerta (0-100)",
    ),
    _: str = Depends(require_api_key),
):
    from api.routers.predictions import _generate_prediction
    from api.main import app

    ensemble = app.state.ensemble
    alerts = []

    for province in PROVINCES:
        try:
            pred = _generate_prediction(ensemble, province)
            if pred["current_risk_index"] >= threshold:
                alerts.append(ProvincePrediction(
                    province=province,
                    current_risk_index=pred["current_risk_index"],
                    risk_level=pred["risk_level"],
                    forecast_4_weeks=ForecastWeeks(**pred["forecast_4_weeks"]),
                    peak_risk=pred["peak_risk"],
                    peak_week=pred["peak_week"],
                    is_epidemic=pred["is_epidemic"],
                    is_alert=pred["is_alert"],
                    trend=pred["trend"],
                ))
        except Exception as e:
            logger.error(f"Error evaluando alerta {province}: {e}")

    alerts.sort(key=lambda x: x.current_risk_index, reverse=True)
    logger.info(f"Alertas activas ({threshold}+): {len(alerts)} provincias")

    return AlertResponse(total_alerts=len(alerts), alerts=alerts)


@router.get(
    "/critical",
    summary="Provincias en nivel crítico (riesgo > 80)",
    tags=["Alertas Epidemiológicas"],
)
async def get_critical_alerts(_: str = Depends(require_api_key)):
    """Acceso rápido a provincias en nivel crítico (riesgo > 80)."""
    from api.routers.predictions import _generate_prediction
    from api.main import app
    ensemble = app.state.ensemble

    critical = []
    for province in PROVINCES:
        try:
            pred = _generate_prediction(ensemble, province)
            if pred["current_risk_index"] >= 80:
                critical.append({
                    "province": province,
                    "risk_index": pred["current_risk_index"],
                    "trend": pred["trend"],
                    "peak_week": pred["peak_week"],
                })
        except Exception:
            pass

    return {
        "level": "Crítico",
        "threshold": 80,
        "count": len(critical),
        "provinces": sorted(critical, key=lambda x: x["risk_index"], reverse=True),
    }
