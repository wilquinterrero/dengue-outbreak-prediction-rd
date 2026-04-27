"""
Router de provincias: resumen de riesgo para todas las 32 provincias.
GET /provinces → lista completa con índice de riesgo actual
"""

from fastapi import APIRouter, Depends
from loguru import logger

from api.auth import require_api_key
from api.schemas import AllProvincesResponse, ProvinceSummary
from config.settings import PROVINCES
from src.utils.helpers import classify_risk

router = APIRouter(prefix="/provinces", tags=["Provincias"])


@router.get(
    "",
    response_model=AllProvincesResponse,
    summary="Resumen de riesgo de todas las provincias",
    description="Retorna el índice de riesgo actual para las 32 provincias de la República Dominicana.",
)
async def get_all_provinces(_: str = Depends(require_api_key)):
    from api.routers.predictions import _generate_prediction
    from api.main import app

    ensemble = app.state.ensemble
    summaries = []

    for province in PROVINCES:
        try:
            pred = _generate_prediction(ensemble, province)
            summaries.append(ProvinceSummary(
                province=province,
                risk_index=pred["current_risk_index"],
                risk_level=pred["risk_level"],
                is_epidemic=pred["is_epidemic"],
                trend=pred["trend"],
            ))
        except Exception as e:
            logger.warning(f"Error sumario {province}: {e}")
            summaries.append(ProvinceSummary(
                province=province, risk_index=0.0,
                risk_level="Sin datos", is_epidemic=False, trend="Sin datos",
            ))

    summaries.sort(key=lambda x: x.risk_index, reverse=True)
    epidemic_count = sum(1 for s in summaries if s.is_epidemic)
    alert_count = sum(1 for s in summaries if s.risk_index >= 80)
    avg_risk = sum(s.risk_index for s in summaries) / len(summaries) if summaries else 0
    highest = summaries[0].province if summaries else "N/A"

    return AllProvincesResponse(
        total_provinces=len(summaries),
        epidemic_count=epidemic_count,
        alert_count=alert_count,
        average_risk=round(avg_risk, 2),
        highest_risk_province=highest,
        predictions=summaries,
    )


@router.get(
    "/list",
    summary="Lista de nombres de provincias",
    tags=["Provincias"],
)
async def list_province_names(_: str = Depends(require_api_key)):
    return {"provinces": PROVINCES, "total": len(PROVINCES)}
