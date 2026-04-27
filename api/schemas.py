"""Esquemas Pydantic para request/response de la API."""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
from datetime import datetime


class ForecastWeeks(BaseModel):
    week_1: float = Field(..., ge=0, le=100, description="Riesgo semana 1")
    week_2: float = Field(..., ge=0, le=100, description="Riesgo semana 2")
    week_3: float = Field(..., ge=0, le=100, description="Riesgo semana 3")
    week_4: float = Field(..., ge=0, le=100, description="Riesgo semana 4")


class ProvincePrediction(BaseModel):
    province: str
    current_risk_index: float = Field(..., ge=0, le=100)
    risk_level: str
    forecast_4_weeks: ForecastWeeks
    peak_risk: float
    peak_week: int = Field(..., ge=1, le=4)
    is_epidemic: bool
    is_alert: bool
    trend: str
    generated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "province": "Santo Domingo",
                "current_risk_index": 72.5,
                "risk_level": "Epidemia",
                "forecast_4_weeks": {"week_1": 72.5, "week_2": 76.1, "week_3": 71.3, "week_4": 68.0},
                "peak_risk": 76.1,
                "peak_week": 2,
                "is_epidemic": True,
                "is_alert": False,
                "trend": "Ascendente",
                "generated_at": "2026-04-22T10:00:00",
            }
        }


class ProvinceSummary(BaseModel):
    province: str
    risk_index: float
    risk_level: str
    is_epidemic: bool
    trend: str


class AllProvincesResponse(BaseModel):
    total_provinces: int
    epidemic_count: int
    alert_count: int
    average_risk: float
    highest_risk_province: str
    predictions: List[ProvinceSummary]
    generated_at: datetime = Field(default_factory=datetime.now)


class AlertResponse(BaseModel):
    total_alerts: int
    alerts: List[ProvincePrediction]
    generated_at: datetime = Field(default_factory=datetime.now)


class UploadResponse(BaseModel):
    status: str
    records_processed: int
    message: str
    retrain_triggered: bool = False


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    ensemble_accuracy: float
    uptime_seconds: float


class WeeklyDataUpload(BaseModel):
    province: str
    year: int = Field(..., ge=2000, le=2100)
    week: int = Field(..., ge=1, le=53)
    cases: int = Field(..., ge=0)
    deaths: Optional[int] = Field(default=0, ge=0)
    rainfall_mm: Optional[float] = Field(default=None, ge=0)
    temp_avg_c: Optional[float] = Field(default=None)
    humidity_pct: Optional[float] = Field(default=None, ge=0, le=100)

    @validator("province")
    def validate_province(cls, v):
        from config.settings import PROVINCES
        if v not in PROVINCES:
            raise ValueError(f"Provincia '{v}' no válida. Use GET /provinces para ver la lista.")
        return v
