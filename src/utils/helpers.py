"""Utilidades generales del sistema."""

from datetime import datetime, timedelta
from typing import Tuple


RISK_COLORS = {
    "Bajo":      "#28a745",
    "Moderado":  "#ffc107",
    "Alto":      "#fd7e14",
    "Epidemia":  "#dc3545",
    "Crítico":   "#6f0000",
    "Sin datos": "#6c757d",
}

RISK_EMOJIS = {
    "Bajo":      "🟢",
    "Moderado":  "🟡",
    "Alto":      "🟠",
    "Epidemia":  "🔴",
    "Crítico":   "⚫",
    "Sin datos": "⚪",
}


def risk_color(level: str) -> str:
    return RISK_COLORS.get(level, "#6c757d")


def risk_emoji(level: str) -> str:
    return RISK_EMOJIS.get(level, "⚪")


def format_risk_badge(risk_index: float, level: str) -> str:
    color = risk_color(level)
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;">{level} ({risk_index:.1f})</span>'


def week_to_date(year: int, week: int) -> datetime:
    """Convierte año + semana ISO a fecha de inicio de semana."""
    return datetime.strptime(f"{year}-W{week:02d}-1", "%Y-W%W-%w")


def classify_risk(risk_index: float) -> str:
    if risk_index < 25:   return "Bajo"
    if risk_index < 50:   return "Moderado"
    if risk_index < 65:   return "Alto"
    if risk_index < 80:   return "Epidemia"
    return "Crítico"


def format_number(n: float, decimals: int = 1) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.{decimals}f}M"
    if n >= 1_000:
        return f"{n/1_000:.{decimals}f}K"
    return f"{n:.{decimals}f}"


def get_epidemiological_week(date: datetime = None) -> Tuple[int, int]:
    """Retorna (año, semana_epidemiológica) para una fecha dada."""
    d = date or datetime.now()
    iso = d.isocalendar()
    return iso[0], iso[1]
