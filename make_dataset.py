"""Standalone dataset generator — no project imports, pure numpy/pandas."""
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

OUT = Path(__file__).parent / "data" / "raw" / "master_dataset.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

PROVINCES = [
    "Azua", "Bahoruco", "Barahona", "Dajabón", "Duarte",
    "Elías Piña", "El Seibo", "Espaillat", "Hato Mayor", "Hermanas Mirabal",
    "Independencia", "La Altagracia", "La Romana", "La Vega", "María Trinidad Sánchez",
    "Monseñor Nouel", "Monte Cristi", "Monte Plata", "Pedernales", "Peravia",
    "Puerto Plata", "Samaná", "San Cristóbal", "San José de Ocoa", "San Juan",
    "San Pedro de Macorís", "Sánchez Ramírez", "Santiago", "Santiago Rodríguez",
    "Santo Domingo", "Valverde", "Distrito Nacional",
]

DEMOG = {
    "Distrito Nacional": (1078000, 91, 99.5, 18.2),
    "Santo Domingo": (2374000, 1296, 95.3, 22.4),
    "Santiago": (1100000, 2836, 72.1, 28.6),
    "San Cristóbal": (661000, 1265, 65.4, 31.2),
    "La Vega": (395000, 2286, 58.3, 33.8),
}
DEFAULT_DEMOG = (200000, 1500, 50.0, 45.0)

WEEKS = 104
BASE_DATE = datetime(2024, 1, 1)

records = []
for prov in PROVINCES:
    pop, area, urban, poverty = DEMOG.get(prov, DEFAULT_DEMOG)
    density = pop / area
    sanitation = 100 - poverty * 0.6

    for w in range(WEEKS):
        date = BASE_DATE + timedelta(weeks=w)
        year = date.year
        week_num = w % 52 + 1
        month = date.month

        # Climate
        seasonal = max(0.1, 1 + 1.5 * np.sin(np.pi * (month - 4) / 6))
        rainfall = max(0, np.random.exponential(50 * seasonal))
        temp_avg = 27.5 + np.random.normal(0, 0.8)
        temp_max = temp_avg + 3 + np.random.normal(0, 0.5)
        temp_min = temp_avg - 4 + np.random.normal(0, 0.5)
        humidity = min(100, max(40, 75 + np.random.normal(0, 5)))
        wind = max(0, np.random.normal(18, 5))
        enso = np.random.normal(0, 0.6)

        # Epidemiology
        seasonal_epi = 1 + 2.5 * np.sin(np.pi * (week_num - 10) / 26)
        cases = int(np.random.poisson(max(0, 8 * seasonal_epi)))
        deaths = np.random.binomial(cases, 0.002) if cases > 0 else 0
        incidence = cases / pop * 100_000

        # Risk index
        risk = (
            (min(incidence, 100) / 100) * 40
            + (min(rainfall, 300) / 300) * 25
            + max(0, min((temp_avg - 20) / 15, 1)) * 20
            + max(0, min((humidity - 50) / 50, 1)) * 10
            + (poverty / 100) * 5
        )

        records.append({
            "province": prov,
            "date": date.strftime("%Y-%m-%d"),
            "year": year,
            "week": week_num,
            "month": month,
            "rainfall_mm": round(rainfall, 2),
            "temp_max_c": round(temp_max, 2),
            "temp_min_c": round(temp_min, 2),
            "temp_avg_c": round(temp_avg, 2),
            "humidity_pct": round(humidity, 2),
            "wind_speed_kmh": round(wind, 2),
            "enso_index": round(enso, 2),
            "cases": cases,
            "deaths": deaths,
            "incidence_rate_100k": round(incidence, 4),
            "population": pop,
            "area_km2": area,
            "urban_pct": urban,
            "poverty_index": poverty,
            "population_density_km2": round(density, 2),
            "sanitation_index": round(sanitation, 2),
            "outbreak_risk_index": round(min(max(risk, 0), 100), 2),
        })

df = pd.DataFrame(records)
df.to_csv(OUT, index=False)
print(f"Saved {len(df)} rows x {len(df.columns)} cols → {OUT}")
print("Columns:", list(df.columns))
