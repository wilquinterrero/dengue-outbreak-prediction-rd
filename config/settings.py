"""
Configuración centralizada del sistema de predicción de dengue.
Carga variables de entorno y define constantes del proyecto.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# --- PROVINCIAS DE LA REPÚBLICA DOMINICANA ---
PROVINCES = [
    "Azua", "Bahoruco", "Barahona", "Dajabón", "Duarte",
    "Elías Piña", "El Seibo", "Espaillat", "Hato Mayor", "Hermanas Mirabal",
    "Independencia", "La Altagracia", "La Romana", "La Vega", "María Trinidad Sánchez",
    "Monseñor Nouel", "Monte Cristi", "Monte Plata", "Pedernales", "Peravia",
    "Puerto Plata", "Samaná", "San Cristóbal", "San José de Ocoa", "San Juan",
    "San Pedro de Macorís", "Sánchez Ramírez", "Santiago", "Santiago Rodríguez",
    "Santo Domingo", "Valverde", "Distrito Nacional"
]

PROVINCE_CODES = {province: f"RD{str(i+1).zfill(2)}" for i, province in enumerate(PROVINCES)}

# --- UMBRALES EPIDEMIOLÓGICOS ---
RISK_THRESHOLDS = {
    "bajo":     (0, 25),
    "moderado": (25, 50),
    "alto":     (50, 65),
    "epidemia": (65, 80),
    "critico":  (80, 100),
}

EPIDEMIC_THRESHOLD = int(os.getenv("EPIDEMIC_THRESHOLD", "65"))
ALERT_THRESHOLD = int(os.getenv("ALERT_THRESHOLD", "80"))

# --- FEATURES DEL MODELO ---
CLIMATE_FEATURES = [
    "rainfall_mm", "temp_max_c", "temp_min_c", "temp_avg_c",
    "humidity_pct", "wind_speed_kmh", "enso_index"
]

EPIDEMIOLOGICAL_FEATURES = [
    "cases_week_1", "cases_week_2", "cases_week_3", "cases_week_4",
    "cases_cumulative_year", "incidence_rate_100k"
]

DEMOGRAPHIC_FEATURES = [
    "population", "population_density_km2", "urban_pct",
    "poverty_index", "sanitation_index"
]

ALL_FEATURES = CLIMATE_FEATURES + EPIDEMIOLOGICAL_FEATURES + DEMOGRAPHIC_FEATURES
TARGET_COLUMN = "outbreak_risk_index"
FORECAST_HORIZON = 4  # semanas

# --- PARÁMETROS DEL MODELO ---
RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}

LSTM_PARAMS = {
    "units": [128, 64, 32],
    "dropout": 0.2,
    "recurrent_dropout": 0.1,
    "epochs": 100,
    "batch_size": 32,
    "sequence_length": 12,
    "learning_rate": 0.001,
}

ENSEMBLE_WEIGHTS = {"random_forest": 0.6, "lstm": 0.4}
MODEL_RETRAIN_THRESHOLD = float(os.getenv("MODEL_RETRAIN_THRESHOLD", "0.88"))

# --- RUTAS ---
class Paths:
    DATA_RAW = BASE_DIR / "data" / "raw"
    DATA_PROCESSED = BASE_DIR / "data" / "processed"
    DATA_EXTERNAL = BASE_DIR / "data" / "external"
    MODELS = BASE_DIR / "models"
    OUTPUTS = BASE_DIR / "outputs"
    VISUALS = BASE_DIR / "visuals"
    NOTEBOOKS = BASE_DIR / "notebooks"
    LOGS = BASE_DIR / "logs"

    @classmethod
    def ensure_dirs(cls):
        for attr in ["DATA_RAW", "DATA_PROCESSED", "DATA_EXTERNAL",
                     "MODELS", "OUTPUTS", "VISUALS", "LOGS"]:
            getattr(cls, attr).mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    # App
    app_password: str = Field(default="admin", env="APP_PASSWORD")
    app_secret_key: str = Field(default="dev-secret-key-32-characters!!", env="APP_SECRET_KEY")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")

    # Cifrado
    encryption_key: str = Field(default="", env="ENCRYPTION_KEY")
    encryption_salt: str = Field(default="", env="ENCRYPTION_SALT")

    # API
    api_key_secret: str = Field(default="dev-api-key", env="API_KEY_SECRET")
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")

    # Email
    email_host: str = Field(default="smtp.gmail.com", env="EMAIL_HOST")
    email_port: int = Field(default=587, env="EMAIL_PORT")
    email_user: str = Field(default="", env="EMAIL_USER")
    email_password: str = Field(default="", env="EMAIL_PASSWORD")
    email_from: str = Field(default="", env="EMAIL_FROM")
    email_to_alerts: str = Field(default="", env="EMAIL_TO_ALERTS")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/dengue_system.log", env="LOG_FILE")

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

    @property
    def alert_recipients(self) -> List[str]:
        return [e.strip() for e in self.email_to_alerts.split(",") if e.strip()]


settings = Settings()
