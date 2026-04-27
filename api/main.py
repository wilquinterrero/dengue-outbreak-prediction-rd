"""
FastAPI - API REST del Sistema de Predicción de Dengue RD
Punto de entrada principal de la API.
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from api.routers import predictions, provinces, alerts, upload
from src.models.ensemble import EnsembleModel
from src.utils.logger import setup_logger

APP_VERSION = "1.0.0"
_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicialización y limpieza de recursos al arrancar/detener."""
    setup_logger()
    logger.info("=" * 55)
    logger.info(f"  Dengue Outbreak Prediction API v{APP_VERSION}")
    logger.info("  República Dominicana — MSP/DIGEPI")
    logger.info("=" * 55)

    # Cargar modelos al arranque
    ensemble = EnsembleModel()
    try:
        ensemble.load_models()
        accuracy = ensemble.get_ensemble_accuracy()
        logger.success(f"Modelos cargados — Accuracy ensemble: {accuracy:.1f}%")
    except Exception as e:
        logger.warning(f"Modelos no cargados ({e}). Use /train para entrenar primero.")

    app.state.ensemble = ensemble
    app.state.start_time = _start_time

    yield

    logger.info("API detenida correctamente")


app = FastAPI(
    title="Dengue Outbreak Prediction API — República Dominicana",
    description="""
## Sistema de Predicción de Brotes de Dengue

API REST para predicción de riesgo de brotes de dengue en las **32 provincias**
de la República Dominicana, usando ensemble de modelos **Random Forest + LSTM**.

### Características
- Índice de riesgo 0–100 por provincia
- Pronóstico a **4 semanas** adelante
- Alertas automáticas cuando se supera el umbral epidémico
- Integración con datos PAHO, ONAMET y MSP/SINAVE

### Autenticación
Todos los endpoints requieren el header `X-API-Key` con la clave configurada en `.env`.

### Umbrales de riesgo
| Nivel | Rango |
|-------|-------|
| Bajo | 0 – 25 |
| Moderado | 25 – 50 |
| Alto | 50 – 65 |
| Epidemia | 65 – 80 |
| Crítico | 80 – 100 |
    """,
    version=APP_VERSION,
    contact={
        "name": "Equipo de Epidemiología Digital",
        "email": "epidemiologia@msp.gob.do",
    },
    license_info={"name": "MIT"},
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware de logging de requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    logger.debug(f"{request.method} {request.url.path} → {response.status_code} ({duration:.0f}ms)")
    return response


# Manejador global de errores
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error no manejado: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor. Revise los logs para más detalles."},
    )


# Routers
app.include_router(predictions.router)
app.include_router(provinces.router)
app.include_router(alerts.router)
app.include_router(upload.router)


@app.get("/", tags=["Sistema"])
async def root():
    return {
        "sistema": "Dengue Outbreak Prediction — República Dominicana",
        "version": APP_VERSION,
        "estado": "operativo",
        "documentacion": "/docs",
        "documentacion_redoc": "/redoc",
        "endpoints_principales": {
            "prediccion_provincia": "GET /predict/{province}",
            "todas_provincias": "GET /provinces",
            "alertas": "GET /alerts",
            "cargar_datos": "POST /upload",
            "salud": "GET /health",
        },
    }


@app.get("/health", tags=["Sistema"])
async def health_check():
    """Verificación de salud de la API y estado de los modelos."""
    uptime = time.time() - app.state.start_time
    try:
        accuracy = app.state.ensemble.get_ensemble_accuracy()
        model_loaded = app.state.ensemble._rf_ready or app.state.ensemble._lstm_ready
    except Exception:
        accuracy = 0.0
        model_loaded = False

    return {
        "status": "healthy",
        "version": APP_VERSION,
        "model_loaded": model_loaded,
        "ensemble_accuracy_pct": round(accuracy, 2),
        "uptime_seconds": round(uptime, 1),
        "provinces_covered": 32,
    }
