"""
Script para iniciar la API FastAPI con autenticación previa.
Uso: python run_api.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.logger import setup_logger
from src.security.encryption import prompt_password
from config.settings import settings

setup_logger()

if not prompt_password():
    print("Acceso denegado.")
    sys.exit(1)

import uvicorn
from loguru import logger

logger.info(f"Iniciando API en http://{settings.api_host}:{settings.api_port}")
logger.info("Documentación disponible en /docs")

uvicorn.run(
    "api.main:app",
    host=settings.api_host,
    port=settings.api_port,
    reload=settings.debug,
    workers=1 if settings.debug else 2,
    log_level="info",
)
