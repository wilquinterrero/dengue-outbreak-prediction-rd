"""Sistema de logging centralizado con rotación de archivos."""

import sys
from pathlib import Path
from loguru import logger
from config.settings import settings, Paths


def setup_logger() -> None:
    """Configura loguru con salida a consola y archivo rotativo."""
    Paths.LOGS.mkdir(parents=True, exist_ok=True)

    logger.remove()

    # Consola: colorido y conciso
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> — {message}",
        colorize=True,
    )

    # Archivo: detallado con rotación
    log_path = Paths.LOGS / "dengue_system.log"
    logger.add(
        str(log_path),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} — {message}",
        rotation="10 MB",
        retention="30 days",
        compression="gz",
        encoding="utf-8",
    )

    # Archivo separado solo para errores
    error_path = Paths.LOGS / "errors.log"
    logger.add(
        str(error_path),
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line}\n{message}\n{exception}",
        rotation="5 MB",
        retention="60 days",
        encoding="utf-8",
    )

    logger.info("Sistema de logging inicializado")
