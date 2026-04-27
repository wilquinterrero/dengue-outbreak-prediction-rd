"""
Scheduler semanal automatizado:
- Descarga nuevos datos cada lunes a las 6:00 AM (hora RD)
- Re-entrena el modelo si la accuracy cae por debajo del 88%
- Exporta predicciones cifradas para Power BI
- Envía alertas por correo a provincias en epidemia
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
from loguru import logger

from src.utils.logger import setup_logger
from scheduler.jobs import (
    job_fetch_new_data,
    job_retrain_check,
    job_export_predictions,
    job_send_alerts,
    job_health_check,
)

setup_logger()
scheduler = BlockingScheduler(timezone="America/Santo_Domingo")


def register_jobs():
    # Lunes 6:00 AM: Descarga datos nuevos
    scheduler.add_job(
        job_fetch_new_data,
        CronTrigger(day_of_week="mon", hour=6, minute=0),
        id="fetch_data",
        name="Descarga semanal de datos",
        replace_existing=True,
        misfire_grace_time=3600,
    )

    # Lunes 8:00 AM: Verificar accuracy y re-entrenar si necesario
    scheduler.add_job(
        job_retrain_check,
        CronTrigger(day_of_week="mon", hour=8, minute=0),
        id="retrain_check",
        name="Verificación y re-entrenamiento automático",
        replace_existing=True,
        misfire_grace_time=7200,
    )

    # Lunes 10:00 AM: Exportar predicciones cifradas
    scheduler.add_job(
        job_export_predictions,
        CronTrigger(day_of_week="mon", hour=10, minute=0),
        id="export_predictions",
        name="Exportación de predicciones",
        replace_existing=True,
    )

    # Cada 6 horas: Verificar alertas epidémicas
    scheduler.add_job(
        job_send_alerts,
        CronTrigger(hour="*/6", minute=0),
        id="send_alerts",
        name="Verificación y envío de alertas",
        replace_existing=True,
    )

    # Cada hora: Health check
    scheduler.add_job(
        job_health_check,
        CronTrigger(minute=0),
        id="health_check",
        name="Verificación de salud del sistema",
        replace_existing=True,
    )


def main():
    logger.info("=" * 55)
    logger.info("  Scheduler Dengue RD - Iniciando")
    logger.info(f"  Zona horaria: America/Santo_Domingo")
    logger.info("=" * 55)

    register_jobs()

    logger.info("Jobs registrados:")
    for job in scheduler.get_jobs():
        logger.info(f"  [{job.id}] {job.name}")

    logger.info("Scheduler iniciado. Ctrl+C para detener.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler detenido correctamente")


if __name__ == "__main__":
    main()
