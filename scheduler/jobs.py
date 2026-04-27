"""
Funciones de los jobs del scheduler.
Cada función es independiente y registra su ejecución en los logs.
"""

from datetime import datetime
from loguru import logger


def job_fetch_new_data():
    """Descarga datos actualizados de todas las fuentes."""
    logger.info(f"[JOB] Iniciando descarga de datos — {datetime.now().isoformat()}")
    try:
        from src.data.ingestion import DataIngestion
        ingestion = DataIngestion()
        df = ingestion.consolidate_all_sources(weeks=4)
        logger.success(f"[JOB] Datos descargados: {len(df)} registros nuevos")
    except Exception as e:
        logger.error(f"[JOB] Error en descarga de datos: {e}", exc_info=True)


def job_retrain_check():
    """Verifica accuracy del modelo y re-entrena si es necesario."""
    logger.info("[JOB] Verificando accuracy del modelo")
    try:
        from config.settings import Paths
        from src.data.ingestion import DataIngestion
        from src.models.trainer import ModelTrainer

        master_path = Paths.DATA_RAW / "master_dataset.csv"
        if not master_path.exists():
            logger.warning("[JOB] No hay datos disponibles para re-entrenamiento")
            return

        import pandas as pd
        df = pd.read_csv(master_path)
        trainer = ModelTrainer()
        retrained = trainer.check_and_retrain(df)

        if retrained:
            logger.success("[JOB] Re-entrenamiento completado exitosamente")
        else:
            logger.info("[JOB] Modelo en buen estado — no se requiere re-entrenamiento")
    except Exception as e:
        logger.error(f"[JOB] Error en verificación de re-entrenamiento: {e}", exc_info=True)


def job_export_predictions():
    """Genera y exporta predicciones cifradas para Power BI."""
    logger.info("[JOB] Exportando predicciones")
    try:
        from src.models.ensemble import EnsembleModel
        from config.settings import PROVINCES

        ensemble = EnsembleModel()
        ensemble.load_models()

        # Generar predicciones simuladas para todas las provincias
        from api.routers.predictions import _mock_prediction
        predictions = [_mock_prediction(p) for p in PROVINCES]
        output_path = ensemble.export_predictions_csv(predictions, encrypt=True)
        logger.success(f"[JOB] Predicciones exportadas: {output_path}")
    except Exception as e:
        logger.error(f"[JOB] Error en exportación: {e}", exc_info=True)


def job_send_alerts():
    """Verifica provincias en epidemia y envía alertas por correo."""
    logger.info("[JOB] Verificando alertas epidémicas")
    try:
        from scheduler.alerts import AlertNotifier
        notifier = AlertNotifier()
        notifier.check_and_notify()
    except Exception as e:
        logger.error(f"[JOB] Error en verificación de alertas: {e}", exc_info=True)


def job_health_check():
    """Verificación de salud del sistema cada hora."""
    try:
        from config.settings import Paths
        disk_ok = Paths.DATA_RAW.exists() and Paths.MODELS.exists()
        status = "OK" if disk_ok else "DEGRADADO"
        logger.debug(f"[HEALTH] Sistema: {status} — {datetime.now().strftime('%H:%M')}")
    except Exception as e:
        logger.error(f"[HEALTH] Error: {e}")
