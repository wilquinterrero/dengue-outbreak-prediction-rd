"""
Script de entrenamiento principal.
Uso: python train.py [--force] [--weeks N]

Opciones:
  --force    Fuerza re-entrenamiento aunque la accuracy sea suficiente
  --weeks N  Número de semanas de datos históricos a usar (default: 208)
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.logger import setup_logger
from src.security.encryption import prompt_password
from src.data.ingestion import DataIngestion
from src.models.trainer import ModelTrainer
from config.settings import settings, Paths


def main():
    setup_logger()

    # Autenticación
    if not prompt_password():
        print("Acceso denegado.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Entrenamiento del modelo Dengue-RD")
    parser.add_argument("--force", action="store_true", help="Forzar re-entrenamiento")
    parser.add_argument("--weeks", type=int, default=208, help="Semanas de datos históricos")
    parser.add_argument("--no-retrain-check", action="store_true",
                        help="Solo verificar accuracy sin re-entrenar")
    args = parser.parse_args()

    Paths.ensure_dirs()

    from loguru import logger

    # Descargar o cargar datos
    master_path = Paths.DATA_RAW / "master_dataset.csv"
    if master_path.exists() and not args.force:
        logger.info(f"Cargando datos existentes: {master_path}")
        import pandas as pd
        df = pd.read_csv(master_path)
        logger.info(f"Datos cargados: {len(df)} registros")
    else:
        logger.info("Descargando datos desde fuentes externas...")
        ingestion = DataIngestion()
        df = ingestion.consolidate_all_sources(weeks=args.weeks)

    trainer = ModelTrainer()

    if args.no_retrain_check:
        retrained = trainer.check_and_retrain(df)
        logger.info(f"Re-entrenamiento: {'SI' if retrained else 'NO'}")
    else:
        report = trainer.full_training_pipeline(df)
        acc = report["ensemble_accuracy_pct"]
        meets = report["meets_threshold"]
        logger.info(f"Entrenamiento completado — Accuracy: {acc:.1f}% | Umbral 88%: {'✓' if meets else '✗'}")

    logger.success("Pipeline de entrenamiento finalizado.")


if __name__ == "__main__":
    main()
