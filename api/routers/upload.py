"""
Router de carga de datos semanales.
POST /upload → procesa nuevos datos CSV o JSON
"""

import io
import pandas as pd
from typing import List
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from loguru import logger

from api.auth import require_api_key
from api.schemas import UploadResponse, WeeklyDataUpload
from config.settings import Paths

router = APIRouter(prefix="/upload", tags=["Carga de Datos"])


@router.post(
    "",
    response_model=UploadResponse,
    summary="Cargar nuevos datos semanales",
    description="""
Sube un archivo CSV con datos epidemiológicos y climáticos semanales.
El sistema procesa, valida y almacena los datos para actualizar las predicciones.

**Formato CSV esperado:**
`province, year, week, cases, deaths, rainfall_mm, temp_avg_c, humidity_pct`
    """,
)
async def upload_weekly_data(
    file: UploadFile = File(..., description="Archivo CSV con datos semanales"),
    _: str = Depends(require_api_key),
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Solo se aceptan archivos CSV.",
        )

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error leyendo CSV: {str(e)}",
        )

    required_cols = {"province", "year", "week", "cases"}
    missing = required_cols - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Columnas requeridas faltantes: {missing}",
        )

    # Validar provincias
    from config.settings import PROVINCES
    invalid_provinces = set(df["province"].unique()) - set(PROVINCES)
    if invalid_provinces:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Provincias no válidas: {invalid_provinces}",
        )

    # Guardar datos cifrados
    Paths.DATA_RAW.mkdir(parents=True, exist_ok=True)
    try:
        from src.security import DataEncryptor
        enc = DataEncryptor()
        encrypted = enc.encrypt_dataframe(df)
        out_path = Paths.DATA_RAW / f"upload_{file.filename}.enc"
        out_path.write_bytes(encrypted)
        logger.info(f"Datos cargados y cifrados: {len(df)} registros → {out_path}")
    except Exception as e:
        logger.warning(f"Guardando sin cifrado: {e}")
        out_path = Paths.DATA_RAW / f"upload_{file.filename}"
        df.to_csv(out_path, index=False)

    # Verificar si se necesita re-entrenar
    retrain_triggered = False
    try:
        from api.main import app
        ensemble = app.state.ensemble
        accuracy = ensemble.get_ensemble_accuracy()
        if accuracy < 88.0:
            logger.warning(f"Accuracy {accuracy:.1f}% < 88% — re-entrenamiento programado")
            retrain_triggered = True
    except Exception:
        pass

    return UploadResponse(
        status="success",
        records_processed=len(df),
        message=f"{len(df)} registros procesados y almacenados correctamente.",
        retrain_triggered=retrain_triggered,
    )


@router.post(
    "/json",
    response_model=UploadResponse,
    summary="Cargar datos en formato JSON",
)
async def upload_json_data(
    data: List[WeeklyDataUpload],
    _: str = Depends(require_api_key),
):
    """Acepta lista de objetos JSON con datos semanales."""
    df = pd.DataFrame([d.dict() for d in data])
    Paths.DATA_RAW.mkdir(parents=True, exist_ok=True)

    try:
        from src.security import DataEncryptor
        enc = DataEncryptor()
        encrypted = enc.encrypt_dataframe(df)
        out_path = Paths.DATA_RAW / "upload_json_latest.enc"
        out_path.write_bytes(encrypted)
    except Exception as e:
        logger.warning(f"Guardando JSON sin cifrado: {e}")
        out_path = Paths.DATA_RAW / "upload_json_latest.csv"
        df.to_csv(out_path, index=False)

    return UploadResponse(
        status="success",
        records_processed=len(df),
        message=f"{len(df)} registros JSON procesados correctamente.",
    )
