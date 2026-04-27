"""
Autenticación por API Key para todos los endpoints FastAPI.
La clave se envía en el header X-API-Key.
"""

import os
import hashlib
from typing import Optional
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from loguru import logger

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

_STORED_KEY_HASH: Optional[str] = None


def _get_stored_key_hash() -> str:
    global _STORED_KEY_HASH
    if _STORED_KEY_HASH is None:
        key = os.getenv("API_KEY_SECRET", "")
        if not key:
            logger.critical("API_KEY_SECRET no configurado en .env")
            raise RuntimeError("API_KEY_SECRET no configurado")
        _STORED_KEY_HASH = hashlib.sha256(key.encode()).hexdigest()
    return _STORED_KEY_HASH


async def require_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Dependencia FastAPI que valida el API Key en cada request."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key requerida. Incluya el header X-API-Key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    try:
        stored_hash = _get_stored_key_hash()
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    provided_hash = hashlib.sha256(api_key.encode()).hexdigest()
    if not hashlib.compare_digest(provided_hash, stored_hash):
        logger.warning(f"Intento de acceso con API Key inválida")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API Key inválida o expirada.",
        )
    return api_key
