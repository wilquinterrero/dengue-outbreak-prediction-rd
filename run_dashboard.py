"""
Script para iniciar el dashboard Streamlit.
La autenticación se maneja dentro de la propia app de Streamlit.
Uso: python run_dashboard.py
"""

import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.logger import setup_logger
from config.settings import settings

setup_logger()

from loguru import logger
logger.info(f"Iniciando Dashboard Streamlit en http://localhost:{settings.streamlit_port if hasattr(settings, 'streamlit_port') else 8501}")

subprocess.run([
    sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py",
    "--server.port", "8501",
    "--server.address", "0.0.0.0",
    "--server.headless", "true",
    "--browser.gatherUsageStats", "false",
], check=True)
