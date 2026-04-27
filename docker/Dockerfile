# ============================================================
# Dengue Outbreak Prediction - República Dominicana
# Dockerfile multi-stage para producción
# ============================================================

# --- STAGE 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias Python
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --user --no-cache-dir -r requirements.txt

# --- STAGE 2: Final ---
FROM python:3.11-slim

WORKDIR /app

# Dependencias de runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar paquetes Python instalados
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copiar código fuente
COPY config/ ./config/
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY scheduler/ ./scheduler/

# Crear directorios necesarios
RUN mkdir -p data/raw data/processed data/external \
             models outputs visuals logs

# Usuario no-root para seguridad
RUN useradd --create-home --shell /bin/bash dengue && \
    chown -R dengue:dengue /app
USER dengue

# Variables de entorno por defecto (sobrescribir con .env en producción)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    STREAMLIT_PORT=8501

EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando por defecto: API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
