# 🦟 Sistema de Predicción de Brotes de Dengue — República Dominicana

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.34-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Sistema de inteligencia artificial para predicción de brotes de dengue en las 32 provincias de la República Dominicana, usando ensemble de modelos Machine Learning (Random Forest + LSTM).**

[Ver API Docs](#api-rest) · [Ver Dashboard](#dashboard-streamlit) · [Instalación](#instalación) · [Metodología](#metodología)

</div>

---

## Descripción

Este proyecto desarrolla un sistema end-to-end de predicción de brotes de dengue para la República Dominicana, integrando:

- **Datos reales** de PAHO/OPS, ONAMET, ONE y MSP-SINAVE
- **Modelo ensemble** Random Forest (60%) + LSTM (40%)
- **Pronóstico a 4 semanas** con índice de riesgo 0-100 por provincia
- **API REST** con FastAPI para integración con Power BI y sistemas externos
- **Dashboard interactivo** con Streamlit y mapa coroplético
- **Automatización semanal** de descarga de datos y re-entrenamiento
- **Cifrado AES-256** para protección de datos epidemiológicos sensibles

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    FUENTES DE DATOS                         │
│  PAHO/OPS  │  ONAMET  │  ONE (Censo)  │  MSP-SINAVE        │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│               PIPELINE DE DATOS                             │
│  DataIngestion → DataPreprocessor → Feature Engineering     │
│  (lag features, rolling windows, normalización, LSTM seqs)  │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                   MODELO ENSEMBLE                           │
│  ┌───────────────────┐    ┌──────────────────────────────┐  │
│  │  Random Forest    │    │  LSTM Encoder-Decoder        │  │
│  │  200 estimadores  │    │  128→64→32 unidades          │  │
│  │  TimeSeriesSplit  │    │  Dropout + BatchNorm         │  │
│  │  Peso: 60%        │    │  Peso: 40%                   │  │
│  └─────────┬─────────┘    └──────────────┬───────────────┘  │
│            └──────────────┬──────────────┘                  │
│                     Ensemble Ponderado                      │
│               Accuracy objetivo: ≥ 88%                     │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│         SALIDAS DEL SISTEMA                                 │
│  FastAPI REST  │  Streamlit Dashboard  │  CSV Cifrado       │
│  Power BI      │  Email Alertas        │  Logs              │
└─────────────────────────────────────────────────────────────┘
```

---

## Instalación

### Requisitos previos

- Python 3.11+
- pip o conda
- Docker y Docker Compose (opcional, para despliegue)

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/dengue-outbreak-prediction-rd.git
cd dengue-outbreak-prediction-rd
```

### 2. Crear entorno virtual

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

```bash
# Copiar plantilla
cp .env.example .env

# Editar con valores reales
nano .env   # o notepad .env en Windows
```

Variables críticas a configurar:

```env
APP_PASSWORD=tu_password_seguro
ENCRYPTION_KEY=tu_clave_aes256
API_KEY_SECRET=tu_api_key
EMAIL_USER=tu@gmail.com
EMAIL_PASSWORD=tu_app_password
EMAIL_TO_ALERTS=epidemiologia@msp.gob.do
```

### 5. Entrenar el modelo

```bash
python train.py
# Ingrese la contraseña configurada en APP_PASSWORD
```

### 6. Iniciar la API

```bash
python run_api.py
# API disponible en: http://localhost:8000
# Documentación en: http://localhost:8000/docs
```

### 7. Iniciar el Dashboard

```bash
python run_dashboard.py
# Dashboard en: http://localhost:8501
```

---

## Docker (Despliegue completo)

```bash
# Copiar y configurar .env
cp .env.example .env
# Editar .env con valores reales

# Construir e iniciar todos los servicios
docker-compose -f docker/docker-compose.yml up -d

# Ver logs
docker-compose -f docker/docker-compose.yml logs -f

# Detener
docker-compose -f docker/docker-compose.yml down
```

Servicios disponibles:
| Servicio | URL |
|----------|-----|
| API REST | http://localhost:8000 |
| Documentación API | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |

---

## API REST

Todos los endpoints requieren el header `X-API-Key`.

### Endpoints principales

#### `GET /predict/{province}` — Predicción por provincia
```bash
curl -H "X-API-Key: tu_api_key" \
  http://localhost:8000/predict/Santiago
```

**Respuesta:**
```json
{
  "province": "Santiago",
  "current_risk_index": 68.5,
  "risk_level": "Epidemia",
  "forecast_4_weeks": {
    "week_1": 68.5,
    "week_2": 72.1,
    "week_3": 69.8,
    "week_4": 65.3
  },
  "peak_risk": 72.1,
  "peak_week": 2,
  "is_epidemic": true,
  "is_alert": false,
  "trend": "Ascendente"
}
```

#### `GET /provinces` — Resumen todas las provincias
```bash
curl -H "X-API-Key: tu_api_key" \
  http://localhost:8000/provinces
```

#### `GET /alerts` — Provincias en alerta epidémica
```bash
curl -H "X-API-Key: tu_api_key" \
  "http://localhost:8000/alerts?threshold=65"
```

#### `POST /upload` — Cargar nuevos datos CSV
```bash
curl -X POST -H "X-API-Key: tu_api_key" \
  -F "file=@datos_semana_15.csv" \
  http://localhost:8000/upload
```

### Escalas de riesgo

| Nivel | Rango | Color |
|-------|-------|-------|
| Bajo | 0 – 25 | 🟢 Verde |
| Moderado | 25 – 50 | 🟡 Amarillo |
| Alto | 50 – 65 | 🟠 Naranja |
| Epidemia | 65 – 80 | 🔴 Rojo |
| Crítico | 80 – 100 | ⚫ Rojo oscuro |

---

## Dashboard Streamlit

El dashboard incluye:

- **Mapa interactivo** de la República Dominicana con riesgo por provincia
- **Medidor gauge** de riesgo individual por provincia
- **Pronóstico visual** a 4 semanas con gráfico de línea
- **Ranking horizontal** de provincias por nivel de riesgo
- **Tendencia histórica** de las últimas 52 semanas
- **Alertas epidémicas** en banner animado
- **Exportación a CSV** de todas las predicciones
- **Conexión a Power BI** con instrucciones integradas

---

## Conectar Power BI

1. Abrir Power BI Desktop
2. **Obtener datos** → **Web**
3. URL: `http://localhost:8000/provinces`
4. En **Configuración avanzada** → Headers HTTP:
   - Nombre: `X-API-Key`
   - Valor: tu clave API del .env
5. **Aceptar** → **Transformar datos**
6. Expandir el campo `predictions` (lista)
7. Cargar y crear visualizaciones

---

## Fuentes de Datos

| Fuente | Datos | Frecuencia | URL |
|--------|-------|-----------|-----|
| PAHO/OPS | Casos semanales de dengue por país | Semanal | paho.org/data |
| ONAMET | Precipitación, temperatura, humedad | Diaria | onamet.gob.do |
| ONE | Demografía, censo 32 provincias | Anual | one.gob.do |
| MSP-DIGEPI | Boletines epidemiológicos | Semanal | digepi.gob.do |
| NOAA | Índice ENSO (ONI) | Mensual | psl.noaa.gov |

---

## Metodología

### Variables del modelo

**Climáticas:**
- Precipitación semanal (mm)
- Temperatura máxima, mínima y promedio (°C)
- Humedad relativa (%)
- Velocidad del viento (km/h)
- Índice ENSO (ONI)

**Epidemiológicas:**
- Casos confirmados semanas 1-4 previas
- Acumulado anual de casos
- Tasa de incidencia por 100,000 habitantes

**Demográficas:**
- Población total por provincia
- Densidad poblacional (hab/km²)
- % de población urbana
- Índice de pobreza
- Índice de saneamiento

**Ingeniería de features:**
- Lag features: 1, 2, 3, 4, 8, 12 semanas
- Rolling mean y std: 4, 8, 12 semanas
- Features cíclicas: semana del año (sin/cos), mes (sin/cos)
- Variable binaria: temporada lluviosa

### Arquitectura del ensemble

**Random Forest (60% del peso):**
- 200 estimadores, max_depth=15
- Multi-output: predice las 4 semanas simultáneamente
- Validación con TimeSeriesSplit (5 folds)

**LSTM Encoder-Decoder (40% del peso):**
- Entrada: secuencias de 12 semanas
- Arquitectura: 128→64 (encoder) + 64→32 (decoder)
- BatchNormalization + Dropout (20%)
- EarlyStopping con paciencia=15

**Índice de riesgo compuesto:**
- 40% tasa de incidencia epidemiológica
- 25% lluvia acumulada (vector de reproducción del Aedes)
- 20% temperatura promedio (ciclo de desarrollo del vector)
- 10% humedad relativa
- 5% índice de pobreza (vulnerabilidad social)

---

## Seguridad

- **Cifrado AES-256-CBC** con HMAC-SHA256 para autenticidad de todos los archivos de datos
- **Derivación de clave PBKDF2** con 100,000 iteraciones
- **API Key authentication** en todos los endpoints REST
- **Contraseña de acceso** requerida antes de ejecutar scripts de entrenamiento
- **Variables de entorno** para todas las credenciales (nunca en el código)
- `.gitignore` excluye datos crudos, modelos entrenados, `.env` y outputs

---

## Automatización

El scheduler semanal ejecuta automáticamente:

| Tarea | Horario | Descripción |
|-------|---------|-------------|
| Descarga de datos | Lunes 6:00 AM | PAHO + ONAMET + ONE |
| Verificación de modelo | Lunes 8:00 AM | Re-entrena si accuracy < 88% |
| Exportación | Lunes 10:00 AM | CSV cifrado para Power BI |
| Verificación alertas | Cada 6 horas | Email si provincia supera umbral |
| Health check | Cada hora | Estado del sistema |

---

## Tests

```bash
# Ejecutar todos los tests
pytest

# Con reporte de cobertura
pytest --cov=src --cov=api --cov-report=html

# Solo un módulo
pytest tests/test_encryption.py -v
pytest tests/test_model.py -v
pytest tests/test_api.py -v
```

Cobertura objetivo: **≥ 75%**

---

## Estructura del Proyecto

```
dengue-outbreak-prediction-rd/
├── config/                 # Configuración centralizada
│   └── settings.py         # Variables de entorno + constantes
├── src/
│   ├── data/
│   │   ├── ingestion.py    # Descarga desde PAHO, ONAMET, ONE
│   │   └── preprocessing.py # Limpieza, lag features, normalización
│   ├── models/
│   │   ├── random_forest.py # Modelo RF multi-output
│   │   ├── lstm.py          # Modelo LSTM encoder-decoder
│   │   ├── ensemble.py      # Combinación ponderada RF+LSTM
│   │   └── trainer.py       # Pipeline de entrenamiento
│   ├── security/
│   │   └── encryption.py    # Cifrado AES-256 + autenticación
│   └── utils/
│       ├── logger.py        # Sistema de logging con Loguru
│       └── helpers.py       # Utilidades generales
├── api/
│   ├── main.py             # FastAPI app principal
│   ├── auth.py             # Autenticación API Key
│   ├── schemas.py          # Modelos Pydantic
│   └── routers/
│       ├── predictions.py   # GET /predict/{province}
│       ├── provinces.py     # GET /provinces
│       ├── alerts.py        # GET /alerts
│       └── upload.py        # POST /upload
├── app/
│   └── streamlit_app.py    # Dashboard interactivo
├── scheduler/
│   ├── weekly_scheduler.py # APScheduler con jobs
│   ├── jobs.py             # Funciones de cada job
│   └── alerts.py           # Sistema de email
├── tests/
│   ├── conftest.py         # Fixtures compartidos
│   ├── test_encryption.py  # Tests de cifrado
│   ├── test_model.py       # Tests de modelos ML
│   └── test_api.py         # Tests de integración API
├── docker/
│   ├── Dockerfile          # Imagen multi-stage
│   ├── docker-compose.yml  # API + Dashboard + Scheduler
│   └── nginx.conf          # Reverse proxy
├── notebooks/
│   └── 01_EDA_Dengue_RD.ipynb # Análisis exploratorio
├── data/
│   ├── raw/               # Datos crudos (gitignored)
│   └── processed/         # Datos procesados (gitignored)
├── models/                # Modelos entrenados (gitignored)
├── outputs/               # Predicciones cifradas (gitignored)
├── visuals/               # Gráficos exportados (gitignored)
├── .env.example           # Plantilla de variables de entorno
├── .gitignore
├── requirements.txt
├── pytest.ini
├── train.py               # Script de entrenamiento
├── run_api.py             # Iniciar API
└── run_dashboard.py       # Iniciar Dashboard
```

---

## Diccionario de Datos

| Variable | Tipo | Descripción | Fuente |
|----------|------|-------------|--------|
| province | str | Nombre de la provincia | ONE |
| year | int | Año epidemiológico | - |
| week | int | Semana epidemiológica (1-53) | - |
| cases | int | Casos confirmados de dengue | MSP-SINAVE |
| deaths | int | Muertes por dengue | MSP-SINAVE |
| rainfall_mm | float | Precipitación semanal en mm | ONAMET |
| temp_max_c | float | Temperatura máxima en °C | ONAMET |
| temp_min_c | float | Temperatura mínima en °C | ONAMET |
| temp_avg_c | float | Temperatura promedio en °C | ONAMET |
| humidity_pct | float | Humedad relativa promedio % | ONAMET |
| wind_speed_kmh | float | Velocidad del viento km/h | ONAMET |
| enso_index | float | Índice ENSO (ONI) | NOAA |
| population | int | Población total de la provincia | ONE |
| population_density_km2 | float | Densidad hab/km² | ONE |
| urban_pct | float | % de población urbana | ONE |
| poverty_index | float | Índice de pobreza % | ONE |
| sanitation_index | float | Índice de saneamiento (0-100) | ONE |
| incidence_rate_100k | float | Casos por 100,000 habitantes | Calculado |
| outbreak_risk_index | float | Índice de riesgo 0-100 (target) | Calculado |

---

## Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Haga fork del repositorio
2. Cree una rama para su feature: `git checkout -b feature/nueva-funcionalidad`
3. Commit sus cambios: `git commit -m "Agrega nueva funcionalidad"`
4. Push a la rama: `git push origin feature/nueva-funcionalidad`
5. Abra un Pull Request

---

## Contacto y Contexto

Este sistema fue desarrollado como herramienta de apoyo a la vigilancia epidemiológica de dengue en la República Dominicana. Los datos reales requieren acceso institucional a PAHO, ONAMET, ONE y MSP-DIGEPI.

Para uso institucional o académico: consulte con la Dirección General de Epidemiología (DIGEPI) del Ministerio de Salud Pública de la República Dominicana.

---

## Licencia

MIT License — Ver archivo [LICENSE](LICENSE) para detalles.

---

<div align="center">
<p>Desarrollado para la vigilancia epidemiológica de la República Dominicana</p>
<p>Datos: PAHO · ONAMET · ONE · MSP-DIGEPI · NOAA</p>
</div>
