# Referencia Completa de la API REST — Dengue Outbreak Prediction RD

## Autenticación

Todos los endpoints requieren el header `X-API-Key`:

```http
X-API-Key: tu_clave_api_del_env
```

Error sin autenticación:
```json
{"detail": "API Key requerida. Incluya el header X-API-Key."}
```

---

## Endpoints

### `GET /` — Información del sistema

No requiere autenticación.

```bash
curl http://localhost:8000/
```

**Respuesta 200:**
```json
{
  "sistema": "Dengue Outbreak Prediction — República Dominicana",
  "version": "1.0.0",
  "estado": "operativo",
  "documentacion": "/docs"
}
```

---

### `GET /health` — Estado del sistema

```bash
curl http://localhost:8000/health
```

**Respuesta 200:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "ensemble_accuracy_pct": 91.3,
  "uptime_seconds": 3600.5,
  "provinces_covered": 32
}
```

---

### `GET /predict/{province}` — Predicción por provincia

**Parámetros:**
- `province` (path, string): Nombre exacto de la provincia (ver `/provinces/list`)

```bash
curl -H "X-API-Key: clave" \
  http://localhost:8000/predict/Santo%20Domingo
```

**Respuesta 200:**
```json
{
  "province": "Santo Domingo",
  "current_risk_index": 72.5,
  "risk_level": "Epidemia",
  "forecast_4_weeks": {
    "week_1": 72.5,
    "week_2": 76.1,
    "week_3": 71.3,
    "week_4": 68.0
  },
  "peak_risk": 76.1,
  "peak_week": 2,
  "is_epidemic": true,
  "is_alert": false,
  "trend": "Ascendente",
  "generated_at": "2026-04-22T10:30:00"
}
```

**Errores:**
- `401`: Sin API Key
- `403`: API Key inválida
- `404`: Provincia no encontrada
- `500`: Error interno del servidor

---

### `GET /provinces` — Todas las provincias

```bash
curl -H "X-API-Key: clave" \
  http://localhost:8000/provinces
```

**Respuesta 200:**
```json
{
  "total_provinces": 32,
  "epidemic_count": 5,
  "alert_count": 1,
  "average_risk": 42.3,
  "highest_risk_province": "Santo Domingo",
  "predictions": [
    {
      "province": "Santo Domingo",
      "risk_index": 72.5,
      "risk_level": "Epidemia",
      "is_epidemic": true,
      "trend": "Ascendente"
    }
  ],
  "generated_at": "2026-04-22T10:30:00"
}
```

---

### `GET /provinces/list` — Lista de nombres

```bash
curl -H "X-API-Key: clave" \
  http://localhost:8000/provinces/list
```

**Respuesta 200:**
```json
{
  "provinces": ["Azua", "Bahoruco", "Barahona", ...],
  "total": 32
}
```

---

### `GET /alerts` — Provincias en alerta epidémica

**Query params:**
- `threshold` (float, 0-100, default=65): Umbral de riesgo para considerar alerta

```bash
# Epidemias (umbral default 65)
curl -H "X-API-Key: clave" \
  http://localhost:8000/alerts

# Nivel crítico (umbral 80)
curl -H "X-API-Key: clave" \
  "http://localhost:8000/alerts?threshold=80"
```

**Respuesta 200:**
```json
{
  "total_alerts": 3,
  "alerts": [
    {
      "province": "Distrito Nacional",
      "current_risk_index": 83.2,
      "risk_level": "Crítico",
      "forecast_4_weeks": { ... },
      "is_epidemic": true,
      "is_alert": true,
      "trend": "Ascendente"
    }
  ],
  "generated_at": "2026-04-22T10:30:00"
}
```

---

### `GET /alerts/critical` — Solo nivel crítico

```bash
curl -H "X-API-Key: clave" \
  http://localhost:8000/alerts/critical
```

**Respuesta 200:**
```json
{
  "level": "Crítico",
  "threshold": 80,
  "count": 2,
  "provinces": [
    {
      "province": "Distrito Nacional",
      "risk_index": 83.2,
      "trend": "Ascendente",
      "peak_week": 1
    }
  ]
}
```

---

### `POST /upload` — Cargar datos CSV

**Formato del CSV:**
```csv
province,year,week,cases,deaths,rainfall_mm,temp_avg_c,humidity_pct
Santiago,2026,16,145,1,78.3,26.8,82.1
Azua,2026,16,23,0,42.1,29.2,71.4
```

```bash
curl -X POST \
  -H "X-API-Key: clave" \
  -F "file=@datos_semana_16.csv" \
  http://localhost:8000/upload
```

**Respuesta 200:**
```json
{
  "status": "success",
  "records_processed": 32,
  "message": "32 registros procesados y almacenados correctamente.",
  "retrain_triggered": false
}
```

---

### `POST /upload/json` — Cargar datos JSON

```bash
curl -X POST \
  -H "X-API-Key: clave" \
  -H "Content-Type: application/json" \
  -d '[{"province":"Santiago","year":2026,"week":16,"cases":145}]' \
  http://localhost:8000/upload/json
```

---

## Códigos de error HTTP

| Código | Descripción |
|--------|-------------|
| 200 | Éxito |
| 401 | Sin autenticación |
| 403 | API Key inválida |
| 404 | Recurso no encontrado |
| 422 | Error de validación |
| 500 | Error interno |

---

## Integración con Power BI

### Paso a paso

1. **Power BI Desktop** → Inicio → **Obtener datos** → **Web**
2. En URL básica: `http://localhost:8000/provinces`
3. **Configuración avanzada** → **Parámetros de encabezado HTTP**:
   - Nombre: `X-API-Key` | Valor: `tu_api_key`
4. **Aceptar** → **Transformar datos**
5. Expandir columna `predictions` → **Expandir a filas nuevas**
6. Expandir campos de la columna expandida
7. **Cerrar y aplicar**

### Para actualización automática
Configurar la actualización programada en Power BI Service semanalmente los lunes por la mañana, después de que el scheduler ejecute el job de exportación.
