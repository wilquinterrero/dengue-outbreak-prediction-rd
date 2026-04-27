# Metodología del Modelo — Dengue Outbreak Prediction RD

## Marco Conceptual

El sistema integra la **triada epidemiológica** con variables climáticas y sociales:

```
Agente → Aedes aegypti (vector)
Huésped → Población humana susceptible
Ambiente → Clima tropical + condiciones socioeconómicas
```

El índice de riesgo de brote **no predice número exacto de casos**, sino la **probabilidad relativa de que ocurra un evento epidémico** en cada provincia durante las próximas 4 semanas.

---

## Fuentes de Datos

### 1. PAHO/OPS — Casos de Dengue
- Reporte semanal: casos confirmados, graves y muertes por país
- URL: `https://www3.paho.org/data/index.php/en/mnu-topics/indicadores-dengue-interface/`
- Granularidad temporal: semanal (Semanas Epidemiológicas)
- Cobertura: 2003-presente

### 2. ONAMET — Datos Climáticos
- Variables: precipitación (mm), temperatura (°C), humedad relativa (%)
- Estaciones meteorológicas en todas las provincias
- El **Aedes aegypti** requiere temperatura ≥ 16°C para reproducirse
- Precipitación aumenta criaderos del vector (agua estancada)
- Humedad >60% extiende vida del mosquito adulto

### 3. ONE — Demografía (Censo 2022)
- Población por provincia, densidad, urbanización
- Índice de pobreza (NBI) como proxy de acceso a agua potable y saneamiento
- Mayor densidad = mayor tasa de contacto agente-huésped

### 4. MSP/SINAVE — Boletines Epidemiológicos
- Circulación de serotipos del dengue (DENV 1-4)
- Tasas de hospitalización y mortalidad
- Histórico de brotes previos

### 5. NOAA — Índice ENSO (ONI)
- El Niño asociado con aumento de lluvias en el Caribe
- La Niña con sequías: reduce criaderos pero concentra población en fuentes de agua
- Lag de 3-6 meses entre evento ENSO y pico epidémico

---

## Índice de Riesgo Compuesto (Target)

El índice de riesgo es una variable compuesta 0-100 construida a partir de:

```
Riesgo = 0.40 × Riesgo_Epidemiológico
       + 0.25 × Riesgo_Climático_Lluvia
       + 0.20 × Riesgo_Climático_Temperatura
       + 0.10 × Riesgo_Humedad
       + 0.05 × Riesgo_Social
```

Donde:
- **Riesgo_Epidemiológico** = `min(incidencia_100k / 100, 1) × 100`
- **Riesgo_Climático_Lluvia** = `min(rainfall_mm / 300, 1) × 100`
- **Riesgo_Climático_Temperatura** = `max(0, (temp_avg - 20) / 15) × 100`
- **Riesgo_Humedad** = `max(0, (humidity - 50) / 50) × 100`
- **Riesgo_Social** = `poverty_index` (0-100)

---

## Ingeniería de Features

### Lag Features
Capturan la autocorrelación temporal del dengue:
- Casos lag 1, 2, 3, 4, 8, 12 semanas
- Riesgo lag 1, 2, 3, 4, 8, 12 semanas
- Lluvia lag 1-4 semanas (período de incubación del vector: 8-12 días)

### Rolling Features
Tendencias suavizadas para reducir ruido semanal:
- Media móvil 4, 8, 12 semanas de: casos, lluvia, temperatura, humedad
- Desviación estándar rolling: captura volatilidad

### Features Cíclicas
Para que el modelo reconozca la estacionalidad sin importar el año:
- `week_sin = sin(2π × semana / 52)`
- `week_cos = cos(2π × semana / 52)`
- `is_rainy_season`: 1 si semana 18-44 (mayo-octubre)

---

## Arquitectura del Ensemble

### Random Forest (peso 60%)

**Por qué Random Forest:**
- Robusto a overfitting con muchas features
- Maneja relaciones no lineales entre lluvia y casos
- Proporciona feature importance interpretable
- Excelente con datos tabulares estructurados

**Configuración:**
```python
RandomForestRegressor(
    n_estimators=200,      # Número de árboles
    max_depth=15,          # Profundidad máxima
    min_samples_split=5,   # Muestras mínimas para dividir nodo
    min_samples_leaf=2,    # Muestras mínimas en hoja
    max_features="sqrt",   # Aleatoriedad en selección de features
    random_state=42,
    n_jobs=-1,             # Paralelo en todos los CPUs
)
```

**Multi-Output:** Se predice el riesgo de las 4 semanas simultáneamente usando `MultiOutputRegressor`.

**Validación:** `TimeSeriesSplit(n_splits=5)` — respeta el orden temporal, sin data leakage.

### LSTM Encoder-Decoder (peso 40%)

**Por qué LSTM:**
- Captura dependencias de largo plazo (patrones de múltiples semanas)
- Arquitectura encoder-decoder natural para predicción multi-step
- Aprende patrones estacionales complejos

**Arquitectura:**
```
Input: (batch, 12 semanas, N features)
  ↓
LSTM(128) → BatchNorm → Dropout(0.2)
  ↓
LSTM(64) → BatchNorm → Dropout(0.2)
  ↓
RepeatVector(4)  ← forzar 4 timesteps de salida
  ↓
LSTM(64, return_sequences=True) → Dropout(0.2)
LSTM(32, return_sequences=True)
  ↓
TimeDistributed(Dense(32, relu))
TimeDistributed(Dense(1, sigmoid))
  ↓
Output: (batch, 4, 1) → squeeze → (batch, 4) valores [0,1] × 100
```

**Función de pérdida:** Huber loss — robusta a outliers epidémicos.

**Callbacks:**
- `EarlyStopping(patience=15)`: detiene cuando val_loss no mejora
- `ReduceLROnPlateau(factor=0.5, patience=7)`: ajusta learning rate
- `ModelCheckpoint`: guarda el mejor modelo

### Combinación del Ensemble

```python
prediction_final = 0.6 × rf_prediction + 0.4 × lstm_prediction
```

Los pesos reflejan que RF es más estable con datos limitados, mientras LSTM aporta patrones temporales únicos.

---

## Validación del Modelo

### Metodología de Split Temporal

```
Datos totales (2018-2026)
├── Entrenamiento: 70% (datos más antiguos)
├── Validación: 15% (para tuning y early stopping)
└── Test: 15% (datos más recientes, evaluación final)
```

**Sin shuffle** — el orden temporal es fundamental para series epidemiológicas.

### Métricas

| Métrica | Descripción | Objetivo |
|---------|-------------|---------|
| MAE | Error absoluto medio en el índice 0-100 | < 8.0 |
| RMSE | Raíz del error cuadrático medio | < 12.0 |
| R² | Coeficiente de determinación | > 0.85 |
| Accuracy | 100 - MAE_normalizado | ≥ 88% |

### Re-entrenamiento automático

El sistema verifica cada lunes si la accuracy del ensemble cayó por debajo del 88%. Si es así, descarga datos nuevos y re-entrena automáticamente.

---

## Limitaciones y Consideraciones

1. **Datos de entrenamiento**: El rendimiento depende de la calidad y completitud de los datos históricos de PAHO y ONAMET. Con datos faltantes, el sistema usa interpolación KNN.

2. **Granularidad**: El modelo predice a nivel provincial, no municipal. La heterogeneidad intra-provincial no se captura.

3. **Serotipos**: La circulación simultánea de múltiples serotipos DENV puede generar brotes atípicos difíciles de predecir solo con variables climáticas.

4. **Lag temporal de datos**: Los datos oficiales suelen publicarse con 1-2 semanas de retraso, lo que puede afectar las predicciones a corto plazo.

5. **Cambio climático**: El modelo es re-entrenado periódicamente para adaptarse a cambios en patrones climáticos de largo plazo.

6. **Movilidad humana**: No se incluyen datos de movilidad (turismo, migraciones), que pueden ser vectores de introducción de nuevos serotipos.
