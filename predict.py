"""
Inference script: loads trained RF + MLP ensemble and generates 4-week
dengue outbreak risk forecasts for all 32 Dominican Republic provinces.
Exports Power BI-ready CSV files to outputs/.

Usage:
    python predict.py
    python predict.py --data data/raw/master_dataset.csv
"""

import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    PROVINCES, PROVINCE_CODES, FORECAST_HORIZON,
    EPIDEMIC_THRESHOLD, ALERT_THRESHOLD, Paths,
)
from src.data.preprocessing import DataPreprocessor
from src.models.random_forest import RandomForestModel
from src.models.lstm import LSTMModel

SEQUENCE_LENGTH = 12  # must match LSTM_PARAMS["sequence_length"]

PROVINCE_COORDS = {
    "Azua":                   (18.45, -70.73),
    "Bahoruco":               (18.48, -71.42),
    "Barahona":               (18.20, -71.10),
    "Dajabón":                (19.55, -71.71),
    "Duarte":                 (19.20, -70.02),
    "Elías Piña":             (18.88, -71.70),
    "El Seibo":               (18.77, -69.05),
    "Espaillat":              (19.62, -70.28),
    "Hato Mayor":             (18.77, -69.25),
    "Hermanas Mirabal":       (19.35, -70.05),
    "Independencia":          (18.52, -71.85),
    "La Altagracia":          (18.62, -68.72),
    "La Romana":              (18.43, -68.97),
    "La Vega":                (19.22, -70.53),
    "María Trinidad Sánchez": (19.47, -69.98),
    "Monseñor Nouel":         (18.92, -70.38),
    "Monte Cristi":           (19.87, -71.65),
    "Monte Plata":            (18.80, -69.78),
    "Pedernales":             (18.04, -71.75),
    "Peravia":                (18.28, -70.33),
    "Puerto Plata":           (19.80, -70.68),
    "Samaná":                 (19.20, -69.33),
    "San Cristóbal":          (18.42, -70.12),
    "San José de Ocoa":       (18.55, -70.50),
    "San Juan":               (18.81, -71.23),
    "San Pedro de Macorís":   (18.46, -69.30),
    "Sánchez Ramírez":        (19.05, -70.15),
    "Santiago":               (19.45, -70.70),
    "Santiago Rodríguez":     (19.47, -71.35),
    "Santo Domingo":          (18.45, -69.97),
    "Valverde":               (19.57, -71.07),
    "Distrito Nacional":      (18.48, -69.90),
}


def classify_risk(risk: float) -> str:
    if risk < 25:   return "Bajo"
    if risk < 50:   return "Moderado"
    if risk < 65:   return "Alto"
    if risk < 80:   return "Epidemia"
    return "Critico"


def trend_label(forecast: np.ndarray) -> str:
    if len(forecast) < 2:
        return "Estable"
    slope = np.polyfit(range(len(forecast)), forecast, 1)[0]
    if slope > 2:   return "Ascendente"
    if slope < -2:  return "Descendente"
    return "Estable"


def build_province_inputs(df_proc, province, rf_feature_names):
    """Returns (X_rf, X_lstm, last_date) for one province or (None, None, None)."""
    prov = df_proc[df_proc["province"] == province].copy()
    if "date" in prov.columns:
        prov = prov.sort_values("date")
    elif "year" in prov.columns:
        prov = prov.sort_values(["year", "week"])

    if len(prov) == 0:
        return None, None, None

    # --- RF: single row (last available) ---
    numeric_cols = [
        c for c in prov.columns
        if c not in ["outbreak_risk_index", "province", "date"]
        and prov[c].dtype.kind in "fiui"
    ]
    last_row = prov[numeric_cols].tail(1).fillna(0.0)
    # Align to exact training column order; fill unseen columns with 0
    X_rf = last_row.reindex(columns=rf_feature_names, fill_value=0.0)

    # --- LSTM: last SEQUENCE_LENGTH rows as a sequence ---
    lstm_feature_cols = [
        c for c in prov.columns
        if c not in ["outbreak_risk_index", "province", "date", "week", "year", "month"]
    ]
    tail = prov.tail(SEQUENCE_LENGTH)
    if len(tail) < SEQUENCE_LENGTH:
        pad = pd.concat([tail.head(1)] * (SEQUENCE_LENGTH - len(tail)), ignore_index=True)
        tail = pd.concat([pad, tail], ignore_index=True)
    feat_arr = tail[lstm_feature_cols].fillna(0.0).values.astype(np.float32)
    X_lstm = feat_arr.reshape(1, SEQUENCE_LENGTH, len(lstm_feature_cols))

    last_date = prov["date"].iloc[-1] if "date" in prov.columns else None
    return X_rf, X_lstm, last_date


def main():
    parser = argparse.ArgumentParser(description="Generate dengue predictions for all provinces")
    parser.add_argument("--data", default=str(Paths.DATA_RAW / "master_dataset.csv"),
                        help="Path to master_dataset.csv")
    args = parser.parse_args()

    Paths.ensure_dirs()

    # ------------------------------------------------------------------ #
    # 1. Load raw data                                                     #
    # ------------------------------------------------------------------ #
    raw_path = Path(args.data)
    if not raw_path.exists():
        print(f"ERROR: dataset not found at {raw_path}. Run make_dataset.py first.")
        sys.exit(1)

    print(f"Loading dataset: {raw_path}")
    df_raw = pd.read_csv(raw_path)
    print(f"  {len(df_raw)} rows x {len(df_raw.columns)} columns")

    # ------------------------------------------------------------------ #
    # 2. Preprocess using fitted scalers/imputer from training            #
    # ------------------------------------------------------------------ #
    print("Applying preprocessing pipeline (fitted from training)...")
    preprocessor = DataPreprocessor()
    df_proc = preprocessor.transform(df_raw)
    print(f"  Processed shape: {df_proc.shape}")

    # ------------------------------------------------------------------ #
    # 3. Load trained models                                               #
    # ------------------------------------------------------------------ #
    print("Loading trained models...")
    rf_model = RandomForestModel()
    rf_model._load()

    lstm_model = LSTMModel()
    lstm_model._load()

    # Target scaler: inverse-transforms RF predictions from [0,1] → [0,100]
    target_scaler = joblib.load(Paths.MODELS / "target_scaler.pkl")

    print(f"  RF feature count: {len(rf_model.feature_names)}")
    mlp_input = rf_model.feature_names  # reuse this reference

    # ------------------------------------------------------------------ #
    # 4. Generate predictions province by province                        #
    # ------------------------------------------------------------------ #
    prediction_date = datetime.now()
    forecast_dates = [
        (prediction_date + timedelta(weeks=i + 1)).strftime("%Y-%m-%d")
        for i in range(FORECAST_HORIZON)
    ]

    print(f"\nForecasting {len(PROVINCES)} provinces (4-week horizon)...\n")

    summary_rows   = []
    forecast_rows  = []
    alert_rows     = []
    history_rows   = []   # province-level aggregated history for trend charts

    for idx, province in enumerate(PROVINCES, 1):
        X_rf, X_lstm, last_date = build_province_inputs(df_proc, province, rf_model.feature_names)

        if X_rf is None:
            print(f"  [{idx:02d}/{len(PROVINCES)}] {province}: NO DATA")
            continue

        # RF predicts scaled [0,1] values → inverse transform to [0,100]
        rf_scaled = rf_model.model.predict(X_rf)            # (1, 4)
        rf_pred = np.clip(
            target_scaler.inverse_transform(
                rf_scaled.flatten()[:FORECAST_HORIZON].reshape(-1, 1)
            ).flatten(),
            0, 100,
        )

        # MLP predicts [0,100] values directly (scaled * 100 inside lstm.predict)
        lstm_pred = lstm_model.predict(X_lstm)[0]           # (4,)

        # Weighted ensemble: 60% RF + 40% MLP
        ensemble = np.clip(0.6 * rf_pred + 0.4 * lstm_pred, 0, 100)

        current_risk = float(round(ensemble[0], 2))
        peak_risk    = float(round(float(np.max(ensemble)), 2))
        peak_week    = int(np.argmax(ensemble) + 1)
        trend        = trend_label(ensemble)
        risk_level   = classify_risk(current_risk)
        lat, lon     = PROVINCE_COORDS.get(province, (18.7, -70.2))
        prov_code    = PROVINCE_CODES.get(province, "RD00")

        # -- Summary row --
        summary_rows.append({
            "province_code":    prov_code,
            "province":         province,
            "latitude":         lat,
            "longitude":        lon,
            "prediction_date":  prediction_date.strftime("%Y-%m-%d"),
            "last_data_date":   str(last_date)[:10] if last_date else "",
            "risk_index":       current_risk,
            "risk_level":       risk_level,
            "week_1_forecast":  float(round(ensemble[0], 2)),
            "week_2_forecast":  float(round(ensemble[1], 2)),
            "week_3_forecast":  float(round(ensemble[2], 2)),
            "week_4_forecast":  float(round(ensemble[3], 2)),
            "peak_risk":        peak_risk,
            "peak_week":        peak_week,
            "trend":            trend,
            "is_epidemic":      current_risk >= EPIDEMIC_THRESHOLD,
            "is_alert":         current_risk >= ALERT_THRESHOLD,
            "rf_week1":         float(round(rf_pred[0], 2)),
            "mlp_week1":        float(round(lstm_pred[0], 2)),
        })

        # -- Weekly forecast rows (long format) --
        for w in range(FORECAST_HORIZON):
            forecast_rows.append({
                "province_code": prov_code,
                "province":      province,
                "forecast_week": w + 1,
                "forecast_date": forecast_dates[w],
                "risk_index":    float(round(ensemble[w], 2)),
                "rf_risk":       float(round(rf_pred[w], 2)),
                "mlp_risk":      float(round(lstm_pred[w], 2)),
                "risk_level":    classify_risk(float(ensemble[w])),
            })

        # -- Alerts --
        if current_risk >= EPIDEMIC_THRESHOLD:
            alert_rows.append({
                "province_code":   prov_code,
                "province":        province,
                "latitude":        lat,
                "longitude":       lon,
                "risk_index":      current_risk,
                "risk_level":      risk_level,
                "alert_type":      "Critico" if current_risk >= ALERT_THRESHOLD else "Epidemia",
                "prediction_date": prediction_date.strftime("%Y-%m-%d"),
                "trend":           trend,
                "week_2_forecast": float(round(ensemble[1], 2)),
                "week_3_forecast": float(round(ensemble[2], 2)),
                "week_4_forecast": float(round(ensemble[3], 2)),
            })

        print(f"  [{idx:02d}/{len(PROVINCES)}] {province:<25} "
              f"Risk: {current_risk:5.1f}  Level: {risk_level:<10}  Trend: {trend}")

    # ------------------------------------------------------------------ #
    # 5. Build historical time-series table from raw data                 #
    # ------------------------------------------------------------------ #
    hist_keep = [
        "province", "date", "year", "week", "month",
        "cases", "deaths", "incidence_rate_100k", "outbreak_risk_index",
        "rainfall_mm", "temp_avg_c", "humidity_pct",
        "population", "urban_pct", "poverty_index",
    ]
    hist_df = df_raw[[c for c in hist_keep if c in df_raw.columns]].copy()
    hist_df.insert(0, "province_code", hist_df["province"].map(PROVINCE_CODES).fillna("RD00"))

    # ------------------------------------------------------------------ #
    # 6. Export CSVs (utf-8-sig BOM for Power BI / Excel compatibility)   #
    # ------------------------------------------------------------------ #
    out = Paths.OUTPUTS

    # Sort summary by risk descending before saving
    summary_rows.sort(key=lambda r: r["risk_index"], reverse=True)
    alert_rows.sort(key=lambda r: r["risk_index"], reverse=True)

    exports = []

    summary_df = pd.DataFrame(summary_rows)
    p = out / "predictions_summary.csv"
    summary_df.to_csv(p, index=False, encoding="utf-8-sig")
    exports.append((p, len(summary_df), "Main forecast table (1 row / province)"))

    forecast_df = pd.DataFrame(forecast_rows)
    p = out / "predictions_forecast_weekly.csv"
    forecast_df.to_csv(p, index=False, encoding="utf-8-sig")
    exports.append((p, len(forecast_df), "Long-format 4-week forecast (4 rows / province)"))

    hist_df_sorted = hist_df.sort_values(["province", "date"] if "date" in hist_df.columns else ["province", "year", "week"])
    p = out / "predictions_historical.csv"
    hist_df_sorted.to_csv(p, index=False, encoding="utf-8-sig")
    exports.append((p, len(hist_df_sorted), "Historical epidemiological time series"))

    if alert_rows:
        alerts_df = pd.DataFrame(alert_rows)
        p = out / "predictions_alerts.csv"
        alerts_df.to_csv(p, index=False, encoding="utf-8-sig")
        exports.append((p, len(alerts_df), "Active epidemic / critical alerts"))

    # Risk band pivot table (useful for Power BI matrix visual)
    band_counts = summary_df.groupby("risk_level")["province"].count().reset_index()
    band_counts.columns = ["risk_level", "province_count"]
    risk_order = {"Bajo": 1, "Moderado": 2, "Alto": 3, "Epidemia": 4, "Critico": 5}
    band_counts["sort_order"] = band_counts["risk_level"].map(risk_order).fillna(9)
    band_counts = band_counts.sort_values("sort_order").drop(columns="sort_order")
    p = out / "predictions_risk_bands.csv"
    band_counts.to_csv(p, index=False, encoding="utf-8-sig")
    exports.append((p, len(band_counts), "Province count per risk band (KPI card / bar chart)"))

    # ------------------------------------------------------------------ #
    # 7. Print final summary                                               #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 65)
    print("FORECAST SUMMARY — all 32 provinces")
    print("=" * 65)
    print(f"  {'Province':<26} {'Risk':>6}  {'Level':<10}  Trend")
    print("  " + "-" * 57)
    for row in summary_rows:
        epic_flag = " *" if row["is_epidemic"] else ""
        print(f"  {row['province']:<26} {row['risk_index']:>6.1f}  "
              f"{row['risk_level']:<10}  {row['trend']}{epic_flag}")
    print("=" * 65)

    epidemic_n = sum(1 for r in summary_rows if r["is_epidemic"])
    alert_n    = sum(1 for r in summary_rows if r["is_alert"])
    print(f"\n  Provinces at epidemic level  (>={EPIDEMIC_THRESHOLD}): {epidemic_n}")
    print(f"  Provinces at critical alert  (>={ALERT_THRESHOLD}): {alert_n}")

    print("\n" + "=" * 65)
    print("EXPORTED FILES")
    print("=" * 65)
    for path, n_rows, desc in exports:
        print(f"  {path.name:<40}  {n_rows:>5} rows  —  {desc}")
    print(f"\n  Folder: {out}")
    print("Done.")


if __name__ == "__main__":
    main()
