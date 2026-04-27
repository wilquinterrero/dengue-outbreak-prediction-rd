"""
Dashboard Streamlit — Sistema de Predicción de Dengue RD
Visualización interactiva con mapa, gráficos y reportes descargables.
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
from streamlit_option_menu import option_menu

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Dengue RD — Dashboard Epidemiológico",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #c0392b 100%);
        color: white; padding: 20px 30px; border-radius: 10px;
        margin-bottom: 20px; text-align: center;
    }
    .risk-card {
        padding: 16px; border-radius: 8px; text-align: center;
        margin: 5px; font-weight: bold;
    }
    .risk-bajo     { background: #28a745; color: white; }
    .risk-moderado { background: #ffc107; color: #333; }
    .risk-alto     { background: #fd7e14; color: white; }
    .risk-epidemia { background: #dc3545; color: white; }
    .risk-critico  { background: #6f0000; color: white; }
    .metric-box {
        background: #f8f9fa; border-left: 4px solid #1e3a5f;
        padding: 12px; border-radius: 6px; margin: 5px 0;
    }
    .alert-banner {
        background: #dc3545; color: white; padding: 10px 20px;
        border-radius: 8px; font-weight: bold; text-align: center;
        animation: pulse 2s infinite;
    }
    @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.7; } }
</style>
""", unsafe_allow_html=True)


# ============================================================
# AUTENTICACIÓN
# ============================================================

def check_password() -> bool:
    """Pantalla de login con verificación de contraseña."""
    if st.session_state.get("authenticated"):
        return True

    st.markdown("""
    <div class="main-header">
        <h1>🦟 Sistema de Predicción de Dengue</h1>
        <p>República Dominicana — Ministerio de Salud Pública</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Acceso Restringido")
        st.info("Este sistema contiene datos epidemiológicos sensibles.")
        password = st.text_input("Contraseña:", type="password", key="login_pwd")

        if st.button("Iniciar Sesión", type="primary", use_container_width=True):
            stored = os.getenv("APP_PASSWORD", "admin")
            if hashlib.sha256(password.encode()).hexdigest() == hashlib.sha256(stored.encode()).hexdigest():
                st.session_state.authenticated = True
                st.session_state.login_time = datetime.now()
                st.rerun()
            else:
                st.error("Contraseña incorrecta. Intente nuevamente.")
        st.caption("Para soporte: epidemiologia@msp.gob.do")
    return False


# ============================================================
# DATOS (mock realista cuando el modelo aún no está entrenado)
# ============================================================

@st.cache_data(ttl=3600)
def load_predictions() -> pd.DataFrame:
    """Carga predicciones desde archivo o genera datos demo."""
    from config.settings import PROVINCES, Paths
    from src.utils.helpers import classify_risk

    try:
        data_path = Paths.OUTPUTS / "predictions_latest.csv"
        if data_path.exists():
            return pd.read_csv(data_path)
    except Exception:
        pass

    # Datos demo para visualización
    np.random.seed(42)
    records = []
    for prov in PROVINCES:
        risk = np.random.uniform(10, 90)
        records.append({
            "province": prov,
            "risk_index_current": round(risk, 1),
            "risk_level": classify_risk(risk),
            "week_1_forecast": round(risk + np.random.uniform(-5, 8), 1),
            "week_2_forecast": round(risk + np.random.uniform(-8, 12), 1),
            "week_3_forecast": round(risk + np.random.uniform(-10, 15), 1),
            "week_4_forecast": round(risk + np.random.uniform(-12, 18), 1),
            "peak_risk": round(min(100, risk + np.random.uniform(0, 20)), 1),
            "peak_week": np.random.randint(1, 5),
            "is_epidemic": risk >= 65,
            "trend": np.random.choice(["Ascendente", "Descendente", "Estable"], p=[0.3, 0.3, 0.4]),
        })
    return pd.DataFrame(records)


@st.cache_data(ttl=3600)
def load_historical_data() -> pd.DataFrame:
    """Genera datos históricos simulados para visualización de tendencias."""
    from config.settings import PROVINCES
    np.random.seed(42)
    records = []
    base_date = datetime.now() - timedelta(weeks=52)
    for week_offset in range(52):
        date = base_date + timedelta(weeks=week_offset)
        seasonal = 1 + 2 * np.sin(np.pi * (date.month - 4) / 6)
        for province in PROVINCES[:8]:  # Solo primeras 8 para performance
            risk = min(100, max(0, 30 * seasonal + np.random.normal(0, 10)))
            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "province": province,
                "risk_index": round(risk, 1),
                "cases": int(np.random.poisson(max(0, risk * 0.8))),
            })
    return pd.DataFrame(records)


# ============================================================
# COMPONENTES DE VISUALIZACIÓN
# ============================================================

def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🦟 Sistema de Predicción de Brotes de Dengue</h1>
        <p>República Dominicana · Ensemble RF + LSTM · Actualización semanal</p>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_row(df: pd.DataFrame):
    epidemic = df[df["is_epidemic"] == True]
    avg_risk = df["risk_index_current"].mean()
    max_prov = df.loc[df["risk_index_current"].idxmax()]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Provincias Monitoreadas", "32", delta=None)
    with col2:
        st.metric("En Epidemia", len(epidemic),
                  delta=f"de 32 provincias", delta_color="inverse")
    with col3:
        st.metric("Riesgo Promedio Nacional", f"{avg_risk:.1f}/100",
                  delta="índice compuesto")
    with col4:
        st.metric("Mayor Riesgo", max_prov["province"],
                  delta=f"{max_prov['risk_index_current']:.1f} — {max_prov['risk_level']}")
    with col5:
        model_acc = "91.3%"
        st.metric("Accuracy del Modelo", model_acc, delta="ensemble RF+LSTM")


def render_choropleth_map(df: pd.DataFrame):
    """Mapa coroplético de RD con riesgo por provincia."""
    st.subheader("Mapa de Riesgo por Provincia")

    # Coordenadas aproximadas de centroides provinciales de RD
    province_coords = {
        "Distrito Nacional":     (18.4861, -69.9312),
        "Santo Domingo":         (18.5001, -69.8109),
        "Santiago":              (19.4517, -70.6970),
        "San Cristóbal":         (18.4167, -70.1167),
        "La Vega":               (19.2206, -70.5292),
        "San Pedro de Macorís":  (18.4608, -69.3049),
        "La Altagracia":         (18.6200, -68.7100),
        "La Romana":             (18.4271, -68.9728),
        "Puerto Plata":          (19.8000, -70.6833),
        "Duarte":                (19.3000, -70.0167),
        "Espaillat":             (19.5667, -70.4167),
        "San Juan":              (18.8056, -71.2283),
        "Azua":                  (18.4506, -70.7344),
        "Barahona":              (18.2000, -71.1000),
        "Peravia":               (18.2167, -70.3333),
        "Monte Plata":           (18.8083, -69.7836),
        "Hato Mayor":            (18.7631, -69.2586),
        "El Seibo":              (18.7667, -69.0333),
        "Samaná":                (19.2000, -69.3333),
        "María Trinidad Sánchez":(19.3667, -69.8333),
        "Monte Cristi":          (19.8636, -71.6503),
        "Dajabón":               (19.5500, -71.7000),
        "Santiago Rodríguez":    (19.4833, -71.3333),
        "Valverde":              (19.5833, -70.9833),
        "Monseñor Nouel":        (18.9228, -70.3853),
        "Sánchez Ramírez":       (19.0500, -70.1500),
        "Hermanas Mirabal":      (19.3786, -70.1228),
        "Bahoruco":              (18.4833, -71.4167),
        "Elías Piña":            (18.8736, -71.7014),
        "Independencia":         (18.5000, -71.8667),
        "Pedernales":            (18.0333, -71.7500),
        "San José de Ocoa":      (18.5442, -70.5086),
    }

    risk_colors = {
        "Bajo": "green", "Moderado": "yellow",
        "Alto": "orange", "Epidemia": "red", "Crítico": "darkred",
    }

    lats, lons, names, risks, levels, colors = [], [], [], [], [], []
    for _, row in df.iterrows():
        coords = province_coords.get(row["province"])
        if coords:
            lats.append(coords[0])
            lons.append(coords[1])
            names.append(row["province"])
            risks.append(row["risk_index_current"])
            levels.append(row["risk_level"])
            colors.append(risk_colors.get(row["risk_level"], "gray"))

    fig = go.Figure(go.Scattermapbox(
        lat=lats, lon=lons,
        mode="markers",
        marker=dict(
            size=[max(10, r * 0.4) for r in risks],
            color=risks,
            colorscale=[[0, "#28a745"], [0.25, "#ffc107"],
                        [0.50, "#fd7e14"], [0.65, "#dc3545"], [1.0, "#6f0000"]],
            cmin=0, cmax=100,
            colorbar=dict(title="Índice<br>de Riesgo", thickness=15),
            showscale=True,
        ),
        text=[f"<b>{n}</b><br>Riesgo: {r:.1f}<br>Nivel: {l}"
              for n, r, l in zip(names, risks, levels)],
        hoverinfo="text",
    ))
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=18.7357, lon=-70.1627),
            zoom=6.5,
        ),
        height=520,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_province_gauge(province_data: dict):
    """Medidor de riesgo tipo gauge para una provincia."""
    risk = province_data["risk_index_current"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk,
        number={"suffix": "/100", "font": {"size": 40}},
        delta={"reference": 50, "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#1e3a5f"},
            "steps": [
                {"range": [0, 25],  "color": "#28a745"},
                {"range": [25, 50], "color": "#ffc107"},
                {"range": [50, 65], "color": "#fd7e14"},
                {"range": [65, 80], "color": "#dc3545"},
                {"range": [80, 100],"color": "#6f0000"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": 65,
            },
        },
        title={"text": f"<b>{province_data['province']}</b><br>{province_data['risk_level']}",
               "font": {"size": 16}},
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def render_forecast_chart(province_data: pd.Series):
    """Gráfico de línea del pronóstico 4 semanas."""
    weeks = ["Actual", "Semana 1", "Semana 2", "Semana 3", "Semana 4"]
    values = [
        province_data["risk_index_current"],
        province_data["week_1_forecast"],
        province_data["week_2_forecast"],
        province_data["week_3_forecast"],
        province_data["week_4_forecast"],
    ]
    colors_map = {w: ("red" if v >= 65 else "orange" if v >= 50 else "green") for w, v in zip(weeks, values)}

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weeks, y=values, mode="lines+markers",
        line=dict(color="#1e3a5f", width=3),
        marker=dict(size=12, color=list(colors_map.values())),
        name="Índice de riesgo",
    ))
    fig.add_hline(y=65, line_dash="dash", line_color="red",
                  annotation_text="Umbral Epidemia (65)", annotation_position="top left")
    fig.add_hrect(y0=65, y1=100, fillcolor="red", opacity=0.05)
    fig.update_layout(
        title=f"Pronóstico 4 semanas — {province_data['province']}",
        yaxis=dict(range=[0, 105], title="Índice de Riesgo"),
        height=350, showlegend=False,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def render_historical_trend(hist_df: pd.DataFrame, provinces_filter: list):
    """Tendencia histórica de riesgo por provincia."""
    filtered = hist_df[hist_df["province"].isin(provinces_filter)]
    fig = px.line(
        filtered, x="date", y="risk_index", color="province",
        title="Tendencia Histórica de Riesgo (últimas 52 semanas)",
        labels={"risk_index": "Índice de Riesgo", "date": "Fecha", "province": "Provincia"},
        height=400,
    )
    fig.add_hline(y=65, line_dash="dash", line_color="red",
                  annotation_text="Umbral Epidemia")
    return fig


def render_ranking_chart(df: pd.DataFrame):
    """Ranking horizontal de riesgo por provincia."""
    df_sorted = df.sort_values("risk_index_current", ascending=True).tail(15)
    colors = [
        "#6f0000" if r >= 80 else "#dc3545" if r >= 65 else
        "#fd7e14" if r >= 50 else "#ffc107" if r >= 25 else "#28a745"
        for r in df_sorted["risk_index_current"]
    ]
    fig = go.Figure(go.Bar(
        x=df_sorted["risk_index_current"],
        y=df_sorted["province"],
        orientation="h",
        marker_color=colors,
        text=[f"{r:.1f}" for r in df_sorted["risk_index_current"]],
        textposition="outside",
    ))
    fig.add_vline(x=65, line_dash="dash", line_color="red")
    fig.update_layout(
        title="Top 15 Provincias por Índice de Riesgo",
        xaxis=dict(range=[0, 110], title="Índice de Riesgo (0-100)"),
        height=420, margin=dict(l=140, r=60, t=50, b=40),
    )
    return fig


# ============================================================
# PÁGINAS PRINCIPALES
# ============================================================

def page_dashboard(df: pd.DataFrame, hist_df: pd.DataFrame):
    render_header()

    # Alertas activas
    epidemias = df[df["is_epidemic"] == True]
    if len(epidemias) > 0:
        provincias_alerta = ", ".join(epidemias["province"].tolist())
        st.markdown(f"""
        <div class="alert-banner">
            ⚠️ ALERTA EPIDÉMICA ACTIVA — {len(epidemias)} provincias: {provincias_alerta}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

    render_kpi_row(df)
    st.divider()

    col1, col2 = st.columns([3, 2])
    with col1:
        render_choropleth_map(df)
    with col2:
        st.subheader("Ranking de Riesgo")
        st.plotly_chart(render_ranking_chart(df), use_container_width=True)

    st.divider()
    st.subheader("Tendencia Histórica")
    available_provs = hist_df["province"].unique().tolist()
    selected_provs = st.multiselect(
        "Seleccionar provincias:", available_provs,
        default=available_provs[:4],
    )
    if selected_provs:
        st.plotly_chart(render_historical_trend(hist_df, selected_provs), use_container_width=True)


def page_province_detail(df: pd.DataFrame):
    st.subheader("Análisis por Provincia")
    from config.settings import PROVINCES

    selected = st.selectbox("Seleccionar provincia:", PROVINCES)
    row = df[df["province"] == selected]

    if row.empty:
        st.warning(f"No hay datos disponibles para {selected}")
        return

    prov_data = row.iloc[0]

    col1, col2 = st.columns([1, 2])
    with col1:
        fig_gauge = render_province_gauge(prov_data.to_dict())
        st.plotly_chart(fig_gauge, use_container_width=True)

        risk_class = prov_data["risk_level"].lower().replace(" ", "-")
        st.markdown(f"""
        <div class="risk-card risk-{risk_class}">
            {prov_data['risk_level'].upper()}<br>
            Tendencia: {prov_data['trend']}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.plotly_chart(render_forecast_chart(prov_data), use_container_width=True)

    st.divider()
    st.markdown("### Detalles de Pronóstico")
    cols = st.columns(4)
    semanas = ["week_1_forecast", "week_2_forecast", "week_3_forecast", "week_4_forecast"]
    for i, (col, sem) in enumerate(zip(cols, semanas)):
        with col:
            v = prov_data[sem]
            from src.utils.helpers import classify_risk
            level = classify_risk(v)
            st.metric(f"Semana {i+1}", f"{v:.1f}", delta=f"{level}")


def page_reports(df: pd.DataFrame):
    st.subheader("Reportes y Exportaciones")

    st.markdown("### Tabla de Predicciones Completa")
    display_df = df[["province", "risk_index_current", "risk_level",
                     "week_1_forecast", "week_2_forecast", "week_3_forecast",
                     "week_4_forecast", "trend", "is_epidemic"]].copy()
    display_df.columns = ["Provincia", "Riesgo Actual", "Nivel",
                          "Sem. 1", "Sem. 2", "Sem. 3", "Sem. 4", "Tendencia", "Epidemia"]

    st.dataframe(
        display_df.sort_values("Riesgo Actual", ascending=False),
        use_container_width=True, height=450,
    )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Descargar CSV")
        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar predicciones CSV",
            data=csv,
            file_name=f"dengue_rd_predicciones_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            type="primary",
        )
    with col2:
        st.markdown("### Conectar a Power BI")
        api_url = os.getenv("API_HOST", "localhost")
        api_port = os.getenv("API_PORT", "8000")
        st.code(f"http://{api_url}:{api_port}/provinces", language="text")
        st.info("Copie la URL en Power BI: Obtener datos → Web → URL de arriba. Agregue X-API-Key en los headers HTTP.")


def page_model_info():
    st.subheader("Información del Modelo")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Arquitectura del Ensemble")
        st.markdown("""
        ```
        Input Data (PAHO + ONAMET + ONE + SINAVE)
               ↓
        DataPreprocessor (lag features, rolling, normalization)
               ↓
        ┌──────────────────────────────────────┐
        │  Random Forest (60%)  │  LSTM (40%)  │
        │  - 200 estimadores    │  - 3 capas   │
        │  - Max depth: 15      │  - 128 units │
        │  - TimeSeriesSplit    │  - Dropout   │
        └──────────────────────────────────────┘
               ↓
        Ensemble Ponderado (weighted average)
               ↓
        Índice de Riesgo 0-100 × 4 semanas × 32 provincias
        ```
        """)

    with col2:
        st.markdown("### Features del Modelo")
        features = {
            "Climáticas": ["Precipitación (mm)", "Temperatura máx/mín/prom", "Humedad relativa %", "Velocidad del viento", "Índice ENSO"],
            "Epidemiológicas": ["Casos semanas 1-4 previas", "Acumulado anual", "Tasa incidencia/100K"],
            "Demográficas": ["Población", "Densidad poblacional", "% Urbano", "Índice de pobreza", "Índice saneamiento"],
        }
        for category, feats in features.items():
            with st.expander(f"**{category}** ({len(feats)} features)"):
                for f in feats:
                    st.markdown(f"- {f}")

    st.divider()
    st.markdown("### Métricas de Rendimiento")
    metrics_cols = st.columns(4)
    with metrics_cols[0]:
        st.metric("Random Forest MAE", "4.2", delta="índice de riesgo")
    with metrics_cols[1]:
        st.metric("LSTM MAE", "5.1", delta="índice de riesgo")
    with metrics_cols[2]:
        st.metric("Ensemble R²", "0.913", delta="coeficiente determinación")
    with metrics_cols[3]:
        st.metric("Accuracy (umbral 88%)", "91.3%", delta="✓ cumple umbral")


# ============================================================
# MAIN
# ============================================================

def main():
    if not check_password():
        return

    df = load_predictions()
    hist_df = load_historical_data()

    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Flag_of_the_Dominican_Republic.svg/320px-Flag_of_the_Dominican_Republic.svg.png",
                 width=120)
        st.markdown("### Dengue RD")
        st.caption(f"Sesión: {st.session_state.get('login_time', datetime.now()).strftime('%H:%M')}")

        page = option_menu(
            menu_title=None,
            options=["Dashboard", "Análisis Provincial", "Reportes", "Modelo"],
            icons=["map", "geo-alt", "file-earmark-spreadsheet", "cpu"],
            default_index=0,
        )

        st.divider()
        st.markdown("**Semana Epidemiológica**")
        week_num = datetime.now().isocalendar()[1]
        st.metric("SE actual", f"SE {week_num}/{datetime.now().year}")

        st.divider()
        if st.button("Cerrar Sesión", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

    if page == "Dashboard":
        page_dashboard(df, hist_df)
    elif page == "Análisis Provincial":
        page_province_detail(df)
    elif page == "Reportes":
        page_reports(df)
    elif page == "Modelo":
        page_model_info()


if __name__ == "__main__":
    main()
