"""
Generate publication-ready visualizations for the dengue prediction project.
Saves 4 high-resolution PNGs to visuals/.

Usage:
    python generate_visuals.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).parent
OUT    = ROOT / "visuals"
OUT.mkdir(exist_ok=True)

SUMMARY  = ROOT / "outputs" / "predictions_summary.csv"
HIST     = ROOT / "outputs" / "predictions_historical.csv"
FORECAST = ROOT / "outputs" / "predictions_forecast_weekly.csv"

# ── Global theme ─────────────────────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
GRID    = "#21262d"
TEXT    = "#e6edf3"
SUBTEXT = "#8b949e"
ACCENT  = "#58a6ff"

RISK_COLORS = {
    "Bajo":     "#2ea043",
    "Moderado": "#d29922",
    "Alto":     "#e3b341",
    "Epidemia": "#f85149",
    "Critico":  "#da3633",
}

def apply_dark_theme(fig, axes=None):
    fig.patch.set_facecolor(BG)
    if axes is None:
        return
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=SUBTEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.6, linestyle="--", alpha=0.7)

def watermark(fig):
    fig.text(0.99, 0.01, "Dengue Outbreak Prediction · Dominican Republic · 2026",
             ha="right", va="bottom", fontsize=7, color=SUBTEXT, style="italic")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Province Risk Bar Chart (choropleth-style)
# ══════════════════════════════════════════════════════════════════════════════
def plot_province_risk_bar(df_summary: pd.DataFrame):
    df = df_summary.sort_values("risk_index", ascending=True).copy()
    colors = [RISK_COLORS.get(lvl, "#8b949e") for lvl in df["risk_level"]]

    fig, ax = plt.subplots(figsize=(14, 11))
    apply_dark_theme(fig, ax)

    bars = ax.barh(df["province"], df["risk_index"], color=colors,
                   edgecolor=BG, linewidth=0.4, height=0.72)

    # Value labels
    for bar, val, lvl in zip(bars, df["risk_index"], df["risk_level"]):
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", ha="left",
                fontsize=8.5, color=TEXT, fontweight="bold")
        # Risk-level badge on the left
        ax.text(-1.2, bar.get_y() + bar.get_height() / 2,
                lvl, va="center", ha="right",
                fontsize=7.5, color=RISK_COLORS.get(lvl, TEXT),
                fontweight="bold")

    # Risk zone guide lines
    for threshold, label, color in [
        (25, "Moderado ≥25", RISK_COLORS["Moderado"]),
        (50, "Alto ≥50",     RISK_COLORS["Alto"]),
        (65, "Epidemia ≥65", RISK_COLORS["Epidemia"]),
        (80, "Crítico ≥80",  RISK_COLORS["Critico"]),
    ]:
        ax.axvline(threshold, color=color, linewidth=0.8, linestyle=":", alpha=0.6)
        ax.text(threshold + 0.5, len(df) - 0.5, label,
                fontsize=7, color=color, va="top", alpha=0.8)

    ax.set_xlim(-8, 105)
    ax.set_xlabel("Índice de Riesgo (0–100)", fontsize=11, labelpad=8)
    ax.set_title("Índice de Riesgo de Dengue por Provincia\nRepública Dominicana — Semana 1 Pronóstico",
                 fontsize=14, fontweight="bold", color=TEXT, pad=16)

    # Legend
    legend_patches = [mpatches.Patch(facecolor=c, label=l, edgecolor=GRID)
                      for l, c in RISK_COLORS.items()]
    ax.legend(handles=legend_patches, loc="lower right", framealpha=0.15,
              facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

    ax.tick_params(axis="y", labelsize=9.5)
    ax.set_axisbelow(True)
    watermark(fig)
    fig.tight_layout()
    path = OUT / "01_province_risk_bar.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Seasonal Trend — dengue cases by month with shaded area
# ══════════════════════════════════════════════════════════════════════════════
def plot_seasonal_trend(df_hist: pd.DataFrame):
    MONTH_NAMES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                   "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

    monthly = (
        df_hist.groupby(["year", "month"])["cases"]
        .sum()
        .reset_index()
    )

    # Pivot years as separate series
    pivot = monthly.pivot(index="month", columns="year", values="cases").fillna(0)
    mean_cases = pivot.mean(axis=1)
    std_cases  = pivot.std(axis=1).fillna(0)

    months = np.arange(1, 13)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9),
                             gridspec_kw={"height_ratios": [2.5, 1]})
    ax_main, ax_heat = axes
    apply_dark_theme(fig, axes)

    # Shaded band (mean ± std)
    ax_main.fill_between(months, mean_cases - std_cases, mean_cases + std_cases,
                         alpha=0.18, color=ACCENT)

    # Individual year lines
    year_colors = ["#388bfd", "#56d364", "#e3b341", "#f85149"]
    for (yr, col) in zip(sorted(pivot.columns), year_colors):
        ax_main.plot(months, pivot[yr], color=col, linewidth=1.2,
                     alpha=0.55, linestyle="--", label=str(yr))

    # Mean line
    ax_main.plot(months, mean_cases, color=ACCENT, linewidth=2.5,
                 label="Promedio 2024–2025", zorder=5)
    ax_main.fill_between(months, 0, mean_cases, alpha=0.08, color=ACCENT)

    # Rainy season shading (May–Oct)
    ax_main.axvspan(5, 10, alpha=0.06, color="#56d364", label="Temporada lluviosa")

    ax_main.set_xticks(months)
    ax_main.set_xticklabels(MONTH_NAMES, fontsize=10)
    ax_main.set_ylabel("Casos de Dengue (total nacional)", fontsize=11)
    ax_main.set_title("Tendencia Estacional del Dengue — República Dominicana",
                      fontsize=14, fontweight="bold", color=TEXT, pad=14)
    ax_main.legend(loc="upper left", framealpha=0.15, facecolor=PANEL,
                   edgecolor=GRID, labelcolor=TEXT, fontsize=9)
    ax_main.set_xlim(1, 12)

    # Heatmap of monthly cases by province (bottom panel)
    prov_month = (
        df_hist.groupby(["province", "month"])["cases"]
        .mean()
        .unstack()
        .fillna(0)
    )
    # Pick top 12 provinces by total cases
    top_provs = prov_month.sum(axis=1).nlargest(12).index
    prov_month = prov_month.loc[top_provs]

    cmap = LinearSegmentedColormap.from_list(
        "dengue", ["#0d1117", "#1f6feb", "#388bfd", "#e3b341", "#f85149"], N=256
    )
    im = ax_heat.imshow(prov_month.values, aspect="auto", cmap=cmap,
                        interpolation="nearest")
    ax_heat.set_xticks(range(12))
    ax_heat.set_xticklabels(MONTH_NAMES, fontsize=8.5)
    ax_heat.set_yticks(range(len(top_provs)))
    ax_heat.set_yticklabels(top_provs, fontsize=8)
    ax_heat.set_title("Calor de Casos por Provincia (Top 12)",
                      fontsize=10, color=TEXT, pad=8)
    cbar = fig.colorbar(im, ax=ax_heat, orientation="vertical", pad=0.01)
    cbar.ax.tick_params(colors=SUBTEXT, labelsize=8)
    cbar.set_label("Casos promedio/semana", color=SUBTEXT, fontsize=8)
    ax_heat.grid(False)

    watermark(fig)
    fig.tight_layout(h_pad=2.5)
    path = OUT / "02_seasonal_trend.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. 4-Week Forecast Chart — all 32 provinces as small multiples
# ══════════════════════════════════════════════════════════════════════════════
def plot_forecast_multiples(df_fc: pd.DataFrame, df_summary: pd.DataFrame):
    provs = df_summary.sort_values("risk_index", ascending=False)["province"].tolist()
    n = len(provs)
    ncols = 8
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(22, nrows * 2.6))
    fig.patch.set_facecolor(BG)
    axes_flat = axes.flatten()

    weeks = [1, 2, 3, 4]
    week_labels = ["S1", "S2", "S3", "S4"]

    for i, prov in enumerate(provs):
        ax = axes_flat[i]
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.tick_params(colors=SUBTEXT, labelsize=7)

        pf = df_fc[df_fc["province"] == prov].sort_values("forecast_week")
        risk_lvl = df_summary.loc[df_summary["province"] == prov, "risk_level"].values[0]
        color = RISK_COLORS.get(risk_lvl, ACCENT)

        ens  = pf["risk_index"].values
        rf   = pf["rf_risk"].values
        mlp  = pf["mlp_risk"].values

        # Shaded area between RF and MLP (uncertainty band)
        ax.fill_between(weeks, np.minimum(rf, mlp), np.maximum(rf, mlp),
                        alpha=0.2, color=color)
        ax.plot(weeks, rf,  color=SUBTEXT, linewidth=0.8, linestyle="--", alpha=0.6)
        ax.plot(weeks, mlp, color=SUBTEXT, linewidth=0.8, linestyle=":",  alpha=0.6)
        ax.plot(weeks, ens, color=color, linewidth=2.0, marker="o",
                markersize=4, zorder=5)

        # Risk zone backgrounds
        ax.axhspan(0,  25, alpha=0.04, color=RISK_COLORS["Bajo"])
        ax.axhspan(25, 50, alpha=0.04, color=RISK_COLORS["Moderado"])
        ax.axhspan(50, 65, alpha=0.04, color=RISK_COLORS["Alto"])
        ax.axhspan(65, 80, alpha=0.04, color=RISK_COLORS["Epidemia"])
        ax.axhspan(80, 100, alpha=0.04, color=RISK_COLORS["Critico"])

        ax.set_ylim(0, 100)
        ax.set_xlim(0.8, 4.2)
        ax.set_xticks(weeks)
        ax.set_xticklabels(week_labels, fontsize=7)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
        ax.grid(color=GRID, linewidth=0.4, linestyle="--", alpha=0.5)

        # Province name + current risk
        short_name = prov.replace("San Pedro de Macorís", "S.P. Macorís") \
                        .replace("María Trinidad Sánchez", "M.T. Sánchez") \
                        .replace("Santiago Rodríguez", "Stgo. Rodríguez") \
                        .replace("Hermanas Mirabal", "H. Mirabal")
        ax.set_title(f"{short_name}\n{ens[0]:.1f} · {risk_lvl}",
                     fontsize=7.5, color=color, fontweight="bold", pad=3)

    # Hide unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Shared legend
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], color=ACCENT,  linewidth=2, label="Ensemble"),
        Line2D([0], [0], color=SUBTEXT, linewidth=1, linestyle="--", label="RF"),
        Line2D([0], [0], color=SUBTEXT, linewidth=1, linestyle=":",  label="MLP"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=3,
               framealpha=0.15, facecolor=PANEL, edgecolor=GRID,
               labelcolor=TEXT, fontsize=9, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Pronóstico 4 Semanas de Riesgo de Dengue — 32 Provincias\nRepública Dominicana",
                 fontsize=15, fontweight="bold", color=TEXT, y=1.01)
    watermark(fig)
    fig.tight_layout(h_pad=1.2, w_pad=0.5)
    path = OUT / "03_forecast_multiples.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Risk Dashboard Summary
# ══════════════════════════════════════════════════════════════════════════════
def plot_risk_dashboard(df_summary: pd.DataFrame, df_hist: pd.DataFrame):
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(BG)
    gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)

    # ── A. KPI cards (top row) ───────────────────────────────────────────────
    band_counts = df_summary["risk_level"].value_counts()
    kpi_items = [
        ("Provincias\nMonitoreadas", len(df_summary), ACCENT),
        ("Nivel Bajo",      band_counts.get("Bajo", 0),     RISK_COLORS["Bajo"]),
        ("Nivel Moderado",  band_counts.get("Moderado", 0), RISK_COLORS["Moderado"]),
        ("Nivel Alto +",    band_counts.get("Alto", 0) + band_counts.get("Epidemia", 0) + band_counts.get("Critico", 0),
                                                             RISK_COLORS["Alto"]),
    ]
    for col_idx, (label, value, color) in enumerate(kpi_items):
        ax_kpi = fig.add_subplot(gs[0, col_idx])
        ax_kpi.set_facecolor(PANEL)
        ax_kpi.set_xlim(0, 1); ax_kpi.set_ylim(0, 1)
        ax_kpi.axis("off")
        for sp in ax_kpi.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(1.5); sp.set_visible(True)
        ax_kpi.text(0.5, 0.60, str(value), ha="center", va="center",
                    fontsize=36, fontweight="bold", color=color)
        ax_kpi.text(0.5, 0.18, label, ha="center", va="center",
                    fontsize=10, color=SUBTEXT, multialignment="center")

    # ── B. Donut — risk level distribution ──────────────────────────────────
    ax_donut = fig.add_subplot(gs[1, 0])
    apply_dark_theme(fig, ax_donut)
    risk_order = ["Bajo", "Moderado", "Alto", "Epidemia", "Critico"]
    sizes  = [band_counts.get(r, 0) for r in risk_order]
    colors = [RISK_COLORS[r] for r in risk_order]
    non_zero = [(s, c, r) for s, c, r in zip(sizes, colors, risk_order) if s > 0]
    if non_zero:
        sz, co, lb = zip(*non_zero)
        wedges, _ = ax_donut.pie(sz, colors=co, startangle=90,
                                 wedgeprops={"width": 0.55, "edgecolor": BG, "linewidth": 1.5})
        ax_donut.legend(wedges, [f"{l} ({s})" for l, s in zip(lb, sz)],
                        loc="lower center", fontsize=8, labelcolor=TEXT,
                        framealpha=0.1, facecolor=PANEL, edgecolor=GRID,
                        bbox_to_anchor=(0.5, -0.25), ncol=2)
    ax_donut.set_title("Distribución por Nivel de Riesgo",
                       fontsize=10, color=TEXT, pad=8)

    # ── C. Top 10 provinces scatter — risk_index vs week_4_forecast ─────────
    ax_scatter = fig.add_subplot(gs[1, 1:3])
    apply_dark_theme(fig, ax_scatter)
    top10 = df_summary.nlargest(10, "risk_index")
    sc_colors = [RISK_COLORS.get(l, ACCENT) for l in top10["risk_level"]]
    ax_scatter.scatter(top10["risk_index"], top10["week_4_forecast"],
                       c=sc_colors, s=90, zorder=5, edgecolors=BG, linewidths=0.8)
    for _, row in top10.iterrows():
        ax_scatter.annotate(row["province"].split()[0],
                            (row["risk_index"], row["week_4_forecast"]),
                            xytext=(4, 3), textcoords="offset points",
                            fontsize=7.5, color=SUBTEXT)
    ax_scatter.plot([0, 100], [0, 100], color=GRID, linewidth=0.8, linestyle="--")
    ax_scatter.set_xlabel("Riesgo Semana 1", fontsize=9)
    ax_scatter.set_ylabel("Pronóstico Semana 4", fontsize=9)
    ax_scatter.set_title("Top 10 Provincias — Riesgo Actual vs Semana 4",
                         fontsize=10, color=TEXT)
    ax_scatter.set_xlim(0, 100); ax_scatter.set_ylim(0, 100)

    # ── D. RF vs MLP week-1 scatter ──────────────────────────────────────────
    ax_model = fig.add_subplot(gs[1, 3])
    apply_dark_theme(fig, ax_model)
    sc2_colors = [RISK_COLORS.get(l, ACCENT) for l in df_summary["risk_level"]]
    ax_model.scatter(df_summary["rf_week1"], df_summary["mlp_week1"],
                     c=sc2_colors, s=55, alpha=0.85, edgecolors=BG, linewidths=0.6)
    lo = min(df_summary["rf_week1"].min(), df_summary["mlp_week1"].min()) - 1
    hi = max(df_summary["rf_week1"].max(), df_summary["mlp_week1"].max()) + 1
    ax_model.plot([lo, hi], [lo, hi], color=GRID, linewidth=0.8, linestyle="--")
    ax_model.set_xlabel("RF Semana 1", fontsize=9)
    ax_model.set_ylabel("MLP Semana 1", fontsize=9)
    ax_model.set_title("Concordancia RF vs MLP", fontsize=10, color=TEXT)

    # ── E. National time series — cases (bottom row, full width) ────────────
    ax_ts = fig.add_subplot(gs[2, :])
    apply_dark_theme(fig, ax_ts)
    national = df_hist.groupby("date")[["cases", "outbreak_risk_index"]].mean().reset_index()
    national["date"] = pd.to_datetime(national["date"])
    national = national.sort_values("date")

    ax_ts.fill_between(national["date"], national["cases"],
                       alpha=0.25, color=ACCENT)
    ax_ts.plot(national["date"], national["cases"],
               color=ACCENT, linewidth=1.5, label="Casos (prom. nacional)")

    ax_risk = ax_ts.twinx()
    ax_risk.set_facecolor("none")
    ax_risk.plot(national["date"], national["outbreak_risk_index"],
                 color=RISK_COLORS["Alto"], linewidth=1.2, linestyle="--",
                 alpha=0.8, label="Índice de Riesgo")
    ax_risk.set_ylabel("Índice de Riesgo", fontsize=9, color=RISK_COLORS["Alto"])
    ax_risk.tick_params(colors=SUBTEXT, labelsize=8)
    ax_risk.spines["right"].set_edgecolor(GRID)

    # Epidemic threshold line
    ax_risk.axhline(65, color=RISK_COLORS["Epidemia"], linewidth=0.8,
                    linestyle=":", alpha=0.7, label="Umbral epidemia (65)")

    ax_ts.set_ylabel("Casos promedio (por provincia)", fontsize=9)
    ax_ts.set_xlabel("Fecha", fontsize=9)
    ax_ts.set_title("Serie Temporal Nacional — Casos de Dengue e Índice de Riesgo",
                    fontsize=11, color=TEXT, pad=8)

    # Combined legend
    lines1, labels1 = ax_ts.get_legend_handles_labels()
    lines2, labels2 = ax_risk.get_legend_handles_labels()
    ax_ts.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
                 framealpha=0.15, facecolor=PANEL, edgecolor=GRID,
                 labelcolor=TEXT, fontsize=8.5)

    # ── Title ────────────────────────────────────────────────────────────────
    fig.suptitle("Dashboard de Vigilancia Epidemiológica del Dengue\nRepública Dominicana — Abril 2026",
                 fontsize=16, fontweight="bold", color=TEXT, y=1.01)
    watermark(fig)

    path = OUT / "04_risk_dashboard.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    df_summary  = pd.read_csv(SUMMARY,  encoding="utf-8-sig")
    df_hist     = pd.read_csv(HIST,     encoding="utf-8-sig")
    df_forecast = pd.read_csv(FORECAST, encoding="utf-8-sig")
    print(f"  summary: {df_summary.shape}  hist: {df_hist.shape}  forecast: {df_forecast.shape}")

    print("\nGenerating visualizations...")
    plot_province_risk_bar(df_summary)
    plot_seasonal_trend(df_hist)
    plot_forecast_multiples(df_forecast, df_summary)
    plot_risk_dashboard(df_summary, df_hist)

    print(f"\nAll visuals saved to: {OUT}")


if __name__ == "__main__":
    main()
