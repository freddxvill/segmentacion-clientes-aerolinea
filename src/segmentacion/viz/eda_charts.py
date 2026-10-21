"""Gráficos Altair del EDA (páginas `01-datos.qmd` y `02-eda.qmd`).

Cada función recibe el dataset limpio (o el reporte de limpieza) y **devuelve** un gráfico.
Los datos se pre-agregan en pandas (histogramas, cuantiles, conteos) para no incrustar las
62k filas en el HTML; solo el scatter usa una muestra de filas.
"""

from typing import Any

import altair as alt
import numpy as np
import pandas as pd
from scipy import stats

from segmentacion.viz.theme import DIVERGING, GENDER_COLORS, MAIN, NEUTRAL, TIER_COLORS

DAYS_PER_MONTH = 365.25 / 12

LRFMC_LABELS: dict[str, str] = {
    "L": "L · Antigüedad (meses)",
    "R": "R · Recencia (meses)",
    "F": "F · Nº de vuelos",
    "M": "M · Km volados",
    "C": "C · Descuento medio",
}

# Etiquetas cortas para ejes con muchas variables (matriz de correlación).
LRFMC_SHORT: dict[str, str] = {
    "L": "L · Antigüedad",
    "R": "R · Recencia",
    "F": "F · Vuelos",
    "M": "M · Km",
    "C": "C · Descuento",
}

TIER_ORDER = [4, 5, 6]
TIER_LABELS = {4: "Nivel 4", 5: "Nivel 5", 6: "Nivel 6"}


# --- Datos auxiliares ---------------------------------------------------------------------


def lrfmc_preview(df: pd.DataFrame) -> pd.DataFrame:
    """Métricas LRFMC para explorar en el EDA.

    Vista previa con las fórmulas de plan-data.md §3.4; la versión oficial (con tests de
    fórmulas y artefacto `lrfmc.parquet`) es `features/lrfmc.py`, de la Fase 4.
    """
    return pd.DataFrame(
        {
            "L": (df["LOAD_TIME"] - df["FFP_DATE"]).dt.days / DAYS_PER_MONTH,
            "R": df["LAST_TO_END"] / 30,
            "F": df["FLIGHT_COUNT"],
            "M": df["SEG_KM_SUM"],
            "C": df["avg_discount"],
        },
        index=df.index,
    )


def revenue_balance(df: pd.DataFrame) -> pd.Series:
    """Balance de ingresos año 2 vs año 1, acotado a [-1, 1].

    -1 = solo ingresó en el año 1 (dejó de volar), +1 = solo en el año 2 (cliente nuevo o
    reactivado). A diferencia de `(Y2 - Y1) / (Y1 + 1)`, no explota cuando `SUM_YR_1 = 0`.
    """
    total = df["SUM_YR_1"] + df["SUM_YR_2"]
    return ((df["SUM_YR_2"] - df["SUM_YR_1"]) / total).where(total > 0)


def eda_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Cifras que citan los párrafos de insight (se calculan, no se escriben a mano)."""
    lrfmc = lrfmc_preview(df)
    revenue = df["SUM_YR_1"] + df["SUM_YR_2"]
    revenue_per_km = revenue / df["SEG_KM_SUM"]
    balance = revenue_balance(df)
    elite = df["FFP_TIER"] >= 5
    top_province = df["WORK_PROVINCE"].value_counts(normalize=True)
    monthly = df["FFP_DATE"].dt.to_period("M").value_counts()
    male = df["GENDER"] == "Male"
    return {
        "n_members": len(df),
        "skew": lrfmc.skew().round(2).to_dict(),
        "skew_log1p": np.log1p(lrfmc).skew().round(2).to_dict(),
        "skew_yeojohnson": {c: stats.skew(stats.yeojohnson(lrfmc[c])[0]) for c in lrfmc},
        "spearman_fm": lrfmc["F"].corr(lrfmc["M"], method="spearman"),
        "spearman_rf": lrfmc["R"].corr(lrfmc["F"], method="spearman"),
        "spearman_c_revenue_km": df["avg_discount"].corr(revenue_per_km, method="spearman"),
        "discount_median": df["avg_discount"].median(),
        "discount_above_1": (df["avg_discount"] > 1).mean(),
        "flights_median": df["FLIGHT_COUNT"].median(),
        "km_median": df["SEG_KM_SUM"].median(),
        "elite_share": elite.mean(),
        "elite_revenue_share": revenue[elite].sum() / revenue.sum(),
        "tier_flights_median": df.groupby("FFP_TIER")["FLIGHT_COUNT"].median().to_dict(),
        "male_share": male.mean(),
        "male_share_elite": male[elite].mean(),
        "male_share_tier4": male[~elite].mean(),
        "age_median_elite": df.loc[elite, "AGE"].median(),
        "age_median": df["AGE"].median(),
        "top_province": top_province.index[0],
        "top_province_share": top_province.iloc[0],
        "top3_province_share": top_province.iloc[:3].sum(),
        "country_cn_share": (df["WORK_COUNTRY"] == "CN").mean(),
        "enroll_peak_year": int(df["FFP_DATE"].dt.year.value_counts().idxmax()),
        "enroll_last_date": df["FFP_DATE"].max(),
        "enroll_min_month": monthly.idxmin().to_timestamp(),
        "enroll_min_count": int(monthly.min()),
        "recency_median": lrfmc["R"].median(),
        "recency_median_elite": lrfmc.loc[elite, "R"].median(),
        "km_per_flight_median": (df["SEG_KM_SUM"] / df["FLIGHT_COUNT"]).median(),
        "recency_over_12m": (df["LAST_TO_END"] > 365).mean(),
        "revenue_yr1_zero": (df["SUM_YR_1"] == 0).mean(),
        "revenue_yr2_zero": (df["SUM_YR_2"] == 0).mean(),
        "yr2_zero_inactive": (df.loc[df["SUM_YR_2"] == 0, "LAST_TO_END"] > 365).mean(),
        "balance_lost": (balance == -1).mean(),
        "balance_new": (balance == 1).mean(),
        "balance_declining": (balance < 0).mean(),
    }


def _histogram(values: pd.Series, edges: np.ndarray) -> pd.DataFrame:
    counts, edges = np.histogram(values.dropna(), bins=edges)
    return pd.DataFrame({"inicio": edges[:-1], "fin": edges[1:], "socios": counts})


def _es_int(n: int) -> str:
    """Entero con separador de miles español (para textos fijos fuera de Vega)."""
    return f"{n:,}".replace(",", ".")


def _pct(part: pd.Series, total: float) -> pd.Series:
    return part / total if total else part * 0.0


# --- 01-datos: calidad y limpieza -----------------------------------------------------------


def null_values_chart(report: dict) -> alt.Chart:
    """% de nulos por columna en el CSV crudo y en el dataset limpio."""
    counts = report["null_counts"]
    rows = []
    for stage, label, n in (
        ("raw", "Crudo", report["input_rows"]),
        ("clean", "Limpio", report["output_rows"]),
    ):
        for column, nulls in counts[stage].items():
            rows.append({"columna": column, "etapa": label, "nulos": nulls, "pct": nulls / n})
    data = pd.DataFrame(rows)
    with_nulls = data.groupby("columna")["nulos"].transform("max") > 0
    data = data[with_nulls]
    order = data[data["etapa"] == "Limpio"].sort_values("pct", ascending=False)["columna"]
    return (
        alt.Chart(data, title="Valores nulos por columna, antes y después de limpiar")
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("pct:Q", title="% de socios con el valor vacío", axis=alt.Axis(format="%")),
            y=alt.Y("columna:N", title=None, sort=list(order), scale=alt.Scale(paddingInner=0.3)),
            yOffset=alt.YOffset("etapa:N", sort=["Crudo", "Limpio"]),
            color=alt.Color(
                "etapa:N",
                title="Etapa",
                scale=alt.Scale(domain=["Crudo", "Limpio"], range=[NEUTRAL, MAIN]),
            ),
            tooltip=[
                alt.Tooltip("columna:N", title="Columna"),
                alt.Tooltip("etapa:N", title="Etapa"),
                alt.Tooltip("nulos:Q", title="Filas vacías", format=",.0f"),
                alt.Tooltip("pct:Q", title="% de filas", format=".2%"),
            ],
        )
        .properties(width="container", height=alt.Step(32))
    )


def cleaning_funnel_chart(report: dict) -> alt.Chart:
    """Embudo de limpieza: socios que quedan tras cada regla que elimina filas."""
    remaining = report["input_rows"]
    rows = [{"paso": "CSV crudo", "socios": remaining, "eliminadas": 0, "orden": 0}]
    for step in report["steps"]:
        if step["action"] == "drop" and step["rows"]:
            remaining -= step["rows"]
            rows.append(
                {
                    "paso": step["description"],
                    "socios": remaining,
                    "eliminadas": step["rows"],
                    "orden": len(rows),
                }
            )
    data = pd.DataFrame(rows)
    data["retenido"] = data["socios"] / report["input_rows"]
    base = alt.Chart(data, title="Embudo de limpieza").encode(
        y=alt.Y(
            "paso:N", title=None, sort=alt.EncodingSortField("orden"), axis=alt.Axis(labelLimit=320)
        ),
        tooltip=[
            alt.Tooltip("paso:N", title="Paso"),
            alt.Tooltip("socios:Q", title="Socios restantes", format=",.0f"),
            alt.Tooltip("eliminadas:Q", title="Filas eliminadas", format=",.0f"),
            alt.Tooltip("retenido:Q", title="% retenido", format=".1%"),
        ],
    )
    bars = base.mark_bar(cornerRadiusEnd=3, height=22).encode(
        x=alt.X("socios:Q", title="Socios", axis=alt.Axis(format=",.0f")),
        color=alt.value(MAIN),
    )
    labels = base.mark_text(align="right", dx=-6, color="white", fontWeight=600).encode(
        x="socios:Q", text=alt.Text("socios:Q", format=",.0f")
    )
    return (bars + labels).properties(width="container", height=alt.Step(36))


def nulled_values_chart(report: dict) -> alt.Chart:
    """Reglas que anulan valores sin eliminar al socio."""
    data = pd.DataFrame([s for s in report["steps"] if s["action"] == "set_null" and s["rows"]])
    return (
        alt.Chart(data, title="Valores anulados (el socio se conserva)")
        .mark_bar(cornerRadiusEnd=3, height=16)
        .encode(
            x=alt.X("rows:Q", title="Valores anulados", axis=alt.Axis(format=",.0f")),
            y=alt.Y("description:N", title=None, sort="-x", axis=alt.Axis(labelLimit=360)),
            color=alt.value(NEUTRAL),
            tooltip=[
                alt.Tooltip("description:N", title="Regla"),
                alt.Tooltip("rows:Q", title="Valores", format=",.0f"),
            ],
        )
        .properties(width="container", height=alt.Step(28))
    )


# --- 02-eda ---------------------------------------------------------------------------------


def distribution_chart(df: pd.DataFrame, bins: int = 40) -> alt.Chart:
    """Histograma de cada métrica LRFMC con selector de variable y de escala (lineal/log)."""
    lrfmc = lrfmc_preview(df)
    frames = []
    for column, label in LRFMC_LABELS.items():
        values = lrfmc[column]
        low, high = values.min(), values.max()
        linear = _histogram(values, np.linspace(low, high, bins + 1)).assign(escala="Lineal")
        linear["x0"], linear["x1"] = linear["inicio"], linear["fin"]
        log_low = max(low, values[values > 0].min())
        log_edges = np.geomspace(log_low, high, bins + 1)
        log_edges[0] = low if low > 0 else log_low
        log = _histogram(values.clip(lower=log_low), log_edges).assign(escala="Logarítmica")
        log["x0"], log["x1"] = np.log10(log["inicio"]), np.log10(log["fin"])
        hist = pd.concat([linear, log]).assign(variable=label)
        hist["pct"] = hist["socios"] / len(values)
        frames.append(hist)
    data = pd.concat(frames, ignore_index=True)

    variable = alt.param(
        name="variable",
        value=LRFMC_LABELS["F"],
        bind=alt.binding_select(options=list(LRFMC_LABELS.values()), name="Variable: "),
    )
    scale = alt.param(
        name="escala",
        value="Lineal",
        bind=alt.binding_radio(options=["Lineal", "Logarítmica"], name="Escala del eje X: "),
    )
    # En escala log se dibuja log10(valor) y el eje muestra el valor original.
    label_expr = (
        "escala == 'Lineal' ? format(datum.value, ',.3~r') : format(pow(10, datum.value), ',.3~r')"
    )
    return (
        alt.Chart(data, title="Distribución de las métricas LRFMC")
        .mark_bar(cornerRadiusEnd=2)
        .encode(
            x=alt.X("x0:Q", title="Valor", axis=alt.Axis(labelExpr=label_expr, tickCount=6)),
            x2="x1:Q",
            y=alt.Y("socios:Q", title="Socios", axis=alt.Axis(format=",.0f")),
            y2=alt.datum(0),
            color=alt.value(MAIN),
            tooltip=[
                alt.Tooltip("variable:N", title="Variable"),
                alt.Tooltip("inicio:Q", title="Desde", format=",.2f"),
                alt.Tooltip("fin:Q", title="Hasta", format=",.2f"),
                alt.Tooltip("socios:Q", title="Socios", format=",.0f"),
                alt.Tooltip("pct:Q", title="% de socios", format=".1%"),
            ],
        )
        .add_params(variable, scale)
        .transform_filter("datum.variable == variable && datum.escala == escala")
        .properties(width="container", height=300)
    )


def skewness_chart(df: pd.DataFrame, bins: int = 30) -> alt.FacetChart:
    """Forma de cada métrica estandarizada según la transformación elegida (+ asimetría)."""
    lrfmc = lrfmc_preview(df)
    transforms = {
        "Sin transformar": lambda v: v,
        "log1p": np.log1p,
        "Yeo-Johnson": lambda v: stats.yeojohnson(v)[0],
    }
    hists, skews = [], []
    for name, func in transforms.items():
        for column in LRFMC_LABELS:
            values = np.asarray(func(lrfmc[column].to_numpy(dtype=float)))
            z = (values - values.mean()) / values.std()
            hist = _histogram(pd.Series(z), np.linspace(z.min(), z.max(), bins + 1))
            hists.append(hist.assign(transformacion=name, variable=column))
            skews.append({"transformacion": name, "variable": column, "asimetria": stats.skew(z)})
    hist_data = pd.DataFrame(pd.concat(hists, ignore_index=True))
    skew_data = pd.DataFrame(skews)
    skew_data["etiqueta"] = skew_data["asimetria"].map(
        lambda s: f"asimetría {s:+.2f}".replace(".", ",")
    )
    data = hist_data.merge(skew_data, on=["transformacion", "variable"])
    data["variable"] = data["variable"].map(LRFMC_LABELS)

    transform = alt.param(
        name="transformacion",
        value="Sin transformar",
        bind=alt.binding_radio(options=list(transforms), name="Transformación: "),
    )
    base = alt.Chart().transform_filter("datum.transformacion == transformacion")
    bars = base.mark_bar(cornerRadiusEnd=1, color=MAIN).encode(
        x=alt.X("inicio:Q", title="Valor estandarizado (z)"),
        x2="fin:Q",
        y=alt.Y("socios:Q", title="Socios", axis=alt.Axis(format=",.0f", tickCount=4)),
        y2=alt.datum(0),
        tooltip=[
            alt.Tooltip("variable:N", title="Variable"),
            alt.Tooltip("inicio:Q", title="z desde", format=".2f"),
            alt.Tooltip("fin:Q", title="z hasta", format=".2f"),
            alt.Tooltip("socios:Q", title="Socios", format=",.0f"),
            alt.Tooltip("asimetria:Q", title="Asimetría", format="+.2f"),
        ],
    )
    label = (
        base.mark_text(align="right", baseline="top", x="width", dx=-2, y=2)
        .encode(text="etiqueta:N")
        .transform_aggregate(etiqueta="min(etiqueta)", groupby=["variable", "transformacion"])
    )
    return (
        alt.layer(bars, label, data=data)
        .properties(width=175, height=140)
        .facet(
            facet=alt.Facet("variable:N", title=None, sort=list(LRFMC_LABELS.values())),
            columns=3,
            title="Asimetría de cada métrica según la transformación",
        )
        .add_params(transform)
        .resolve_scale(x="independent", y="independent")
    )


def correlation_heatmap(df: pd.DataFrame) -> alt.Chart:
    """Matriz de correlación de Spearman (triángulo inferior) de LRFMC y variables de perfil."""
    lrfmc = lrfmc_preview(df)
    data = pd.DataFrame(
        {
            **{label: lrfmc[c] for c, label in LRFMC_SHORT.items()},
            "Ingreso total": df["SUM_YR_1"] + df["SUM_YR_2"],
            "Intervalo medio": df["AVG_INTERVAL"],
            "Puntos": df["Points_Sum"],
            "Canjes": df["EXCHANGE_COUNT"],
            "Edad": df["AGE"],
        }
    )
    columns = list(data.columns)
    corr = data.corr(method="spearman")
    rows = [
        {"var_x": a, "var_y": b, "rho": corr.loc[a, b], "ix": i, "iy": j}
        for i, a in enumerate(columns)
        for j, b in enumerate(columns)
        if j > i
    ]
    matrix = pd.DataFrame(rows)
    base = alt.Chart(matrix, title="Correlación de Spearman").encode(
        x=alt.X("var_x:N", title=None, sort=columns, axis=alt.Axis(labelAngle=-40)),
        y=alt.Y("var_y:N", title=None, sort=columns),
        tooltip=[
            alt.Tooltip("var_x:N", title="Variable 1"),
            alt.Tooltip("var_y:N", title="Variable 2"),
            alt.Tooltip("rho:Q", title="ρ de Spearman", format="+.2f"),
        ],
    )
    cells = base.mark_rect(cornerRadius=2, stroke="transparent", strokeWidth=2).encode(
        color=alt.Color(
            "rho:Q",
            title="ρ",
            scale=alt.Scale(domain=[-1, 0, 1], range=list(DIVERGING), interpolate="lab"),
            legend=alt.Legend(format="+.1f", gradientLength=180),
        )
    )
    text = base.mark_text(fontSize=11).encode(
        text=alt.Text("rho:Q", format="+.2f"),
        color=alt.condition("abs(datum.rho) > 0.5", alt.value("white"), alt.value("#1f1f1d")),
    )
    return (cells + text).properties(width="container", height=alt.Step(34))


def frequency_km_scatter(df: pd.DataFrame, n: int = 4000, seed: int = 42) -> alt.HConcatChart:
    """Vuelos vs km (muestra) con brush enlazado al histograma del descuento medio."""
    sample = df.sample(n=min(n, len(df)), random_state=seed)
    data = pd.DataFrame(
        {
            "vuelos": sample["FLIGHT_COUNT"],
            "km": sample["SEG_KM_SUM"],
            "descuento": sample["avg_discount"],
            "nivel": sample["FFP_TIER"].map(TIER_LABELS),
        }
    )
    brush = alt.selection_interval(name="seleccion", encodings=["x", "y"])
    scatter = (
        alt.Chart(
            data,
            title=alt.Title(
                "Vuelos vs km volados",
                subtitle=f"Muestra aleatoria de {_es_int(len(data))} socios"
                " · arrastra para seleccionar",
            ),
        )
        .mark_circle(size=14)
        .encode(
            x=alt.X(
                "vuelos:Q",
                title="Nº de vuelos (log)",
                scale=alt.Scale(type="log", nice=False),
                axis=alt.Axis(values=[2, 5, 10, 20, 50, 100, 200], grid=True),
            ),
            y=alt.Y(
                "km:Q",
                title="Km volados (log)",
                scale=alt.Scale(type="log", nice=False),
                axis=alt.Axis(
                    values=[
                        500,
                        1_000,
                        2_000,
                        5_000,
                        10_000,
                        20_000,
                        50_000,
                        100_000,
                        200_000,
                        500_000,
                    ],
                    format=",.0f",
                ),
            ),
            color=alt.condition(brush, alt.value(MAIN), alt.value(NEUTRAL)),
            opacity=alt.condition(brush, alt.value(0.45), alt.value(0.12)),
            tooltip=[
                alt.Tooltip("vuelos:Q", title="Vuelos"),
                alt.Tooltip("km:Q", title="Km", format=",.0f"),
                alt.Tooltip("descuento:Q", title="Descuento medio", format=".2f"),
                alt.Tooltip("nivel:N", title="Nivel"),
            ],
        )
        .add_params(brush)
        .properties(width=360, height=320)
    )
    hist_base = alt.Chart(data, title="Descuento medio (C)").encode(
        x=alt.X("descuento:Q", bin=alt.Bin(maxbins=30), title="Descuento medio"),
        y=alt.Y("count():Q", title="Socios"),
    )
    background = hist_base.mark_bar(color=NEUTRAL, opacity=0.35)
    selected = (
        hist_base.mark_bar(color=MAIN)
        .transform_filter(brush)
        .encode(tooltip=[alt.Tooltip("count():Q", title="Socios seleccionados")])
    )
    return alt.hconcat(scatter, (background + selected).properties(width=240, height=320))


def _box_stats(df: pd.DataFrame) -> pd.DataFrame:
    lrfmc = lrfmc_preview(df).assign(nivel=df["FFP_TIER"])
    rows = []
    for tier, group in lrfmc.groupby("nivel"):
        for column, label in LRFMC_LABELS.items():
            q = group[column].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            rows.append(
                {
                    "nivel": TIER_LABELS[tier],
                    "variable": label,
                    "p5": q[0.05],
                    "q1": q[0.25],
                    "mediana": q[0.5],
                    "q3": q[0.75],
                    "p95": q[0.95],
                    "socios": len(group),
                }
            )
    return pd.DataFrame(rows)


def tier_boxplot_chart(df: pd.DataFrame) -> alt.FacetChart:
    """Cajas (p5–p95, cuartiles y mediana) de cada métrica por nivel del programa."""
    data = _box_stats(df)
    tiers = [TIER_LABELS[t] for t in TIER_ORDER]
    legend = alt.selection_point(name="nivel_sel", fields=["nivel"], bind="legend")
    color = alt.Color(
        "nivel:N",
        title="Nivel (clic para resaltar)",
        scale=alt.Scale(domain=tiers, range=[TIER_COLORS[t] for t in TIER_ORDER]),
    )
    opacity = alt.condition(legend, alt.value(1.0), alt.value(0.15))
    tooltip = [
        alt.Tooltip("nivel:N", title="Nivel"),
        alt.Tooltip("socios:Q", title="Socios", format=",.0f"),
        alt.Tooltip("p5:Q", title="P5", format=",.2f"),
        alt.Tooltip("q1:Q", title="Q1", format=",.2f"),
        alt.Tooltip("mediana:Q", title="Mediana", format=",.2f"),
        alt.Tooltip("q3:Q", title="Q3", format=",.2f"),
        alt.Tooltip("p95:Q", title="P95", format=",.2f"),
    ]
    base = alt.Chart().encode(x=alt.X("nivel:N", title=None, sort=tiers, axis=None))
    whisker = base.mark_rule(strokeWidth=1.5).encode(
        y=alt.Y("p5:Q", title=None, axis=alt.Axis(format=",~f")),
        y2="p95:Q",
        color=color,
        opacity=opacity,
        tooltip=tooltip,
    )
    box = base.mark_bar(size=26, cornerRadius=3).encode(
        y="q1:Q", y2="q3:Q", color=color, opacity=opacity, tooltip=tooltip
    )
    median = base.mark_tick(size=26, thickness=2, color="white").encode(
        y="mediana:Q", tooltip=tooltip
    )
    return (
        alt.layer(whisker, box.add_params(legend), median, data=data)
        .properties(width=150, height=200)
        .facet(
            facet=alt.Facet("variable:N", title=None, sort=list(LRFMC_LABELS.values())),
            columns=3,
            title=alt.Title(
                "Métricas LRFMC por nivel del programa",
                subtitle="Caja = cuartiles · línea blanca = mediana · bigotes = P5–P95",
            ),
        )
        .resolve_scale(y="independent")
    )


def age_gender_chart(df: pd.DataFrame, bin_width: int = 5) -> alt.Chart:
    """Pirámide de edad por género con filtro por nivel del programa."""
    data = df.dropna(subset=["AGE", "GENDER"]).assign(
        genero=lambda d: d["GENDER"].map({"Male": "Hombre", "Female": "Mujer"}),
        edad=lambda d: (d["AGE"] // bin_width * bin_width).astype(int),
    )
    groups = {"Todos": data, **{str(t): data[data["FFP_TIER"] == t] for t in TIER_ORDER}}
    frames = []
    for name, group in groups.items():
        counts = group.groupby(["genero", "edad"]).size().rename("socios").reset_index()
        counts["pct"] = _pct(counts["socios"], len(group))
        counts["nivel"] = name
        frames.append(counts)
    pyramid = pd.concat(frames, ignore_index=True)
    pyramid["rango"] = pyramid["edad"].map(lambda a: f"{a}–{a + bin_width - 1}")
    pyramid["x"] = np.where(pyramid["genero"] == "Hombre", -pyramid["pct"], pyramid["pct"])

    tier = alt.param(
        name="nivel",
        value="Todos",
        bind=alt.binding_radio(
            options=list(groups),
            labels=["Todos", *[f"Nivel {t}" for t in TIER_ORDER]],
            name="Nivel del programa: ",
        ),
    )
    ages = sorted(pyramid["edad"].unique(), reverse=True)
    return (
        alt.Chart(
            pyramid,
            title=alt.Title(
                "Edad y género de los socios", subtitle="% de los socios del nivel elegido"
            ),
        )
        .mark_bar(cornerRadius=2, height=14)
        .encode(
            x=alt.X(
                "x:Q",
                title="← Hombres · Mujeres →",
                axis=alt.Axis(labelExpr="format(abs(datum.value), '.0%')"),
            ),
            y=alt.Y("rango:N", title="Edad", sort=[f"{a}–{a + bin_width - 1}" for a in ages]),
            color=alt.Color(
                "genero:N",
                title="Género",
                scale=alt.Scale(domain=list(GENDER_COLORS), range=list(GENDER_COLORS.values())),
            ),
            tooltip=[
                alt.Tooltip("genero:N", title="Género"),
                alt.Tooltip("rango:N", title="Edad"),
                alt.Tooltip("socios:Q", title="Socios", format=",.0f"),
                alt.Tooltip("pct:Q", title="% del nivel", format=".1%"),
            ],
        )
        .add_params(tier)
        .transform_filter("datum.nivel == nivel")
        .properties(width="container", height=alt.Step(18))
    )


def top_locations_chart(df: pd.DataFrame, n: int = 15) -> alt.Chart:
    """Top-n provincias o ciudades de trabajo, ordenables por socios o por nombre."""
    revenue = df["SUM_YR_1"] + df["SUM_YR_2"]
    frames = []
    for column, dimension in (("WORK_PROVINCE", "Provincia"), ("WORK_CITY", "Ciudad")):
        grouped = (
            df.assign(elite=df["FFP_TIER"] >= 5, ingreso=revenue)
            .groupby(column)
            .agg(
                socios=("MEMBER_NO", "size"), elite=("elite", "mean"), ingreso=("ingreso", "median")
            )
            .nlargest(n, "socios")
            .reset_index(names="nombre")
        )
        grouped["nombre"] = grouped["nombre"].str.title()
        grouped["pct"] = grouped["socios"] / len(df)
        grouped["rank_socios"] = grouped["socios"].rank(ascending=False, method="first")
        grouped["rank_nombre"] = grouped["nombre"].astype(object).rank(method="first")
        frames.append(grouped.assign(dimension=dimension))
    data = pd.concat(frames, ignore_index=True)

    dimension = alt.param(
        name="dimension",
        value="Provincia",
        bind=alt.binding_radio(options=["Provincia", "Ciudad"], name="Ver por: "),
    )
    order = alt.param(
        name="orden",
        value="Socios",
        bind=alt.binding_radio(options=["Socios", "Nombre"], name="Ordenar por: "),
    )
    base = (
        alt.Chart(data, title=f"Top {n} ubicaciones de trabajo de los socios")
        .transform_filter("datum.dimension == dimension")
        .transform_calculate(clave="orden == 'Socios' ? datum.rank_socios : datum.rank_nombre")
        .encode(
            y=alt.Y("nombre:N", title=None, sort=alt.EncodingSortField("clave", order="ascending")),
            tooltip=[
                alt.Tooltip("nombre:N", title="Ubicación"),
                alt.Tooltip("socios:Q", title="Socios", format=",.0f"),
                alt.Tooltip("pct:Q", title="% del total", format=".1%"),
                alt.Tooltip("elite:Q", title="% en niveles 5–6", format=".1%"),
                alt.Tooltip("ingreso:Q", title="Ingreso mediano (2 años)", format=",.0f"),
            ],
        )
    )
    bars = base.mark_bar(cornerRadiusEnd=3, height=14, color=MAIN).encode(
        x=alt.X("socios:Q", title="Socios", axis=alt.Axis(format=",.0f"))
    )
    labels = base.mark_text(align="left", dx=4).encode(
        x="socios:Q", text=alt.Text("socios:Q", format=",.0f")
    )
    return (
        (bars + labels)
        .add_params(dimension, order)
        .properties(width="container", height=alt.Step(20))
    )


def enrollment_chart(df: pd.DataFrame) -> alt.Chart:
    """Altas mensuales en el programa, con zoom y desplazamiento en el eje temporal."""
    monthly = (
        df["FFP_DATE"].dt.to_period("M").value_counts().sort_index().rename("altas").reset_index()
    )
    monthly["mes"] = monthly["FFP_DATE"].dt.to_timestamp().dt.strftime("%Y-%m-%d")
    monthly = monthly[["mes", "altas"]]
    zoom = alt.selection_interval(bind="scales", encodings=["x"])
    hover = alt.selection_point(
        name="hover", fields=["mes"], nearest=True, on="pointerover", empty=False
    )
    base = alt.Chart(
        monthly,
        title=alt.Title(
            "Altas mensuales en el programa",
            subtitle="Rueda del ratón para hacer zoom · arrastra para desplazarte",
        ),
    ).encode(x=alt.X("mes:T", title=None, axis=alt.Axis(format="%Y", tickCount="year")))
    area = base.mark_area(opacity=0.18, color=MAIN, line={"color": MAIN, "strokeWidth": 2})
    area = area.encode(y=alt.Y("altas:Q", title="Altas por mes", axis=alt.Axis(format=",.0f")))
    points = (
        base.mark_point(size=80, filled=True, color=MAIN)
        .encode(
            y="altas:Q",
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=[
                alt.Tooltip("mes:T", title="Mes", format="%B %Y"),
                alt.Tooltip("altas:Q", title="Altas", format=",.0f"),
            ],
        )
        .add_params(hover)
    )
    rule = base.mark_rule(color=NEUTRAL).encode(
        opacity=alt.condition(hover, alt.value(0.6), alt.value(0))
    )
    return (area + rule + points).add_params(zoom).properties(width="container", height=280)


def revenue_trend_chart(df: pd.DataFrame, bins: int = 40) -> alt.LayerChart:
    """Balance de ingresos año 2 vs año 1 con selección de rango y conteo de socios."""
    data = pd.DataFrame(
        {"balance": revenue_balance(df), "recencia": df["LAST_TO_END"] / 30}
    ).dropna()
    edges = np.linspace(-1, 1, bins + 1)
    data["bin"] = pd.cut(data["balance"], edges, include_lowest=True, labels=False)
    hist = (
        data.groupby("bin")
        .agg(socios=("balance", "size"), recencia=("recencia", "median"))
        .reindex(range(bins), fill_value=0)
        .reset_index()
    )
    hist["inicio"], hist["fin"] = edges[:-1], edges[1:]
    hist["pct"] = hist["socios"] / len(data)

    brush = alt.selection_interval(name="rango", encodings=["x"])
    base = alt.Chart(
        hist,
        title=alt.Title(
            "Balance de ingresos: año 2 vs año 1",
            subtitle="(Y2 − Y1) / (Y1 + Y2) · −1 = dejó de volar · +1 = solo voló el año 2"
            " · arrastra para seleccionar un rango",
        ),
    )
    bars = (
        base.mark_bar(cornerRadiusEnd=2)
        .encode(
            x=alt.X("inicio:Q", title="Balance de ingresos", scale=alt.Scale(domain=[-1, 1])),
            x2="fin:Q",
            y=alt.Y("socios:Q", title="Socios", axis=alt.Axis(format=",.0f")),
            y2=alt.datum(0),
            color=alt.condition(brush, alt.value(MAIN), alt.value(NEUTRAL)),
            tooltip=[
                alt.Tooltip("inicio:Q", title="Desde", format="+.2f"),
                alt.Tooltip("fin:Q", title="Hasta", format="+.2f"),
                alt.Tooltip("socios:Q", title="Socios", format=",.0f"),
                alt.Tooltip("pct:Q", title="% de socios", format=".1%"),
                alt.Tooltip("recencia:Q", title="Recencia mediana (meses)", format=".1f"),
            ],
        )
        .add_params(brush)
    )
    # Resumen de la selección (sin selección cuenta a todos los socios).
    summary = (
        base.transform_filter(brush)
        .transform_aggregate(socios="sum(socios)", pct="sum(pct)")
        .transform_calculate(
            texto="'Selección: ' + format(datum.socios, ',.0f') + ' socios ('"
            " + format(datum.pct, '.1%') + ')'"
        )
        .mark_text(align="left", baseline="top", fontSize=13, fontWeight=600)
        .encode(text="texto:N", x=alt.value(6), y=alt.value(4))
    )
    return (bars + summary).properties(width="container", height=280)


def lineage_table(report: dict) -> pd.DataFrame:
    """Reglas de limpieza con efecto, listas para mostrar como tabla en el sitio."""
    steps = pd.DataFrame([s for s in report["steps"] if s["rows"]])
    return pd.DataFrame(
        {
            "Regla": steps["description"],
            "Acción": steps["action"].map(
                {"drop": "Elimina el socio", "set_null": "Anula el valor"}
            ),
            "Filas": steps["rows"],
            "% del crudo": steps["rows"] / report["input_rows"],
        }
    )
