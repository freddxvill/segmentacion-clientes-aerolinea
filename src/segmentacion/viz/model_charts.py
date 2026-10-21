"""Gráficos Altair del preprocesamiento y del modelado (`03-lrfmc.qmd` en adelante).

Igual que en `eda_charts`, cada función recibe datos ya calculados (tablas de comparación,
features escaladas) y devuelve un gráfico con datos pre-agregados.
"""

import altair as alt
import pandas as pd

from segmentacion.features.lrfmc import LRFMC_COLUMNS
from segmentacion.viz.eda_charts import LRFMC_LABELS
from segmentacion.viz.theme import INK, MAIN, NEUTRAL, SEQUENTIAL

TRANSFORM_LABELS = {"none": "Sin transformar", "log1p": "log1p", "yeo-johnson": "Yeo-Johnson"}
SCALER_LABELS = {"standard": "StandardScaler", "robust": "RobustScaler"}


def option_label(transform: str, scaler: str) -> str:
    return f"{TRANSFORM_LABELS[transform]} + {SCALER_LABELS[scaler]}"


def transform_comparison_chart(comparison: pd.DataFrame, chosen: tuple[str, str]) -> alt.LayerChart:
    """Silhouette medio (barra) y por *k* (marcas) de cada combinación transformación × escalado."""
    ks = [int(c.removeprefix("silhouette_k")) for c in comparison if c.startswith("silhouette_k")]
    pairs = list(zip(comparison["transform"], comparison["scaler"], strict=True))
    data = comparison.assign(
        opcion=[option_label(t, s) for t, s in pairs],
        elegida=[pair == chosen for pair in pairs],
    )
    order = list(data.sort_values("silhouette_mean", ascending=False)["opcion"])
    per_k = data.melt(
        id_vars=["opcion"],
        value_vars=[f"silhouette_k{k}" for k in ks],
        var_name="k",
        value_name="silhouette",
    )
    per_k["k"] = "k = " + per_k["k"].str.removeprefix("silhouette_k")
    y = alt.Y("opcion:N", title=None, sort=order, axis=alt.Axis(labelLimit=260))
    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=3, height=18)
        .encode(
            x=alt.X("silhouette_mean:Q", title="Silhouette (media de los k comparados)"),
            y=y,
            color=alt.condition("datum.elegida", alt.value(MAIN), alt.value(NEUTRAL)),
            tooltip=[
                alt.Tooltip("opcion:N", title="Opción"),
                alt.Tooltip("silhouette_mean:Q", title="Silhouette medio", format=".3f"),
                *[alt.Tooltip(f"silhouette_k{k}:Q", title=f"k = {k}", format=".3f") for k in ks],
                alt.Tooltip("max_abs_skew:Q", title="|asimetría| máxima", format=".2f"),
                alt.Tooltip("n_abs_skew_lt_1:Q", title="Features con |asimetría| < 1"),
            ],
        )
    )
    ticks = (
        alt.Chart(per_k)
        .mark_tick(thickness=2, size=14, color=INK)
        .encode(
            x="silhouette:Q",
            y=y,
            tooltip=[
                alt.Tooltip("opcion:N", title="Opción"),
                alt.Tooltip("k:N", title="k"),
                alt.Tooltip("silhouette:Q", title="Silhouette", format=".3f"),
            ],
        )
    )
    return (bars + ticks).properties(
        width="container",
        height=alt.Step(34),
        title=alt.Title(
            "Silhouette de K-Means según el preprocesamiento",
            subtitle=f"Barra = media · marcas = cada k ({', '.join(map(str, ks))}) · "
            "azul = opción elegida",
        ),
    )


def skew_comparison_chart(comparison: pd.DataFrame) -> alt.LayerChart:
    """|Asimetría| de cada feature tras cada transformación (el escalado no la cambia)."""
    rows = comparison[comparison["scaler"] == "standard"]
    data = rows.melt(
        id_vars=["transform"],
        value_vars=[f"skew_{c}" for c in LRFMC_COLUMNS],
        var_name="feature",
        value_name="asimetria",
    )
    data["feature"] = data["feature"].str.removeprefix("skew_").map(LRFMC_LABELS)
    data["transformacion"] = data["transform"].map(TRANSFORM_LABELS)
    data["abs"] = data["asimetria"].abs()
    base = alt.Chart(data).encode(
        x=alt.X(
            "feature:N",
            title=None,
            sort=list(LRFMC_LABELS.values()),
            axis=alt.Axis(labelAngle=0, labelLimit=140),
        ),
        y=alt.Y("transformacion:N", title=None, sort=list(TRANSFORM_LABELS.values())),
        tooltip=[
            alt.Tooltip("transformacion:N", title="Transformación"),
            alt.Tooltip("feature:N", title="Feature"),
            alt.Tooltip("asimetria:Q", title="Asimetría", format="+.2f"),
        ],
    )
    cells = base.mark_rect(cornerRadius=3).encode(
        color=alt.Color(
            "abs:Q",
            title="|asimetría|",
            scale=alt.Scale(range=list(SEQUENTIAL[:4]), domain=[0, 4]),
        )
    )
    text = base.mark_text(fontSize=12, fontWeight=600).encode(
        text=alt.Text("asimetria:Q", format="+.2f"),
        color=alt.condition("datum.abs > 2", alt.value("white"), alt.value("#1f1f1d")),
    )
    return (cells + text).properties(
        width="container",
        height=alt.Step(44),
        title=alt.Title(
            "Asimetría de cada feature según la transformación",
            subtitle="Objetivo: |asimetría| < 1 (idealmente cerca de 0)",
        ),
    )


def scaled_distribution_chart(scaled: pd.DataFrame) -> alt.LayerChart:
    """Cajas (P1–P99, cuartiles, mediana) de cada feature ya transformada y escalada."""
    rows = []
    for column in LRFMC_COLUMNS:
        q = scaled[column].quantile([0.01, 0.25, 0.5, 0.75, 0.99])
        rows.append(
            {
                "feature": LRFMC_LABELS[column],
                "p1": q[0.01],
                "q1": q[0.25],
                "mediana": q[0.5],
                "q3": q[0.75],
                "p99": q[0.99],
                "media": scaled[column].mean(),
                "desviacion": scaled[column].std(ddof=0),
            }
        )
    data = pd.DataFrame(rows)
    tooltip = [
        alt.Tooltip("feature:N", title="Feature"),
        alt.Tooltip("media:Q", title="Media", format="+.3f"),
        alt.Tooltip("desviacion:Q", title="Desviación", format=".3f"),
        alt.Tooltip("p1:Q", title="P1", format="+.2f"),
        alt.Tooltip("mediana:Q", title="Mediana", format="+.2f"),
        alt.Tooltip("p99:Q", title="P99", format="+.2f"),
    ]
    y = alt.Y("feature:N", title=None, sort=list(LRFMC_LABELS.values()))
    base = alt.Chart(data).encode(y=y, tooltip=tooltip)
    whisker = base.mark_rule(color=MAIN, strokeWidth=1.5).encode(
        x=alt.X("p1:Q", title="Valor escalado"), x2="p99:Q"
    )
    box = base.mark_bar(color=MAIN, height=18, cornerRadius=3).encode(x="q1:Q", x2="q3:Q")
    median = base.mark_tick(color="white", thickness=2, size=18).encode(x="mediana:Q")
    zero = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(color=NEUTRAL, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    return (zero + whisker + box + median).properties(
        width="container",
        height=alt.Step(36),
        title=alt.Title(
            "Features listas para K-Means",
            subtitle="Caja = cuartiles · línea blanca = mediana · bigotes = P1–P99",
        ),
    )
