"""Gráficos Altair del perfilado de segmentos (Fase 9, `05-segmentos.qmd`).

Todos reciben `segment_profiles.csv` (una fila por segmento, ya ordenada por valor) y usan el
color fijo de cada segmento (`theme.segment_colors`).
"""

import altair as alt
import pandas as pd

from segmentacion.segments.profile import TREND
from segmentacion.viz.eda_charts import LRFMC_LABELS
from segmentacion.viz.theme import DIVERGING, INK, MAIN, NEUTRAL, chart_title, segment_colors

FEATURE_LABELS = {**LRFMC_LABELS, TREND: "T · Tendencia de ingreso"}
SHORT_LABELS = {
    "L": "L · Antigüedad",
    "R": "R · Recencia",
    "F": "F · Vuelos",
    "M": "M · Km",
    "C": "C · Descuento",
    TREND: "T · Tendencia",
}
Z_DOMAIN = 1.6  # los centroides caen en ±1,5: misma escala en todos los gráficos


def _long_centroids(
    profiles: pd.DataFrame, labels: dict[str, str] = FEATURE_LABELS
) -> pd.DataFrame:
    data = profiles.melt(
        id_vars="segment",
        value_vars=[f"z_{c}" for c in labels],
        var_name="feature",
        value_name="z",
    )
    data["feature"] = data["feature"].str.removeprefix("z_").map(labels)
    return data


def centroid_heatmap(profiles: pd.DataFrame) -> alt.LayerChart:
    """Centroide de cada segmento en z-score (0 = socio medio)."""
    data = _long_centroids(profiles, SHORT_LABELS)
    data["abs"] = data["z"].abs()
    base = alt.Chart(data).encode(
        x=alt.X(
            "feature:N",
            title=None,
            sort=list(SHORT_LABELS.values()),
            axis=alt.Axis(labelAngle=0, orient="top", labelLimit=120),
        ),
        y=alt.Y(
            "segment:N", title=None, sort=list(profiles["segment"]), axis=alt.Axis(labelLimit=180)
        ),
        tooltip=[
            alt.Tooltip("segment:N", title="Segmento"),
            alt.Tooltip("feature:N", title="Feature"),
            alt.Tooltip("z:Q", title="z-score", format="+.2f"),
        ],
    )
    cells = base.mark_rect(cornerRadius=3).encode(
        color=alt.Color(
            "z:Q",
            title="z-score",
            scale=alt.Scale(domain=[-Z_DOMAIN, 0, Z_DOMAIN], range=list(DIVERGING), clamp=True),
        )
    )
    text = base.mark_text(fontSize=12, fontWeight=600).encode(
        text=alt.Text("z:Q", format="+.2f"),
        color=alt.condition("datum.abs > 0.9", alt.value("white"), alt.value("#1f1f1d")),
    )
    return (cells + text).properties(
        width="container",
        height=alt.Step(42),
        title=chart_title(
            "Perfil de cada segmento",
            subtitle="Centroide en z-score: 0 = socio medio · azul = por debajo · rojo = por "
            "encima · R alto = más tiempo sin volar · C alto = tarifas cercanas a la completa",
        ),
    )


def size_value_chart(profiles: pd.DataFrame) -> alt.Chart:
    """% de socios frente a % del ingreso total de cada segmento."""
    data = profiles.melt(
        id_vars="segment",
        value_vars=["pct", "revenue_share"],
        var_name="medida",
        value_name="valor",
    )
    data["medida"] = data["medida"].map({"pct": "% de socios", "revenue_share": "% del ingreso"})
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=3, height={"band": 0.8})
        .encode(
            x=alt.X("valor:Q", title=None, axis=alt.Axis(format="%")),
            y=alt.Y(
                "segment:N",
                title=None,
                sort=list(profiles["segment"]),
                axis=alt.Axis(labelLimit=180),
            ),
            yOffset=alt.YOffset("medida:N", sort=["% de socios", "% del ingreso"]),
            color=alt.Color(
                "medida:N",
                title=None,
                scale=alt.Scale(domain=["% de socios", "% del ingreso"], range=[NEUTRAL, MAIN]),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[
                alt.Tooltip("segment:N", title="Segmento"),
                alt.Tooltip("medida:N", title="Medida"),
                alt.Tooltip("valor:Q", title="Valor", format=".1%"),
            ],
        )
        .properties(
            width="container",
            height=alt.Step(16),
            title=chart_title(
                "Tamaño frente a valor",
                subtitle="Ingreso total = SUM_YR_1 + SUM_YR_2 (no entró al modelo)",
            ),
        )
    )


def segment_profile_chart(profiles: pd.DataFrame, segment: str) -> alt.LayerChart:
    """Perfil z-score de un segmento (barras) con el resto de segmentos como referencia (marcas)."""
    data = _long_centroids(profiles)
    data["este"] = data["segment"] == segment
    color = segment_colors(list(profiles["segment"]))[segment]
    y = alt.Y(
        "feature:N", title=None, sort=list(FEATURE_LABELS.values()), axis=alt.Axis(labelLimit=180)
    )
    x_scale = alt.Scale(domain=[-Z_DOMAIN, Z_DOMAIN], clamp=True)
    tooltip = [
        alt.Tooltip("segment:N", title="Segmento"),
        alt.Tooltip("feature:N", title="Feature"),
        alt.Tooltip("z:Q", title="z-score", format="+.2f"),
    ]
    others = (
        alt.Chart(data)
        .transform_filter("!datum.este")
        .mark_tick(color=NEUTRAL, thickness=2, size=14, opacity=0.8)
        .encode(x=alt.X("z:Q", scale=x_scale), y=y, tooltip=tooltip)
    )
    bars = (
        alt.Chart(data)
        .transform_filter("datum.este")
        .mark_bar(color=color, height=16, cornerRadius=3)
        .encode(
            x=alt.X("z:Q", title="z-score (0 = socio medio)", scale=x_scale),
            y=y,
            tooltip=tooltip,
        )
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=INK).encode(x="x:Q")
    return (zero + others + bars).properties(
        width="container",
        height=alt.Step(30),
        title=chart_title(
            f"Perfil de «{segment}»",
            subtitle="Barras = este segmento · marcas grises = los demás segmentos",
        ),
    )
