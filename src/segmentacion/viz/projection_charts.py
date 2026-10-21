"""Gráficos Altair de las proyecciones PCA y t-SNE (Fase 8, `04-clustering.qmd`).

Los socios se colorean por clúster con la paleta fija de `theme.segment_scale`, la misma que usa
el PCA 3D de Plotly (`charts_3d`). Hacer clic en la leyenda resalta un clúster.
"""

import altair as alt
import numpy as np
import pandas as pd

from segmentacion.viz.eda_charts import LRFMC_LABELS
from segmentacion.viz.fmt import num, pct
from segmentacion.viz.theme import DIVERGING, INK, MAIN, NEUTRAL, chart_title, segment_scale


def cluster_names(clusters: pd.Series | np.ndarray) -> pd.Series:
    """0, 1, … → «Clúster 1», «Clúster 2», … (hasta tener los nombres de la Fase 9)."""
    return "Clúster " + (pd.Series(np.asarray(clusters)) + 1).astype(str)


def _segment_order(names: pd.Series) -> list[str]:
    return sorted(names.unique(), key=lambda name: int(name.rsplit(" ", 1)[1]))


def _legend_selection() -> alt.SelectionParameter:
    return alt.selection_point(name="segmento_sel", fields=["segmento"], bind="legend")


def _segment_color(order: list[str]) -> alt.Color:
    return alt.Color(
        "segmento:N",
        title="Clúster (clic para resaltar)",
        scale=segment_scale(order),
        # Etiquetas «1…5»: la leyenda horizontal no se parte en líneas y con «Clúster N» se
        # cortaría en móvil (y `columns` rompe el ancho "container" de Vega).
        legend=alt.Legend(
            orient="bottom",
            symbolOpacity=1,
            labelExpr="replace(datum.label, 'Clúster ', '')",
        ),
    )


def pca_variance_chart(explained: pd.DataFrame) -> alt.LayerChart:
    """Varianza explicada por componente (barras) y acumulada (línea)."""
    data = explained.assign(usada=explained.index < 3)
    x = alt.X("component:N", title="Componente principal", axis=alt.Axis(labelAngle=0))
    tooltip = [
        alt.Tooltip("component:N", title="Componente"),
        alt.Tooltip("variance:Q", title="Varianza explicada", format=".1%"),
        alt.Tooltip("cumulative:Q", title="Acumulada", format=".1%"),
    ]
    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=3, width=alt.RelativeBandSize(0.6))
        .encode(
            x=x,
            y=alt.Y(
                "variance:Q",
                title="% de la varianza",
                axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.condition("datum.usada", alt.value(MAIN), alt.value(NEUTRAL)),
            tooltip=tooltip,
        )
    )
    line = alt.Chart(data).mark_line(color=INK, strokeWidth=2).encode(x=x, y="cumulative:Q")
    points = (
        alt.Chart(data)
        .mark_point(filled=True, color=INK, size=50, opacity=1)
        .encode(x=x, y="cumulative:Q", tooltip=tooltip)
    )
    labels = (
        alt.Chart(data)
        .mark_text(dy=-10, fontSize=11, color=INK)
        .encode(x=x, y="cumulative:Q", text=alt.Text("cumulative:Q", format=".0%"))
    )
    cumulative = explained["cumulative"].to_numpy()
    return (bars + line + points + labels).properties(
        width="container",
        height=260,
        title=chart_title(
            "Varianza explicada por PCA",
            subtitle=f"Barras = cada componente (azul = las usadas en 3D) · línea = acumulada · "
            f"2D: {pct(cumulative[1])} · 3D: {pct(cumulative[2])}",
        ),
    )


def pca_loadings_chart(loadings: pd.DataFrame) -> alt.LayerChart:
    """Peso de cada feature en cada componente (−1 a 1)."""
    data = loadings.rename_axis("component").reset_index()
    data = data.melt(id_vars="component", var_name="feature", value_name="loading")
    data["feature"] = data["feature"].map(LRFMC_LABELS)
    base = alt.Chart(data).encode(
        x=alt.X(
            "feature:N",
            title=None,
            sort=list(LRFMC_LABELS.values()),
            axis=alt.Axis(labelAngle=0, labelLimit=140, orient="top"),
        ),
        y=alt.Y("component:N", title=None),
        tooltip=[
            alt.Tooltip("component:N", title="Componente"),
            alt.Tooltip("feature:N", title="Feature"),
            alt.Tooltip("loading:Q", title="Loading", format="+.2f"),
        ],
    )
    cells = base.mark_rect(cornerRadius=3).encode(
        color=alt.Color(
            "loading:Q",
            title="Loading",
            scale=alt.Scale(domain=[-1, 0, 1], range=list(DIVERGING), interpolate="lab"),
        )
    )
    text = base.mark_text(fontSize=12, fontWeight=600).encode(
        text=alt.Text("loading:Q", format="+.2f"),
        color=alt.condition("datum.loading < -0.75", alt.value("white"), alt.value("#1f1f1d")),
    )
    return (cells + text).properties(
        width="container",
        height=alt.Step(44),
        title=chart_title(
            "Qué representa cada componente",
            subtitle="Loading = peso de la feature en la componente · signo = dirección",
        ),
    )


def _arrow_labels(arrows: pd.DataFrame, min_angle: float = 15.0) -> pd.DataFrame:
    """Una etiqueta (letra LRFMC) por flecha, más allá de su punta.

    Las flechas casi paralelas (menos de `min_angle` grados) comparten etiqueta («F · M») para
    que los textos no se pisen.
    """
    angles = np.degrees(np.arctan2(arrows["PC2"], arrows["PC1"]))
    ordered = arrows.assign(angle=angles).sort_values("angle")
    groups, current, last = [], [], None
    for row in ordered.itertuples():
        if last is not None and row.angle - last >= min_angle:
            groups.append(current)
            current = []
        current.append(row)
        last = row.angle
    groups.append(current)
    reach = float(np.hypot(arrows["PC1"], arrows["PC2"]).max())
    rows = []
    for group in groups:
        angle = np.radians(np.mean([r.angle for r in group]))
        length = max(np.hypot(r.PC1, r.PC2) for r in group) + 0.15 * reach
        rows.append(
            {
                "texto": " · ".join(r.letra for r in group),
                "PC1": length * np.cos(angle),
                "PC2": length * np.sin(angle),
            }
        )
    return pd.DataFrame(rows)


def pca_scatter_chart(
    points: pd.DataFrame, loadings: pd.DataFrame, explained: pd.DataFrame
) -> alt.LayerChart:
    """Biplot PCA 2D: socios por clúster + flecha de cada feature (loadings)."""
    data = points[["PC1", "PC2"]].round(2)
    data["segmento"] = cluster_names(points["cluster"]).to_numpy()
    order = _segment_order(data["segmento"])
    selection = _legend_selection()
    variance = explained.set_index("component")["variance"]
    x = alt.X("PC1:Q", title=f"PC1 ({pct(variance['PC1'])})")
    y = alt.Y("PC2:Q", title=f"PC2 ({pct(variance['PC2'])})")
    scatter = (
        alt.Chart(data)
        .mark_circle(size=14)
        .encode(
            x=x,
            y=y,
            color=_segment_color(order),
            opacity=alt.condition(selection, alt.value(0.6), alt.value(0.05)),
            tooltip=[
                alt.Tooltip("segmento:N", title="Clúster"),
                alt.Tooltip("PC1:Q", format=".2f"),
                alt.Tooltip("PC2:Q", format=".2f"),
            ],
        )
        .add_params(selection)
    )
    # Flechas: loadings escalados al rango de los puntos para que se vean sobre la nube.
    plane = loadings.loc[["PC1", "PC2"]]
    reach = float(np.abs(points[["PC1", "PC2"]]).quantile(0.95).min())
    factor = 0.8 * reach / float(np.abs(plane).to_numpy().max())
    arrows = pd.DataFrame(
        {
            "letra": list(loadings.columns),
            "feature": [LRFMC_LABELS[c] for c in loadings.columns],
            "x0": 0.0,
            "y0": 0.0,
            "PC1": plane.loc["PC1"].to_numpy() * factor,
            "PC2": plane.loc["PC2"].to_numpy() * factor,
            "l1": plane.loc["PC1"].to_numpy(),
            "l2": plane.loc["PC2"].to_numpy(),
        }
    )
    arrow_tooltip = [
        alt.Tooltip("feature:N", title="Feature"),
        alt.Tooltip("l1:Q", title="Loading PC1", format="+.2f"),
        alt.Tooltip("l2:Q", title="Loading PC2", format="+.2f"),
    ]
    rules = (
        alt.Chart(arrows)
        .mark_rule(color=INK, strokeWidth=2)
        .encode(x="x0:Q", y="y0:Q", x2="PC1:Q", y2="PC2:Q", tooltip=arrow_tooltip)
    )
    heads = (
        alt.Chart(arrows)
        .mark_point(shape="circle", filled=True, size=40, color=INK, opacity=1)
        .encode(x="PC1:Q", y="PC2:Q", tooltip=arrow_tooltip)
    )
    names = (
        alt.Chart(_arrow_labels(arrows))
        .mark_text(fontSize=15, fontWeight=700, color=INK)
        .encode(x="PC1:Q", y="PC2:Q", text="texto:N")
    )
    return (scatter + rules + heads + names).properties(
        width="container",
        height=420,
        title=chart_title(
            "Segmentos en el plano PCA (biplot)",
            subtitle=f"{num(len(points))} socios (muestra estratificada) · flechas = dirección "
            "en que crece cada feature",
        ),
    )


def tsne_chart(tsne: pd.DataFrame, default_perplexity: int | None = None) -> alt.Chart:
    """t-SNE 2D por clúster con selector de perplexity y leyenda clicable.

    Los datos se pasan en formato ancho (una fila por socio, dos columnas por perplexity) como
    valores inline con claves cortas: así no se repite el socio por cada perplexity, no aplica
    el límite de filas de Altair (8.000 socios) y el spec pesa menos de la mitad. El nombre del
    clúster se calcula en el navegador.
    """
    perplexities = sorted(int(p) for p in tsne["perplexity"].unique())
    default = default_perplexity if default_perplexity in perplexities else perplexities[0]
    wide = tsne.pivot_table(
        index=["MEMBER_NO", "cluster"], columns="perplexity", values=["TSNE1", "TSNE2"]
    )
    # Los ejes de t-SNE no tienen unidades: un decimal basta para dibujar.
    wide.columns = [f"{'x' if axis == 'TSNE1' else 'y'}{int(p)}" for axis, p in wide.columns]
    wide = wide.astype(float).reset_index().round(1)
    order = _segment_order(cluster_names(wide["cluster"]))
    values = wide.drop(columns="MEMBER_NO").rename(columns={"cluster": "c"})
    values = values.to_dict(orient="records")

    perplexity = alt.param(
        name="perplexity",
        value=default,
        bind=alt.binding_radio(options=perplexities, name="Perplexity: "),
    )
    selection = _legend_selection()

    def pick(axis: str) -> str:
        expr = f"datum.{axis}{perplexities[-1]}"
        for p in reversed(perplexities[:-1]):
            expr = f"perplexity == {p} ? datum.{axis}{p} : {expr}"
        return expr

    no_axis = alt.Axis(labels=False, ticks=False, grid=False, domain=False)
    return (
        alt.Chart(alt.Data(values=values))
        .transform_calculate(x=pick("x"), y=pick("y"), segmento="'Clúster ' + (datum.c + 1)")
        .mark_circle(size=10)
        .encode(
            x=alt.X("x:Q", title="t-SNE 1", axis=no_axis),
            y=alt.Y("y:Q", title="t-SNE 2", axis=no_axis),
            color=_segment_color(order),
            opacity=alt.condition(selection, alt.value(0.6), alt.value(0.04)),
            tooltip=[alt.Tooltip("segmento:N", title="Clúster")],
        )
        .add_params(perplexity, selection)
        .properties(
            width="container",
            height=440,
            title=chart_title(
                "Segmentos en t-SNE",
                subtitle=f"{num(len(wide))} socios (muestra estratificada por clúster) · los "
                "ejes no tienen unidades: solo importa qué socios quedan cerca",
            ),
        )
    )
