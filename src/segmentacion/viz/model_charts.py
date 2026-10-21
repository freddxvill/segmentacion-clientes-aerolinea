"""Gráficos Altair del preprocesamiento y del modelado (`03-lrfmc.qmd` en adelante).

Igual que en `eda_charts`, cada función recibe datos ya calculados (tablas de comparación,
features escaladas) y devuelve un gráfico con datos pre-agregados.
"""

import altair as alt
import numpy as np
import pandas as pd

from segmentacion.features.lrfmc import LRFMC_COLUMNS
from segmentacion.models.baseline import BASELINE_SEGMENTS
from segmentacion.viz.eda_charts import LRFMC_LABELS
from segmentacion.viz.fmt import num
from segmentacion.viz.theme import CATEGORICAL, INK, MAIN, NEUTRAL, SEQUENTIAL, chart_title

ALERT = CATEGORICAL[7]  # rojo: umbrales y socios atípicos

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
        title=chart_title(
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
        title=chart_title(
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
        title=chart_title(
            "Features listas para K-Means",
            subtitle="Caja = cuartiles · línea blanca = mediana · bigotes = P1–P99",
        ),
    )


K_METRICS = {
    "inertia": "Inercia (WCSS) · buscar el codo",
    "silhouette": "Silhouette · ↑ mejor",
    "ch": "Calinski-Harabasz · ↑ mejor",
    "db": "Davies-Bouldin · ↓ mejor",
}


def k_metrics_chart(table: pd.DataFrame, chosen_k: int) -> alt.FacetChart:
    """Inercia, silhouette, CH y DB por *k*, con hover enlazado y el *k* elegido marcado."""
    data = table.melt(
        id_vars=["k", "min_cluster_pct", "negative_pct", "elbow"],
        value_vars=list(K_METRICS),
        var_name="metrica",
        value_name="valor",
    )
    data["panel"] = data["metrica"].map(K_METRICS)
    data["elegido"] = data["k"] == chosen_k
    hover = alt.selection_point(
        name="k_hover", fields=["k"], on="pointerover", nearest=True, empty=False
    )
    x = alt.X("k:Q", title="Número de segmentos (k)", axis=alt.Axis(tickMinStep=1, format="d"))
    tooltip = [
        alt.Tooltip("k:Q", title="k"),
        alt.Tooltip("panel:N", title="Métrica"),
        alt.Tooltip("valor:Q", title="Valor", format=",.3f"),
        alt.Tooltip("min_cluster_pct:Q", title="Clúster más pequeño", format=".1%"),
        alt.Tooltip("negative_pct:Q", title="Socios con silhouette < 0", format=".1%"),
    ]
    base = alt.Chart().encode(x=x)
    chosen = (
        alt.Chart()
        .mark_rule(color=MAIN, strokeDash=[4, 3], strokeWidth=1.5)
        .encode(x="k:Q")
        .transform_filter("datum.elegido")
    )
    line = base.mark_line(color=NEUTRAL, strokeWidth=2).encode(
        y=alt.Y("valor:Q", title=None, scale=alt.Scale(zero=False), axis=alt.Axis(format=",.3~r"))
    )
    points = (
        base.mark_point(filled=True, opacity=1)
        .encode(
            y="valor:Q",
            color=alt.condition("datum.elegido", alt.value(MAIN), alt.value(NEUTRAL)),
            size=alt.condition(hover, alt.value(160), alt.value(60)),
            tooltip=tooltip,
        )
        .add_params(hover)
    )
    return (
        alt.layer(chosen, line, points, data=data)
        .properties(width=250, height=160)
        .facet(
            facet=alt.Facet("panel:N", title=None, sort=list(K_METRICS.values())),
            columns=2,
            title=chart_title(
                "Métricas de K-Means según el número de segmentos",
                subtitle=f"Línea discontinua = k elegido ({chosen_k}) · pasa el cursor para "
                "comparar un mismo k en los cuatro paneles",
            ),
        )
        .resolve_scale(y="independent")
    )


def silhouette_plot_chart(curves: pd.DataFrame, chosen_k: int) -> alt.LayerChart:
    """Silhouette plot (una franja por clúster, ordenada) con selector de *k*."""
    frames = []
    for _, group in curves.groupby("k"):
        sizes = group.groupby("cluster")["socios"].first()
        share = sizes / sizes.sum()
        offset = share.cumsum() - share
        part = group.copy()
        part["y"] = (
            offset.loc[part["cluster"]].to_numpy()
            + part["posicion"] * share.loc[part["cluster"]].to_numpy() * 0.96
        )  # 4 % de hueco entre franjas
        part["pct"] = share.loc[part["cluster"]].to_numpy()
        part["media_k"] = (group.groupby("cluster")["media"].first() * share).sum()
        frames.append(part)
    data = pd.concat(frames, ignore_index=True)
    data["segmento"] = "Clúster " + (data["cluster"] + 1).astype(str)
    ks = sorted(data["k"].unique())
    k_param = alt.param(
        name="k_sil",
        value=chosen_k,
        bind=alt.binding_radio(options=ks, name="Número de segmentos (k): "),
    )
    base = alt.Chart(data).transform_filter("datum.k == k_sil")
    areas = base.mark_area(orient="horizontal", opacity=0.9).encode(
        x=alt.X(
            "silhouette:Q", title="Silhouette de cada socio", scale=alt.Scale(domain=[-0.3, 0.7])
        ),
        y=alt.Y("y:Q", title=None, axis=None, scale=alt.Scale(domain=[0, 1], reverse=True)),
        color=alt.Color(
            "segmento:N",
            title="Clúster",
            scale=alt.Scale(range=list(CATEGORICAL)),
            sort=[f"Clúster {i}" for i in range(1, max(ks) + 1)],
        ),
        detail="segmento:N",
        tooltip=[
            alt.Tooltip("segmento:N", title="Clúster"),
            alt.Tooltip("pct:Q", title="% de socios", format=".1%"),
            alt.Tooltip("media:Q", title="Silhouette medio del clúster", format=".3f"),
        ],
    )
    mean_rule = base.mark_rule(color=INK, strokeDash=[4, 3]).encode(
        x="mean(media_k):Q",
        tooltip=[alt.Tooltip("mean(media_k):Q", title="Silhouette medio", format=".3f")],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=NEUTRAL).encode(x="x:Q")
    return (
        (zero + areas + mean_rule)
        .add_params(k_param)
        .properties(
            width="container",
            height=320,
            title=chart_title(
                "Silhouette plot",
                subtitle="Cada franja es un clúster (alto ∝ nº de socios) · línea discontinua = "
                "silhouette medio · valores < 0 = socios más cerca de otro clúster",
            ),
        )
    )


STABILITY_LABELS = {
    "semillas": "Pares de semillas",
    "submuestras": "Submuestras del 80 %",
}


def stability_chart(stability: pd.DataFrame, threshold: float) -> alt.LayerChart:
    """ARI de cada comparación de estabilidad, con la media y el umbral exigido."""
    data = stability.assign(metodo=stability["method"].map(STABILITY_LABELS))
    y = alt.Y(
        "metodo:N",
        title=None,
        sort=list(STABILITY_LABELS.values()),
        axis=alt.Axis(labelLimit=280),
    )
    # Todas las capas comparten la escala: sin esto, la media (agregada) incluiría el 0.
    lower = min(0.85, float(data["ari"].min()) - 0.01, threshold - 0.05)
    scale = alt.Scale(domain=[lower, 1], zero=False)
    x = alt.X("ari:Q", title="ARI (1 = mismos segmentos)", scale=scale)
    ticks = (
        alt.Chart(data)
        .mark_tick(thickness=2, size=22, opacity=0.6, color=MAIN)
        .encode(
            x=x,
            y=y,
            tooltip=[
                alt.Tooltip("metodo:N", title="Comparación"),
                alt.Tooltip("run:N", title="Semillas / ejecución"),
                alt.Tooltip("ari:Q", title="ARI", format=".3f"),
            ],
        )
    )
    mean = (
        alt.Chart(data)
        .mark_point(shape="diamond", filled=True, size=140, color=INK, opacity=1)
        .encode(
            x=alt.X("mean(ari):Q", scale=scale),
            y=y,
            tooltip=[
                alt.Tooltip("metodo:N", title="Comparación"),
                alt.Tooltip("mean(ari):Q", title="ARI medio", format=".3f"),
                alt.Tooltip("min(ari):Q", title="ARI mínimo", format=".3f"),
                alt.Tooltip("count():Q", title="Comparaciones"),
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"x": [threshold]}))
        .mark_rule(color=ALERT, strokeDash=[4, 3], strokeWidth=1.5)
        .encode(x="x:Q")
    )
    return (rule + ticks + mean).properties(
        width="container",
        height=alt.Step(56),
        title=chart_title(
            "Estabilidad de K-Means",
            subtitle="Cada marca es una comparación · rombo = media · línea roja = umbral "
            f"(ARI ≥ {num(threshold, 2)})",
        ),
    )


MODEL_ORDER = ("K-Means", "Quintiles LRFMC", "GMM", "Agglomerative (Ward)", "DBSCAN")
# Columna de `segments.parquet` con las etiquetas de cada modelo (DBSCAN: `dbscan_outlier`).
SEGMENT_COLUMNS = {
    "K-Means": "cluster",
    "Quintiles LRFMC": "baseline_segment",
    "GMM": "gmm_cluster",
    "Agglomerative (Ward)": "agglomerative_cluster",
}


def _group_names(values: pd.Series) -> pd.Series:
    """Etiquetas legibles: los grupos numéricos pasan a «Grupo 1…»; los nombres se mantienen."""
    if pd.api.types.is_numeric_dtype(values):
        return "Grupo " + (values + 1).astype(str)
    return values.astype(str)


def model_sizes_chart(segments: pd.DataFrame) -> alt.Chart:
    """Reparto de los socios entre los grupos de cada modelo (ruido de DBSCAN en rojo)."""
    frames = []
    for model, column in SEGMENT_COLUMNS.items():
        shares = _group_names(segments[column]).value_counts(normalize=True)
        frames.append(
            pd.DataFrame(
                {
                    "modelo": model,
                    "grupo": shares.index,
                    "pct": shares.to_numpy(),
                    "orden": range(len(shares)),
                }
            )
        )
    noise = float(segments["dbscan_outlier"].mean())
    frames.append(
        pd.DataFrame(
            {
                "modelo": "DBSCAN",
                "grupo": ["Zona densa", "Ruido (atípicos)"],
                "pct": [1 - noise, noise],
                "orden": [0, 1],
            }
        )
    )
    data = pd.concat(frames, ignore_index=True)
    # Tramos alternados azul / gris para distinguir grupos sin sugerir que el «Grupo 1» de un
    # modelo equivale al de otro (las etiquetas no se corresponden entre modelos).
    data["color"] = np.where(
        data["grupo"] == "Ruido (atípicos)",
        ALERT,
        np.where(data["orden"] % 2 == 0, MAIN, NEUTRAL),
    )
    return (
        alt.Chart(data)
        .mark_bar(height=22, stroke="white", strokeWidth=1.5, cornerRadiusEnd=0)
        .encode(
            x=alt.X(
                "pct:Q",
                title="% de socios",
                stack="zero",
                axis=alt.Axis(format="%", values=[i / 10 for i in range(11)]),
            ),
            y=alt.Y("modelo:N", title=None, sort=list(MODEL_ORDER), axis=alt.Axis(labelLimit=200)),
            order=alt.Order("orden:Q"),
            color=alt.Color("color:N", scale=None),
            tooltip=[
                alt.Tooltip("modelo:N", title="Modelo"),
                alt.Tooltip("grupo:N", title="Grupo"),
                alt.Tooltip("pct:Q", title="% de socios", format=".1%"),
            ],
        )
        .properties(
            width="container",
            height=alt.Step(36),
            title=chart_title(
                "Tamaño de los grupos de cada modelo",
                subtitle="Cada tramo es un grupo, de mayor a menor · rojo = ruido de DBSCAN",
            ),
        )
    )


def k_distance_chart(curve: pd.DataFrame, eps: float, min_samples: int) -> alt.LayerChart:
    """Curva k-distance (ordenada) con el `eps` elegido en su codo."""
    line = (
        alt.Chart(curve)
        .mark_line(color=MAIN, strokeWidth=2)
        .encode(
            x=alt.X(
                "cuantil:Q",
                title="Socios ordenados por distancia",
                axis=alt.Axis(format="%", values=[i / 10 for i in range(11)]),
            ),
            y=alt.Y("distancia:Q", title=f"Distancia al {min_samples}.º vecino"),
            tooltip=[
                alt.Tooltip("cuantil:Q", title="Percentil", format=".1%"),
                alt.Tooltip("distancia:Q", title="Distancia", format=".3f"),
            ],
        )
    )
    knee = curve.loc[[(curve["distancia"] - eps).abs().idxmin()]].assign(eps=eps)
    rule = alt.Chart(knee).mark_rule(color=ALERT, strokeDash=[4, 3]).encode(y="eps:Q")
    point = (
        alt.Chart(knee)
        .mark_point(filled=True, size=90, color=ALERT, opacity=1)
        .encode(
            x="cuantil:Q",
            y="distancia:Q",
            tooltip=[
                alt.Tooltip("eps:Q", title="eps elegido", format=".3f"),
                alt.Tooltip("cuantil:Q", title="Socios por debajo", format=".1%"),
            ],
        )
    )
    return (line + rule + point).properties(
        width="container",
        height=280,
        title=chart_title(
            "Gráfico k-distance para elegir eps de DBSCAN",
            subtitle=f"Codo de la curva = eps ({num(eps, 2)}): por encima, la distancia se "
            "dispara y los socios quedan como ruido",
        ),
    )


def crosstab_chart(segments: pd.DataFrame, column: str, title: str) -> alt.LayerChart:
    """% de cada clúster de K-Means que cae en cada grupo de otra segmentación."""
    table = pd.crosstab(
        "Clúster " + (segments["cluster"] + 1).astype(str),
        _group_names(segments[column]),
        normalize="index",
    )
    # Segmentos del baseline en el orden de sus reglas; grupos numéricos en orden natural.
    order = [c for c in BASELINE_SEGMENTS if c in table.columns] or list(table.columns)
    data = table.stack().rename("pct").reset_index()
    data.columns = ["cluster", "grupo", "pct"]
    base = alt.Chart(data).encode(
        x=alt.X(
            "grupo:N",
            title=None,
            sort=order,
            axis=alt.Axis(labelAngle=0, orient="top", labelLimit=120),
        ),
        y=alt.Y("cluster:N", title="K-Means"),
        tooltip=[
            alt.Tooltip("cluster:N", title="K-Means"),
            alt.Tooltip("grupo:N", title="Grupo"),
            alt.Tooltip("pct:Q", title="% del clúster", format=".1%"),
        ],
    )
    cells = base.mark_rect(cornerRadius=3).encode(
        color=alt.Color(
            "pct:Q",
            title="% del clúster",
            scale=alt.Scale(range=list(SEQUENTIAL), domain=[0, 1]),
            legend=alt.Legend(format="%"),
        )
    )
    text = base.mark_text(fontSize=12, fontWeight=600).encode(
        text=alt.Text("pct:Q", format=".0%"),
        color=alt.condition("datum.pct > 0.5", alt.value("white"), alt.value("#1f1f1d")),
    )
    return (cells + text).properties(
        width="container",
        height=alt.Step(40),
        title=chart_title(
            title, subtitle="Cada fila suma 100 %: a qué grupo va cada clúster de K-Means"
        ),
    )
