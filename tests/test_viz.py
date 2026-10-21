import json

import altair as alt
import pytest

from segmentacion.data.clean import clean, lineage_report, null_summary
from segmentacion.viz import eda_charts as ec
from segmentacion.viz.fmt import num, pct
from segmentacion.viz.theme import CATEGORICAL, plotly_color_map, segment_colors, segment_scale

DF_CHARTS = [
    ec.distribution_chart,
    ec.skewness_chart,
    ec.correlation_heatmap,
    ec.frequency_km_scatter,
    ec.tier_boxplot_chart,
    ec.age_gender_chart,
    ec.top_locations_chart,
    ec.enrollment_chart,
    ec.revenue_trend_chart,
]
REPORT_CHARTS = [ec.null_values_chart, ec.cleaning_funnel_chart, ec.nulled_values_chart]


@pytest.fixture
def clean_df(flight_df):
    flight_df.loc[0, "WORK_CITY"] = "."
    flight_df.loc[1, "AGE"] = 120.0
    flight_df.loc[2, "SUM_YR_1"] = None
    return flight_df


@pytest.fixture
def cleaned(clean_df):
    df, lineage = clean(clean_df)
    report = lineage_report(len(clean_df), len(df), lineage)
    report["null_counts"] = null_summary(clean_df, df)
    return df, report


def _titles(spec: dict) -> list[str]:
    """Todos los títulos de ejes del spec (en cualquier nivel)."""
    found = []
    if isinstance(spec, dict):
        for key, value in spec.items():
            if key in ("x", "y") and isinstance(value, dict) and "title" in value:
                found.append(value["title"])
            found.extend(_titles(value))
    elif isinstance(spec, list):
        for item in spec:
            found.extend(_titles(item))
    return found


@pytest.mark.parametrize("make_chart", DF_CHARTS, ids=lambda f: f.__name__)
def test_eda_charts_return_altair_chart(cleaned, make_chart):
    df, _ = cleaned
    chart = make_chart(df)
    assert isinstance(chart, alt.TopLevelMixin)


@pytest.mark.parametrize("make_chart", DF_CHARTS + REPORT_CHARTS, ids=lambda f: f.__name__)
def test_chart_spec_is_valid_vegalite(cleaned, make_chart):
    df, report = cleaned
    chart = make_chart(report if make_chart in REPORT_CHARTS else df)
    spec = chart.to_dict(validate=True)
    assert "title" in json.dumps(spec)
    assert "tooltip" in json.dumps(spec)
    # Ningún eje queda con el nombre técnico de la columna.
    assert not any(t in {"socios", "inicio", "x0", "pct"} for t in _titles(spec))


def test_lrfmc_frame_uses_official_features(cleaned):
    df, _ = cleaned
    frame = ec.lrfmc_frame(df)
    assert list(frame.columns) == ["L", "R", "F", "M", "C"]
    assert frame.index.equals(df.index)


def test_revenue_balance_is_bounded(flight_df):
    flight_df.loc[0, ["SUM_YR_1", "SUM_YR_2"]] = [0, 500]
    flight_df.loc[1, ["SUM_YR_1", "SUM_YR_2"]] = [800, 0]
    balance = ec.revenue_balance(flight_df)
    assert balance.iloc[0] == 1 and balance.iloc[1] == -1
    assert balance.dropna().between(-1, 1).all()


def test_eda_summary_has_consistent_shares(cleaned):
    df, _ = cleaned
    summary = ec.eda_summary(df)
    assert summary["n_members"] == len(df)
    for key in ("elite_share", "elite_revenue_share", "male_share", "balance_lost"):
        assert 0 <= summary[key] <= 1


def test_lineage_table_lists_only_rules_with_effect(cleaned):
    _, report = cleaned
    table = ec.lineage_table(report)
    assert (table["Filas"] > 0).all()
    assert set(table["Acción"]) <= {"Elimina el socio", "Anula el valor"}


def test_segment_palette_consistent():
    segments = ["VIP", "En riesgo", "Nuevos", "Ocasionales"]
    scale = segment_scale(segments).to_dict()
    assert dict(zip(scale["domain"], scale["range"], strict=True)) == plotly_color_map(segments)
    assert list(segment_colors(segments).values()) == list(CATEGORICAL[:4])


def test_segment_palette_rejects_too_many_segments():
    with pytest.raises(ValueError, match="Máximo"):
        segment_colors([f"S{i}" for i in range(len(CATEGORICAL) + 1)])


def test_spanish_number_format():
    assert num(62044) == "62.044"
    assert num(1234.567, 2) == "1.234,57"
    assert pct(0.3171) == "31,7 %"
    assert num(-1e-17, 2) == "0,00"


@pytest.fixture
def notebook_renderer():
    from segmentacion.viz.notebook import enable_notebook_renderer

    previous = alt.renderers.active
    enable_notebook_renderer()
    yield
    alt.renderers.enable(previous)


def test_notebook_renderer_embeds_static_png_fallback(cleaned, notebook_renderer):
    df, _ = cleaned
    bundle = ec.enrollment_chart(df)._repr_mimebundle_()
    bundle = bundle[0] if isinstance(bundle, tuple) else bundle
    html = bundle["text/html"]
    assert "data:image/png;base64," in html  # visible en GitHub (sin JavaScript)
    assert "vegaEmbed" in html  # interactivo en JupyterLab / VS Code / nbviewer


@pytest.fixture
def comparison(skewed_lrfmc):
    from segmentacion.preprocessing.transform import compare_preprocessors

    return compare_preprocessors(skewed_lrfmc, ks=(2, 3), sample_size=150)


@pytest.fixture
def skewed_lrfmc(cleaned):
    from segmentacion.features.lrfmc import build_lrfmc

    df, _ = cleaned
    return build_lrfmc(df)[["L", "R", "F", "M", "C"]]


def test_model_charts_are_valid_vegalite(comparison, skewed_lrfmc):
    from segmentacion.preprocessing.transform import make_preprocessor
    from segmentacion.viz import model_charts as mc

    scaled = make_preprocessor().fit_transform(skewed_lrfmc)
    charts = [
        mc.transform_comparison_chart(comparison, chosen=("yeo-johnson", "standard")),
        mc.skew_comparison_chart(comparison),
        mc.scaled_distribution_chart(scaled),
    ]
    for chart in charts:
        spec = json.dumps(chart.to_dict(validate=True))
        assert "title" in spec and "tooltip" in spec


def test_transform_comparison_highlights_chosen_option(comparison):
    from segmentacion.viz import model_charts as mc

    chart = mc.transform_comparison_chart(comparison, chosen=("log1p", "robust"))
    rows = chart.to_dict()["datasets"]
    chosen = [r for rows_ in rows.values() for r in rows_ if r.get("elegida")]
    assert [r["opcion"] for r in chosen] == ["log1p + RobustScaler"]


@pytest.fixture
def segments():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame(
        {
            "cluster": rng.integers(0, 4, n),
            "baseline_segment": rng.choice(["Campeones", "Leales", "Hibernando"], n),
            "gmm_cluster": rng.integers(0, 4, n),
            "agglomerative_cluster": rng.integers(0, 4, n),
            "dbscan_outlier": rng.random(n) < 0.05,
        }
    )


def test_clustering_charts_are_valid_vegalite(segments):
    import numpy as np
    import pandas as pd

    from segmentacion.viz import model_charts as mc

    stability = pd.DataFrame(
        {"method": ["semillas"] * 3 + ["submuestras"] * 2, "run": list("01201"), "ari": 0.95}
    )
    curve = pd.DataFrame(
        {"cuantil": np.linspace(0, 1, 21), "distancia": np.linspace(0, 1, 21) ** 4}
    )
    charts = [
        mc.stability_chart(stability, threshold=0.9),
        mc.model_sizes_chart(segments),
        mc.k_distance_chart(curve, eps=0.4, min_samples=10),
        mc.crosstab_chart(segments, "baseline_segment", "K-Means vs quintiles"),
        mc.crosstab_chart(segments, "gmm_cluster", "K-Means vs GMM"),
    ]
    for chart in charts:
        spec = json.dumps(chart.to_dict(validate=True), ensure_ascii=False)
        assert "title" in spec and "tooltip" in spec


def test_crosstab_rows_sum_to_one_and_follow_rule_order(segments):
    from segmentacion.viz import model_charts as mc

    spec = mc.crosstab_chart(segments, "baseline_segment", "t").to_dict()
    rows = next(iter(spec["datasets"].values()))
    totals = {}
    for row in rows:
        totals[row["cluster"]] = totals.get(row["cluster"], 0) + row["pct"]
    assert all(abs(t - 1) < 1e-9 for t in totals.values())
    assert spec["layer"][0]["encoding"]["x"]["sort"] == ["Campeones", "Leales", "Hibernando"]


def test_model_sizes_chart_shows_dbscan_noise(segments):
    from segmentacion.viz import model_charts as mc

    rows = next(iter(mc.model_sizes_chart(segments).to_dict()["datasets"].values()))
    noise = [r for r in rows if r["grupo"] == "Ruido (atípicos)"]
    assert len(noise) == 1 and noise[0]["pct"] == segments["dbscan_outlier"].mean()
    for model in mc.SEGMENT_COLUMNS:
        assert abs(sum(r["pct"] for r in rows if r["modelo"] == model) - 1) < 1e-9


@pytest.fixture
def projections():
    import numpy as np
    import pandas as pd

    from segmentacion.projection.reduce import pca_project

    rng = np.random.default_rng(1)
    features = pd.DataFrame(rng.normal(size=(120, 5)), columns=list("LRFMC"))
    clusters = rng.integers(0, 4, len(features))
    pca = pca_project(features, n=3)
    points = pca.scores.assign(cluster=clusters, MEMBER_NO=range(len(features)))
    tsne = pd.concat(
        [
            pd.DataFrame(
                {
                    "MEMBER_NO": range(len(features)),
                    "cluster": clusters,
                    "perplexity": perplexity,
                    "TSNE1": rng.normal(size=len(features)).astype("float32"),
                    "TSNE2": rng.normal(size=len(features)).astype("float32"),
                }
            )
            for perplexity in (30, 50)
        ]
    )
    return pca, points, tsne


def test_projection_charts_are_valid_vegalite(projections):
    from segmentacion.viz import projection_charts as pc

    pca, points, tsne = projections
    charts = [
        pc.pca_variance_chart(pca.explained),
        pc.pca_loadings_chart(pca.loadings),
        pc.pca_scatter_chart(points, pca.loadings, pca.explained),
        pc.tsne_chart(tsne, 30),
    ]
    for chart in charts:
        spec = json.dumps(chart.to_dict(validate=True), ensure_ascii=False)
        assert "title" in spec and "tooltip" in spec


def test_tsne_chart_switches_perplexity_and_is_compact(projections):
    from segmentacion.viz import projection_charts as pc

    _, _, tsne = projections
    spec = pc.tsne_chart(tsne, 50).to_dict()
    values = spec["data"]["values"]
    assert len(values) == tsne["MEMBER_NO"].nunique()  # una fila por socio
    assert set(values[0]) == {"c", "x30", "y30", "x50", "y50"}
    assert all(len(str(v)) <= 6 for v in values[0].values())  # sin float32 sin redondear
    param = next(p for p in spec["params"] if p["name"] == "perplexity")
    assert param["value"] == 50 and param["bind"]["options"] == [30, 50]


def test_arrow_labels_merge_parallel_features():
    import pandas as pd

    from segmentacion.viz.projection_charts import _arrow_labels

    arrows = pd.DataFrame(
        {"letra": ["F", "M", "R"], "PC1": [1.0, 1.0, -1.0], "PC2": [0.0, 0.1, 0.0]}
    )
    assert sorted(_arrow_labels(arrows)["texto"]) == ["F · M", "R"]


def test_pca_3d_uses_segment_palette(projections):
    from segmentacion.viz.charts_3d import figure_html, pca_3d_figure
    from segmentacion.viz.projection_charts import cluster_names
    from segmentacion.viz.theme import plotly_color_map

    pca, points, _ = projections
    fig = pca_3d_figure(points, pca.explained)
    names = sorted(cluster_names(points["cluster"]).unique())
    assert [t.name for t in fig.data] == names
    assert {t.name: t.marker.color for t in fig.data} == plotly_color_map(names)
    html = figure_html(fig)
    assert "cdn.plot.ly" in html and len(html) < 500_000  # plotly.js no va incrustado


def test_wrap_subtitle_breaks_long_text_at_separators():
    from segmentacion.viz.theme import SUBTITLE_MAX_CHARS, chart_title, wrap_subtitle

    text = (
        "Cada franja es un clúster (alto ∝ nº de socios) · línea discontinua = silhouette "
        "medio · valores < 0 = socios más cerca de otro clúster"
    )
    lines = wrap_subtitle(text)
    assert len(lines) > 1 and all(len(line) <= SUBTITLE_MAX_CHARS for line in lines)
    assert " · ".join(lines) == text  # solo se corta en los separadores
    assert wrap_subtitle("Corto") == ["Corto"]
    assert chart_title("T", "Corto").to_dict()["subtitle"] == ["Corto"]


@pytest.fixture
def segment_profiles():
    import pandas as pd

    from segmentacion.segments.naming import SEGMENT_ORDER

    rows = []
    for i, name in enumerate(SEGMENT_ORDER):
        rows.append(
            {
                "segment": name,
                **{f"z_{c}": (i - 2) * 0.5 for c in ["L", "R", "F", "M", "C", "T"]},
                "pct": 0.2,
                "revenue_share": [0.5, 0.2, 0.15, 0.1, 0.05][i],
            }
        )
    return pd.DataFrame(rows)


def test_segment_charts_are_valid_vegalite(segment_profiles):
    from segmentacion.viz import segment_charts as sc

    charts = [
        sc.centroid_heatmap(segment_profiles),
        sc.size_value_chart(segment_profiles),
        sc.segment_profile_chart(segment_profiles, "VIP"),
    ]
    for chart in charts:
        spec = json.dumps(chart.to_dict(validate=True), ensure_ascii=False)
        assert "title" in spec and "tooltip" in spec


def test_segment_profile_uses_segment_color(segment_profiles):
    from segmentacion.viz import segment_charts as sc
    from segmentacion.viz.theme import segment_colors

    spec = sc.segment_profile_chart(segment_profiles, "En riesgo de abandono").to_dict()
    colors = segment_colors(list(segment_profiles["segment"]))
    bar = next(layer for layer in spec["layer"] if layer["mark"]["type"] == "bar")
    assert bar["mark"]["color"] == colors["En riesgo de abandono"]
