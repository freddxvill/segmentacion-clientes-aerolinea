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
