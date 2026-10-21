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


def test_lrfmc_preview_formulas(cleaned):
    df, _ = cleaned
    lrfmc = ec.lrfmc_preview(df)
    assert list(lrfmc.columns) == ["L", "R", "F", "M", "C"]
    row = df.iloc[0]
    expected_l = (row["LOAD_TIME"] - row["FFP_DATE"]).days / (365.25 / 12)
    assert lrfmc.iloc[0]["L"] == pytest.approx(expected_l)
    assert lrfmc.iloc[0]["R"] == pytest.approx(row["LAST_TO_END"] / 30)
    assert (lrfmc[["L", "F", "M"]] >= 0).all().all()


def test_revenue_balance_is_bounded(flight_df):
    flight_df.loc[0, ["SUM_YR_1", "SUM_YR_2"]] = [0, 500]
    flight_df.loc[1, ["SUM_YR_1", "SUM_YR_2"]] = [800, 0]
    flight_df.loc[2, ["SUM_YR_1", "SUM_YR_2"]] = [0, 0]
    balance = ec.revenue_balance(flight_df)
    assert balance.iloc[0] == 1 and balance.iloc[1] == -1
    assert balance.isna().iloc[2]
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
