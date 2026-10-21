import numpy as np
import pandas as pd
import pytest

from segmentacion.segments.profile import (
    CRAMERS_V_THRESHOLDS,
    association_tests,
    centroids_z,
    cramers_v,
    kruskal_epsilon,
    magnitude,
    segment_kpis,
    trend_z,
)


@pytest.fixture
def members() -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1], 100)
    lrfmc = pd.DataFrame({c: rng.normal(size=200) for c in "LRFMC"})
    profile = pd.DataFrame(
        {
            # El clúster 1 ingresa 10 veces más y está en niveles altos del programa.
            "revenue_total": np.where(labels == 1, 1_000.0, 100.0) + rng.uniform(0, 10, 200),
            "revenue_trend": np.where(labels == 1, 0.5, -0.5),
            "km_per_flight": 1_000.0,
            "FFP_TIER": np.where(labels == 1, 6, 4),
            "GENDER": rng.choice(["Male", "Female"], 200),
            "AGE": rng.integers(20, 70, 200).astype(float),
            "EXCHANGE_COUNT": 1,
        }
    )
    return labels, lrfmc, profile


def test_centroids_z_keeps_feature_names():
    table = centroids_z(np.array([[0.5, -1.0], [0.0, 2.0]]), ["F", "M"])
    assert list(table.columns) == ["F", "M"] and table.index.name == "cluster"


def test_trend_z_sign_follows_group_means(members):
    labels, _, profile = members
    z = trend_z(labels, profile["revenue_trend"])
    assert z[1] > 0 > z[0]
    assert z[1] == pytest.approx(1.0)  # ±0,5 con desviación 0,5


def test_segment_kpis_hand_calculated(members):
    labels, lrfmc, profile = members
    kpis = segment_kpis(labels, lrfmc, profile)
    assert kpis["pct"].sum() == pytest.approx(1)
    assert kpis["members"].tolist() == [100, 100]
    assert kpis["revenue_share"].sum() == pytest.approx(1)
    assert kpis.loc[1, "revenue_share"] > 0.9
    assert kpis["elite_pct"].tolist() == [0, 1]
    assert kpis["falling_pct"].tolist() == [1, 0]


def test_cramers_v_perfect_and_independent():
    labels = np.repeat([0, 1], 50)
    assert cramers_v(labels, pd.Series(np.where(labels == 1, "a", "b")))["effect"] == pytest.approx(
        1
    )
    rng = np.random.default_rng(1)
    independent = pd.Series(rng.choice(["a", "b"], 100_000))
    noise = cramers_v(rng.integers(0, 3, 100_000), independent)["effect"]
    assert noise < 0.02


def test_cramers_v_ignores_missing_values():
    labels = np.array([0, 0, 1, 1, 1])
    other = pd.Series(["a", "a", "b", "b", None])
    assert cramers_v(labels, other)["effect"] == pytest.approx(1)


def test_kruskal_epsilon_large_for_separated_groups():
    labels = np.repeat([0, 1, 2], 100)
    values = pd.Series(labels * 10.0 + np.random.default_rng(0).normal(size=300))
    result = kruskal_epsilon(labels, values)
    assert result["dof"] == 2
    assert result["effect"] > 0.8 and result["p_value"] < 1e-10


def test_magnitude_thresholds():
    assert magnitude(0.01, CRAMERS_V_THRESHOLDS) == "despreciable"
    assert magnitude(0.07, CRAMERS_V_THRESHOLDS) == "pequeño"
    assert magnitude(0.29, CRAMERS_V_THRESHOLDS) == "mediano"
    assert magnitude(0.5, CRAMERS_V_THRESHOLDS) == "grande"


def test_association_tests_report_effect_size(members):
    labels, _, profile = members
    tests = association_tests(labels, profile).set_index("variable")
    assert list(tests.index) == ["FFP_TIER", "GENDER", "revenue_total", "revenue_trend", "AGE"]
    assert tests.loc["FFP_TIER", "effect_name"] == "V de Cramér"
    assert tests.loc["revenue_total", "effect_name"] == "ε²"
    assert tests.loc["FFP_TIER", "magnitude"] == "grande"
    assert tests.loc["revenue_total", "magnitude"] == "grande"
    assert tests["effect"].between(0, 1).all()
