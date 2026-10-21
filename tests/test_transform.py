import numpy as np
import pandas as pd
import pytest

from segmentacion.features.lrfmc import LRFMC_COLUMNS
from segmentacion.preprocessing.transform import (
    SCALERS,
    TRANSFORMS,
    compare_preprocessors,
    load_preprocessor,
    make_preprocessor,
    save_preprocessor,
    skewness,
)


@pytest.fixture
def skewed() -> pd.DataFrame:
    """Features con la forma de LRFMC: F y M log-normales (cola larga), C cerca de 0,7."""
    rng = np.random.default_rng(0)
    n = 2_000
    return pd.DataFrame(
        {
            "L": rng.uniform(12, 110, n),
            "R": rng.exponential(6, n),
            "F": np.round(rng.lognormal(2, 0.9, n)) + 2,
            "M": rng.lognormal(9.3, 0.9, n),
            "C": rng.normal(0.72, 0.15, n).clip(0.1, 1.5),
        },
        index=pd.RangeIndex(100, 100 + n),
    )


def test_standard_scaler_zero_mean_unit_std(skewed):
    scaled = make_preprocessor("yeo-johnson", "standard").fit_transform(skewed)
    assert scaled.mean().abs().max() < 1e-9
    assert np.allclose(scaled.std(ddof=0), 1)


@pytest.mark.parametrize("transform", ["yeo-johnson", "log1p"])
def test_transform_reduces_skew(skewed, transform):
    before = skewness(skewed).abs()
    after = skewness(make_preprocessor(transform, "standard").fit_transform(skewed)).abs()
    assert (after[["F", "M"]] < before[["F", "M"]]).all()
    assert after[["F", "M"]].max() < 1


def test_yeo_johnson_reduces_skew(skewed):
    after = skewness(make_preprocessor("yeo-johnson", "standard").fit_transform(skewed))
    assert after.abs().max() < 0.5


def test_robust_scaler_centers_on_median(skewed):
    scaled = make_preprocessor("none", "robust").fit_transform(skewed)
    assert np.allclose(scaled.median(), 0)


def test_output_keeps_columns_and_index(skewed):
    scaled = make_preprocessor().fit_transform(skewed)
    assert list(scaled.columns) == list(LRFMC_COLUMNS)
    assert scaled.index.equals(skewed.index)


def test_preprocessor_roundtrip_joblib(skewed, tmp_path):
    preprocessor = make_preprocessor("yeo-johnson", "standard").fit(skewed)
    path = tmp_path / "models" / "preprocessor.joblib"
    save_preprocessor(preprocessor, path)
    loaded = load_preprocessor(path)
    pd.testing.assert_frame_equal(loaded.transform(skewed), preprocessor.transform(skewed))


@pytest.mark.parametrize(("transform", "scaler"), [("box-cox", "standard"), ("log1p", "minmax")])
def test_unknown_options_raise(transform, scaler):
    with pytest.raises(ValueError, match="desconocid"):
        make_preprocessor(transform, scaler)


def test_compare_preprocessors_contract(skewed):
    table = compare_preprocessors(skewed.head(300), ks=(2, 3), sample_size=200)
    assert len(table) == len(TRANSFORMS) * len(SCALERS)
    expected = {
        "transform",
        "scaler",
        "n_abs_skew_lt_1",
        "std_ratio",
        "silhouette_k2",
        "min_cluster_pct_k3",
        "silhouette_mean",
    }
    assert expected <= set(table.columns)
    assert table["silhouette_mean"].between(-1, 1).all()
    # Las transformaciones corrigen la asimetría que deja el dato crudo.
    by_transform = table.groupby("transform")["max_abs_skew"].first()
    assert by_transform["yeo-johnson"] < by_transform["none"]


def test_standard_scaler_gives_equal_weights(skewed):
    table = compare_preprocessors(skewed.head(300), ks=(2,), sample_size=200).set_index(
        ["transform", "scaler"]
    )
    assert table.loc[("yeo-johnson", "standard"), "std_ratio"] == pytest.approx(1)
    assert table.loc[("log1p", "robust"), "std_ratio"] > 1
