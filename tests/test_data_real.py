"""Tests sobre el dataset real. Se ejecutan con `uv run poe test-data`."""

import numpy as np
import pandas as pd
import pytest

from segmentacion.config import get_config
from segmentacion.data.clean import clean
from segmentacion.data.download import count_rows, file_size_mb
from segmentacion.data.load import EXPECTED_COLUMNS, load_raw, read_header
from segmentacion.data.schema import clean_schema
from segmentacion.features.lrfmc import LRFMC_COLUMNS, build_lrfmc
from segmentacion.models.k_selection import evaluate_k_range
from segmentacion.models.kmeans import stability_ari
from segmentacion.preprocessing.transform import make_preprocessor

CONFIG = get_config()

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not CONFIG.raw_file.exists(),
        reason=f"Falta {CONFIG.raw_file}; ejecuta `uv run poe download`.",
    ),
]


def test_raw_file_under_20mb():
    assert file_size_mb(CONFIG.raw_file) < CONFIG.data.max_raw_size_mb


def test_raw_columns_match_expected():
    header = read_header(CONFIG.raw_file)
    missing = [c for c in EXPECTED_COLUMNS if c not in header]
    assert not missing, f"Columnas faltantes: {missing}"


def test_raw_is_full_population_of_sources():
    # train (55.000) + test (7.988) = 62.988 socios, el tamaño conocido del dataset LRFMC.
    assert count_rows(CONFIG.raw_file) == sum(count_rows(p) for p in CONFIG.source_paths)
    assert count_rows(CONFIG.raw_file) == 62_988


def test_member_no_is_unique():
    member_no = pd.read_csv(CONFIG.raw_file, usecols=["MEMBER_NO"])["MEMBER_NO"]
    assert member_no.is_unique


@pytest.fixture(scope="module")
def cleaned():
    raw = load_raw(CONFIG.raw_file)
    df, lineage = clean(raw, CONFIG.cleaning)
    return raw, df, lineage


def test_clean_retains_majority_of_rows(cleaned):
    raw, df, _ = cleaned
    assert len(df) / len(raw) >= 0.90


def test_clean_output_passes_clean_schema(cleaned):
    _, df, _ = cleaned
    clean_schema(CONFIG.cleaning).validate(df)


def test_invalid_feb_29_dates_are_kept_as_null(cleaned):
    # 421 socios tienen LAST_FLIGHT_DATE = "2014/2/29" (fecha inexistente) y LAST_TO_END válido.
    _, _, lineage = cleaned
    step = next(s for s in lineage if s.rule == "fecha_invalida_LAST_FLIGHT_DATE")
    assert step.action == "set_null" and step.rows == 421


def test_lrfmc_real_is_finite_and_non_negative(cleaned):
    _, df, _ = cleaned
    lrfmc = build_lrfmc(df)[list(LRFMC_COLUMNS)]
    assert len(lrfmc) == len(df)
    assert np.isfinite(lrfmc.to_numpy()).all()
    assert (lrfmc >= 0).all().all()
    # Todos los socios se dieron de alta al menos 12 meses antes del corte (ver data card).
    assert lrfmc["L"].min() >= 11.9


@pytest.fixture(scope="module")
def scaled_features(cleaned):
    # Cadena completa desde el CSV crudo: features → preprocesador elegido.
    _, df, _ = cleaned
    pre = CONFIG.preprocessing
    return make_preprocessor(pre.transform, pre.scaler).fit_transform(
        build_lrfmc(df)[list(LRFMC_COLUMNS)]
    )


def test_final_silhouette_above_threshold(scaled_features):
    features = scaled_features
    k = CONFIG.k_selection.chosen_k
    metrics = evaluate_k_range(features, [k], CONFIG.k_selection.silhouette_sample_size).iloc[0]
    assert metrics["silhouette"] >= CONFIG.quality.min_silhouette
    assert metrics["min_cluster_pct"] >= CONFIG.quality.min_segment_pct


def test_kmeans_stability_meets_main_criterion(scaled_features):
    # Criterio principal del proyecto: los segmentos no dependen de la semilla.
    seeds = range(CONFIG.modeling.stability_seeds)
    pairs = stability_ari(scaled_features, CONFIG.k_selection.chosen_k, seeds)
    assert pairs["ari"].mean() >= CONFIG.quality.min_stability_ari
