"""Tests sobre el dataset real. Se ejecutan con `uv run poe test-data`."""

import pandas as pd
import pytest

from segmentacion.config import get_config
from segmentacion.data.download import count_rows, file_size_mb
from segmentacion.data.load import EXPECTED_COLUMNS, read_header

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
