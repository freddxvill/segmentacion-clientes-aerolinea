"""Fixtures compartidos: dataset sintético con el mismo esquema que `flight.csv`."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from segmentacion.config import CleaningParams, Config, DataParams, Paths
from segmentacion.data.load import EXPECTED_COLUMNS

LOAD_TIME = pd.Timestamp("2014-03-31")


def make_flight_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Genera `n` socios sintéticos con columnas y rangos plausibles."""
    rng = np.random.default_rng(seed)
    ffp_date = LOAD_TIME - pd.to_timedelta(rng.integers(365, 3650, n), unit="D")
    first_flight = ffp_date + pd.to_timedelta(rng.integers(0, 60, n), unit="D")
    last_to_end = rng.integers(1, 730, n)
    last_flight = LOAD_TIME - pd.to_timedelta(last_to_end, unit="D")
    flight_count = rng.integers(2, 120, n)
    seg_km = flight_count * rng.integers(500, 2500, n)
    fmt = "%m/%d/%Y"  # mismo formato que flight.csv
    df = pd.DataFrame(
        {
            "MEMBER_NO": np.arange(1, n + 1),
            "FFP_DATE": ffp_date.strftime(fmt),
            "FIRST_FLIGHT_DATE": first_flight.strftime(fmt),
            "GENDER": rng.choice(["Male", "Female"], n),
            "FFP_TIER": rng.choice([4, 5, 6], n, p=[0.9, 0.07, 0.03]),
            "WORK_CITY": rng.choice(["Guangzhou", "Beijing", "Shanghai"], n),
            "WORK_PROVINCE": rng.choice(["Guangdong", "Beijing", "Shanghai"], n),
            "WORK_COUNTRY": "CN",
            "AGE": rng.integers(20, 75, n).astype(float),
            "LOAD_TIME": LOAD_TIME.strftime(fmt),
            "FLIGHT_COUNT": flight_count,
            "SUM_YR_1": (seg_km * rng.uniform(0.3, 0.6, n)).round(),
            "SUM_YR_2": (seg_km * rng.uniform(0.3, 0.6, n)).round(),
            "SEG_KM_SUM": seg_km,
            "LAST_FLIGHT_DATE": last_flight.strftime(fmt),
            "LAST_TO_END": last_to_end,
            "AVG_INTERVAL": rng.uniform(5, 180, n).round(2),
            "MAX_INTERVAL": rng.integers(10, 365, n),
            "avg_discount": rng.uniform(0.2, 1.5, n).round(4),
            "BP_SUM": seg_km // 2,
            "EXCHANGE_COUNT": rng.poisson(0.5, n),
            "Points_Sum": seg_km // 2 + rng.integers(0, 500, n),
            "Point_NotFlight": rng.integers(0, 50, n),
        }
    )
    assert tuple(df.columns) == EXPECTED_COLUMNS
    return df


@pytest.fixture
def flight_df() -> pd.DataFrame:
    return make_flight_df()


@pytest.fixture
def tmp_config(tmp_path: Path) -> Config:
    """Config apuntando a directorios temporales."""
    paths = Paths(
        raw_dir=tmp_path / "data" / "raw",
        interim_dir=tmp_path / "data" / "interim",
        processed_dir=tmp_path / "data" / "processed",
        models_dir=tmp_path / "models",
        data_card=tmp_path / "data-card.md",
    )
    paths.raw_dir.mkdir(parents=True)
    data = DataParams(
        kaggle_competition="sa-customer-segmentation",
        source_files=("flight_train.csv", "flight_test.csv"),
        raw_filename="flight.csv",
        max_raw_size_mb=20,
        clean_filename="flight_clean.parquet",
        lineage_filename="cleaning_lineage.json",
    )
    return Config(paths=paths, data=data, cleaning=CleaningParams(), random_state=42)


@pytest.fixture
def flight_csv(tmp_config: Config, flight_df: pd.DataFrame) -> Path:
    """`flight.csv` sintético escrito en el raw_dir temporal."""
    flight_df.to_csv(tmp_config.raw_file, index=False)
    return tmp_config.raw_file
