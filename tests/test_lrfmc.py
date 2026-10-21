import numpy as np
import pandas as pd
import pytest

from segmentacion.data.clean import clean
from segmentacion.features.lrfmc import (
    LRFMC_COLUMNS,
    build_lrfmc,
    build_profile_features,
    revenue_trend,
)


@pytest.fixture
def clean_df(flight_df):
    df, _ = clean(flight_df)
    return df


def hand_case() -> pd.DataFrame:
    """Dos socios con valores elegidos para calcular el resultado a mano."""
    return pd.DataFrame(
        {
            "MEMBER_NO": [1, 2],
            "FFP_DATE": pd.to_datetime(["2012-03-31", "2013-09-30"]),
            "LOAD_TIME": pd.to_datetime(["2014-03-31", "2014-03-31"]),
            "LAST_TO_END": [90, 15],
            "FLIGHT_COUNT": [3, 10],
            "SEG_KM_SUM": [3000, 25000],
            "avg_discount": [0.75, 1.2],
            "SUM_YR_1": [100.0, 0.0],
            "SUM_YR_2": [300.0, 0.0],
            "Points_Sum": [200, 0],
            "Point_NotFlight": [50, 0],
        }
    )


def test_length_in_months_hand_calculated():
    lrfmc = build_lrfmc(hand_case())
    # 2012-03-31 → 2014-03-31 = 730 días; 2013-09-30 → 2014-03-31 = 182 días.
    assert lrfmc["L"].tolist() == pytest.approx([730 / 30.4375, 182 / 30.4375])
    assert lrfmc.loc[0, "L"] == pytest.approx(23.98, abs=0.01)


def test_recency_from_last_to_end():
    lrfmc = build_lrfmc(hand_case())
    assert lrfmc["R"].tolist() == pytest.approx([3.0, 0.5])


def test_frequency_monetary_and_coefficient_are_passthrough():
    lrfmc = build_lrfmc(hand_case())
    assert lrfmc["F"].tolist() == [3, 10]
    assert lrfmc["M"].tolist() == [3000, 25000]
    assert lrfmc["C"].tolist() == pytest.approx([0.75, 1.2])


def test_output_one_row_per_member(clean_df):
    lrfmc = build_lrfmc(clean_df)
    assert list(lrfmc.columns) == ["MEMBER_NO", *LRFMC_COLUMNS]
    assert len(lrfmc) == len(clean_df)
    assert lrfmc["MEMBER_NO"].is_unique
    assert lrfmc.index.equals(clean_df.index)


def test_no_nan_or_inf(clean_df):
    lrfmc = build_lrfmc(clean_df)[list(LRFMC_COLUMNS)]
    assert np.isfinite(lrfmc.to_numpy()).all()
    assert (lrfmc[["L", "F", "M"]] >= 0).all().all()


def test_invalid_input_is_rejected():
    df = hand_case()
    df.loc[0, "LAST_TO_END"] = -5  # R negativa
    with pytest.raises(Exception, match="R"):
        build_lrfmc(df)


def test_revenue_trend_hand_calculated():
    trend = revenue_trend(hand_case())
    assert trend.tolist() == pytest.approx([0.5, 0.0])  # (300-100)/400; sin ingresos → 0


def test_revenue_trend_is_bounded(clean_df):
    clean_df.loc[0, ["SUM_YR_1", "SUM_YR_2"]] = [0, 500]
    clean_df.loc[1, ["SUM_YR_1", "SUM_YR_2"]] = [800, 0]
    trend = revenue_trend(clean_df)
    assert trend.iloc[0] == 1 and trend.iloc[1] == -1
    assert trend.between(-1, 1).all()


def test_profile_features_hand_calculated():
    profile = build_profile_features(
        hand_case().assign(
            FFP_TIER=4,
            GENDER="Male",
            AGE=40.0,
            WORK_PROVINCE="guangdong",
            WORK_CITY="guangzhou",
            AVG_INTERVAL=10.0,
            MAX_INTERVAL=30,
            EXCHANGE_COUNT=0,
        )
    )
    assert profile["revenue_total"].tolist() == [400.0, 0.0]
    assert profile["km_per_flight"].tolist() == [1000.0, 2500.0]
    assert profile["points_ratio"].tolist() == [0.25, 0.0]  # sin puntos → 0


def test_points_ratio_is_clipped_to_one(clean_df):
    clean_df.loc[0, ["Points_Sum", "Point_NotFlight"]] = [10, 40]
    assert build_profile_features(clean_df).loc[0, "points_ratio"] == 1.0


def test_profile_features_one_row_per_member(clean_df):
    profile = build_profile_features(clean_df)
    assert len(profile) == len(clean_df)
    derived = profile[["revenue_total", "revenue_trend", "km_per_flight", "points_ratio"]]
    assert np.isfinite(derived.to_numpy()).all()
    assert {"FFP_TIER", "GENDER", "AGE"} <= set(profile.columns)
