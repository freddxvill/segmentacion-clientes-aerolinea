import pandas as pd
import pandera.errors as pa_errors
import pytest

from segmentacion.data.clean import clean
from segmentacion.data.schema import clean_schema, raw_schema


def test_raw_schema_accepts_valid_fixture(flight_df):
    raw_schema().validate(flight_df)


def test_raw_schema_rejects_duplicate_member_no(flight_df):
    duplicated = pd.concat([flight_df, flight_df.head(1)])
    with pytest.raises(pa_errors.SchemaError):
        raw_schema().validate(duplicated)


def test_raw_schema_rejects_negative_flight_count(flight_df):
    flight_df.loc[0, "FLIGHT_COUNT"] = -1
    with pytest.raises(pa_errors.SchemaError):
        raw_schema().validate(flight_df)


def test_raw_schema_rejects_missing_column(flight_df):
    with pytest.raises(pa_errors.SchemaError):
        raw_schema().validate(flight_df.drop(columns=["SEG_KM_SUM"]))


def test_clean_schema_rejects_unparsed_dates(flight_df):
    with pytest.raises(pa_errors.SchemaError):
        clean_schema().validate(flight_df)


def test_clean_schema_rejects_placeholder_city(flight_df):
    df, _ = clean(flight_df)
    df.loc[0, "WORK_CITY"] = "."
    with pytest.raises(pa_errors.SchemaError):
        clean_schema().validate(df)


def test_clean_schema_rejects_zero_revenue_with_flights(flight_df):
    df, _ = clean(flight_df)
    df.loc[0, ["SUM_YR_1", "SUM_YR_2"]] = 0
    with pytest.raises(pa_errors.SchemaError):
        clean_schema().validate(df)
