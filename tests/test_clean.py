import json

import numpy as np
import pandas as pd
import pytest

from segmentacion.data.clean import clean, lineage_report, load_clean, save_clean


def steps_by_rule(lineage):
    return {step.rule: step for step in lineage}


def test_clean_fixture_keeps_all_rows_and_parses_dates(flight_df):
    df, lineage = clean(flight_df)
    assert len(df) == len(flight_df)
    assert all(step.rows == 0 for step in lineage)
    assert pd.api.types.is_datetime64_any_dtype(df["FFP_DATE"])
    assert df.loc[0, "LOAD_TIME"] == pd.Timestamp("2014-03-31")


def test_drops_null_revenue_rows(flight_df):
    flight_df.loc[[0, 1], "SUM_YR_1"] = np.nan
    flight_df.loc[2, "SUM_YR_2"] = np.nan
    df, lineage = clean(flight_df)
    assert steps_by_rule(lineage)["ingreso_nulo"].rows == 3
    assert not df["MEMBER_NO"].isin(flight_df.loc[[0, 1, 2], "MEMBER_NO"]).any()


def test_drops_zero_revenue_with_flights(flight_df):
    flight_df.loc[[0, 1], ["SUM_YR_1", "SUM_YR_2"]] = 0
    flight_df.loc[1, "avg_discount"] = 0  # canje puro: también se elimina
    _, lineage = clean(flight_df)
    assert steps_by_rule(lineage)["ingreso_cero_con_vuelos"].rows == 2


def test_age_out_of_range_becomes_nan(flight_df):
    flight_df.loc[[0, 1, 2], "AGE"] = [6.0, 110.0, 18.0]
    df, lineage = clean(flight_df)
    assert df.loc[[0, 1], "AGE"].isna().all()
    assert df.loc[2, "AGE"] == 18
    assert steps_by_rule(lineage)["edad_fuera_de_rango"].rows == 2


def test_invalid_last_flight_date_is_nulled_not_dropped(flight_df):
    flight_df.loc[0, "LAST_FLIGHT_DATE"] = "2014/2/29  0:00:00"
    df, lineage = clean(flight_df)
    assert len(df) == len(flight_df)
    assert pd.isna(df.loc[0, "LAST_FLIGHT_DATE"])
    assert steps_by_rule(lineage)["fecha_invalida_LAST_FLIGHT_DATE"].rows == 1


def test_invalid_ffp_date_drops_row(flight_df):
    flight_df.loc[0, "FFP_DATE"] = "2/30/2010"
    df, lineage = clean(flight_df)
    assert steps_by_rule(lineage)["fecha_invalida"].rows == 1
    assert len(df) == len(flight_df) - 1


def test_ffp_after_load_time_drops_row(flight_df):
    flight_df.loc[0, "FFP_DATE"] = "4/1/2014"
    _, lineage = clean(flight_df)
    assert steps_by_rule(lineage)["alta_posterior_al_corte"].rows == 1


@pytest.mark.parametrize("placeholder", [".", "-", "。", "0", "`", "  .  ", ""])
def test_location_placeholders_become_nan(flight_df, placeholder):
    flight_df.loc[0, ["WORK_CITY", "WORK_PROVINCE"]] = placeholder
    df, _ = clean(flight_df)
    assert df.loc[0, ["WORK_CITY", "WORK_PROVINCE"]].isna().all()


def test_text_is_normalized(flight_df):
    flight_df.loc[0, ["WORK_CITY", "WORK_PROVINCE", "WORK_COUNTRY"]] = [
        " Beijing ",
        "BEIJING",
        "cn",
    ]
    flight_df.loc[1, "WORK_COUNTRY"] = "中"
    df, lineage = clean(flight_df)
    assert df.loc[0, ["WORK_CITY", "WORK_PROVINCE", "WORK_COUNTRY"]].tolist() == [
        "beijing",
        "beijing",
        "CN",
    ]
    assert pd.isna(df.loc[1, "WORK_COUNTRY"])
    assert steps_by_rule(lineage)["pais_invalido"].rows == 1


def test_duplicate_member_no_raises(flight_df):
    with pytest.raises(Exception, match="MEMBER_NO"):
        clean(pd.concat([flight_df, flight_df.head(1)]))


def test_lineage_counts_are_consistent(flight_df):
    flight_df.loc[0, "SUM_YR_1"] = np.nan
    flight_df.loc[1, ["SUM_YR_1", "SUM_YR_2"]] = 0
    flight_df.loc[2, "FFP_DATE"] = "13/1/2010"
    df, lineage = clean(flight_df)
    report = lineage_report(len(flight_df), len(df), lineage)
    assert report["input_rows"] - report["dropped_rows"] == report["output_rows"] == len(df)
    assert report["dropped_rows"] == 3


def test_clean_is_idempotent(flight_df):
    flight_df.loc[0, "WORK_CITY"] = "."
    flight_df.loc[1, "AGE"] = 120.0
    flight_df.loc[2, "SUM_YR_2"] = np.nan
    once, _ = clean(flight_df)
    twice, lineage = clean(once)
    pd.testing.assert_frame_equal(once, twice)
    assert all(step.rows == 0 for step in lineage)


def test_save_and_load_roundtrip(tmp_path, flight_df):
    df, lineage = clean(flight_df)
    report = lineage_report(len(flight_df), len(df), lineage)
    clean_file, lineage_file = tmp_path / "out" / "clean.parquet", tmp_path / "out" / "l.json"
    save_clean(df, report, clean_file, lineage_file)
    pd.testing.assert_frame_equal(load_clean(clean_file), df, check_dtype=False)
    assert json.loads(lineage_file.read_text(encoding="utf-8"))["output_rows"] == len(df)
