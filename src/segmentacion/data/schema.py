"""Esquemas pandera del dataset crudo y limpio (Fase 2)."""

import pandas as pd
import pandera.pandas as pa

from segmentacion.config import CleaningParams

DATE_COLUMNS: tuple[str, ...] = ("FFP_DATE", "FIRST_FLIGHT_DATE", "LOAD_TIME", "LAST_FLIGHT_DATE")
LOCATION_COLUMNS: tuple[str, ...] = ("WORK_CITY", "WORK_PROVINCE", "WORK_COUNTRY")
TEXT_COLUMNS: tuple[str, ...] = ("GENDER", *LOCATION_COLUMNS)

# Conteos y acumulados que nunca pueden ser negativos ni nulos.
NON_NEGATIVE_COLUMNS: tuple[str, ...] = (
    "FLIGHT_COUNT",
    "BP_SUM",
    "SEG_KM_SUM",
    "LAST_TO_END",
    "AVG_INTERVAL",
    "MAX_INTERVAL",
    "EXCHANGE_COUNT",
    "avg_discount",
    "Points_Sum",
    "Point_NotFlight",
)
REVENUE_COLUMNS: tuple[str, ...] = ("SUM_YR_1", "SUM_YR_2")

# Se valida "numérico" en vez de un dtype exacto: un subconjunto sin nulos se lee como int64
# y el archivo completo como float64.
_is_numeric = pa.Check(pd.api.types.is_numeric_dtype, element_wise=False, name="is_numeric")
_is_datetime = pa.Check(
    pd.api.types.is_datetime64_any_dtype, element_wise=False, name="is_datetime"
)


def _numeric(nullable: bool = False, *checks: pa.Check) -> pa.Column:
    return pa.Column(checks=[_is_numeric, *checks], nullable=nullable)


# Unicidad a nivel de columna: `DataFrameSchema(unique=[...])` falla al reportar duplicados
# con pandas 3 (pandera 0.34).
_member_no = pa.Column(checks=[_is_numeric, pa.Check.gt(0)], unique=True)


def raw_schema() -> pa.DataFrameSchema:
    """Contrato mínimo del CSV crudo: columnas presentes, IDs únicos y rangos no negativos."""
    columns = {
        "MEMBER_NO": _member_no,
        "FFP_TIER": _numeric(),
        "AGE": _numeric(True, pa.Check.ge(0)),
        **{c: _numeric(False, pa.Check.ge(0)) for c in NON_NEGATIVE_COLUMNS},
        **{c: _numeric(True, pa.Check.ge(0)) for c in REVENUE_COLUMNS},
        **{c: pa.Column(nullable=True) for c in (*DATE_COLUMNS, *TEXT_COLUMNS)},
    }
    return pa.DataFrameSchema(
        columns,
        name="RawFlightSchema",
    )


def clean_schema(params: CleaningParams | None = None) -> pa.DataFrameSchema:
    """Contrato del dataset limpio: tipos parseados, reglas de negocio y sin placeholders."""
    params = params or CleaningParams()
    no_placeholders = pa.Check.notin(list(params.text_placeholders), name="no_placeholders")
    columns = {
        "MEMBER_NO": _member_no,
        "FFP_TIER": _numeric(False, pa.Check.isin([4, 5, 6])),
        "AGE": _numeric(True, pa.Check.in_range(params.age_min, params.age_max)),
        **{c: _numeric(False, pa.Check.ge(0)) for c in NON_NEGATIVE_COLUMNS},
        **{c: _numeric(False, pa.Check.ge(0)) for c in REVENUE_COLUMNS},
        "FFP_DATE": pa.Column(checks=_is_datetime),
        "FIRST_FLIGHT_DATE": pa.Column(checks=_is_datetime),
        "LOAD_TIME": pa.Column(checks=_is_datetime),
        "LAST_FLIGHT_DATE": pa.Column(checks=_is_datetime, nullable=True),
        "GENDER": pa.Column(checks=pa.Check.isin(["Male", "Female"]), nullable=True),
        "WORK_CITY": pa.Column(checks=no_placeholders, nullable=True),
        "WORK_PROVINCE": pa.Column(checks=no_placeholders, nullable=True),
        "WORK_COUNTRY": pa.Column(checks=pa.Check.str_matches(r"^[A-Z]{2}$"), nullable=True),
    }
    return pa.DataFrameSchema(
        columns,
        checks=[
            pa.Check(lambda df: df["FFP_DATE"] <= df["LOAD_TIME"], name="ffp_before_load"),
            pa.Check(
                lambda df: (df["SUM_YR_1"] + df["SUM_YR_2"] > 0) | (df["SEG_KM_SUM"] == 0),
                name="revenue_if_flown",
            ),
        ],
        name="CleanFlightSchema",
    )
