"""Features LRFMC (entrada del clustering) y variables de perfilado (Fase 4).

L = meses entre FFP_DATE y LOAD_TIME   (antigüedad)
R = LAST_TO_END / 30                   (meses desde el último vuelo)
F = FLIGHT_COUNT                       (vuelos en la ventana de observación)
M = SEG_KM_SUM                         (km volados)
C = avg_discount                       (precio pagado / tarifa completa)
"""

import numpy as np
import pandas as pd
import pandera.pandas as pa

LRFMC_COLUMNS: tuple[str, ...] = ("L", "R", "F", "M", "C")
DAYS_PER_MONTH = 365.25 / 12  # mes medio para la antigüedad
DAYS_PER_RECENCY_MONTH = 30

# Columnas del dataset limpio que se conservan tal cual para describir los segmentos.
PROFILE_PASSTHROUGH: tuple[str, ...] = (
    "FFP_TIER",
    "GENDER",
    "AGE",
    "WORK_PROVINCE",
    "WORK_CITY",
    "SUM_YR_1",
    "SUM_YR_2",
    "AVG_INTERVAL",
    "MAX_INTERVAL",
    "EXCHANGE_COUNT",
    "Points_Sum",
)

_metric = pa.Column(float, checks=pa.Check.ge(0), nullable=False)


def lrfmc_schema() -> pa.DataFrameSchema:
    """Contrato de salida: un socio por fila, métricas finitas y no negativas."""
    finite = pa.Check(lambda df: np.isfinite(df[list(LRFMC_COLUMNS)]).all(axis=None), name="finite")
    return pa.DataFrameSchema(
        {"MEMBER_NO": pa.Column(unique=True), **{c: _metric for c in LRFMC_COLUMNS}},
        checks=finite,
        strict=True,
        coerce=True,
        name="LRFMCSchema",
    )


def build_lrfmc(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula `[MEMBER_NO, L, R, F, M, C]` a partir del dataset limpio (conserva el índice)."""
    out = pd.DataFrame(
        {
            "MEMBER_NO": df["MEMBER_NO"],
            "L": (df["LOAD_TIME"] - df["FFP_DATE"]).dt.days / DAYS_PER_MONTH,
            "R": df["LAST_TO_END"] / DAYS_PER_RECENCY_MONTH,
            "F": df["FLIGHT_COUNT"],
            "M": df["SEG_KM_SUM"],
            "C": df["avg_discount"],
        },
        index=df.index,
    )
    return lrfmc_schema().validate(out)


def revenue_trend(df: pd.DataFrame) -> pd.Series:
    """Balance de ingresos año 2 vs año 1, acotado a [-1, 1].

    -1 = solo ingresó en el año 1 (dejó de volar), +1 = solo en el año 2 (cliente nuevo o
    reactivado). Sin ingresos en ningún año se toma 0 (sin cambio).
    """
    total = df["SUM_YR_1"] + df["SUM_YR_2"]
    trend = (df["SUM_YR_2"] - df["SUM_YR_1"]) / total.where(total > 0)
    return trend.fillna(0.0)


def build_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variables de perfilado (no entran al modelo) + columnas descriptivas del socio."""
    flights = df["FLIGHT_COUNT"]
    points = df["Points_Sum"]
    derived = pd.DataFrame(
        {
            "MEMBER_NO": df["MEMBER_NO"],
            "revenue_total": df["SUM_YR_1"] + df["SUM_YR_2"],
            "revenue_trend": revenue_trend(df),
            "km_per_flight": (df["SEG_KM_SUM"] / flights.where(flights > 0)).fillna(0.0),
            # Sin puntos acumulados la proporción es 0; 4 socios tienen más puntos "no vuelo"
            # que puntos totales y se acotan a 1.
            "points_ratio": (df["Point_NotFlight"] / points.where(points > 0))
            .fillna(0.0)
            .clip(0, 1),
        },
        index=df.index,
    )
    return pd.concat([derived, df[list(PROFILE_PASSTHROUGH)]], axis=1)
