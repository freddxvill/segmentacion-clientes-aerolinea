"""Limpieza del dataset crudo (Fase 2, reglas de plan-data.md §3.3).

Cada regla deja un `LineageStep` para mostrar en el sitio cuántas filas elimina o modifica.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from segmentacion.config import CleaningParams
from segmentacion.data.schema import DATE_COLUMNS, TEXT_COLUMNS, clean_schema, raw_schema

# LAST_FLIGHT_DATE no entra al modelo (R usa LAST_TO_END), así que una fecha inválida se anula
# en lugar de descartar al socio. El resto de fechas define L y su invalidez elimina la fila.
NULLABLE_DATE_COLUMNS: tuple[str, ...] = ("LAST_FLIGHT_DATE",)

Action = Literal["drop", "set_null"]


@dataclass(frozen=True)
class LineageStep:
    rule: str
    action: Action
    rows: int
    description: str


def clean(
    df: pd.DataFrame, params: CleaningParams | None = None
) -> tuple[pd.DataFrame, list[LineageStep]]:
    """Aplica las reglas de limpieza y devuelve `(df_limpio, lineage)`."""
    params = params or CleaningParams()
    raw_schema().validate(df)
    out = df.copy()
    lineage: list[LineageStep] = []

    def drop(mask: pd.Series, rule: str, description: str) -> None:
        nonlocal out
        lineage.append(LineageStep(rule, "drop", int(mask.sum()), description))
        out = out.loc[~mask]

    def set_null(column: str, mask: pd.Series, rule: str, description: str) -> None:
        lineage.append(LineageStep(rule, "set_null", int(mask.sum()), description))
        out[column] = out[column].mask(mask)

    # 1. Texto: espacios, mayúsculas/minúsculas y placeholders ("." "-" ...) → nulo.
    for column in TEXT_COLUMNS:
        out[column] = out[column].astype("str").str.strip().mask(out[column].isna())
    for column in ("WORK_CITY", "WORK_PROVINCE"):
        out[column] = out[column].str.lower()
    out["WORK_COUNTRY"] = out["WORK_COUNTRY"].str.upper()
    for column in TEXT_COLUMNS:
        is_placeholder = out[column].isin(params.text_placeholders) | (out[column] == "")
        set_null(column, is_placeholder, f"placeholder_{column}", f"{column} sin dato real → nulo")
    bad_country = out["WORK_COUNTRY"].notna() & ~out["WORK_COUNTRY"].str.fullmatch(r"[A-Z]{2}")
    set_null(
        "WORK_COUNTRY", bad_country, "pais_invalido", "WORK_COUNTRY no es un código ISO-2 → nulo"
    )

    # 2. Fechas (formato M/D/YYYY).
    invalid_dates = pd.Series(False, index=out.index)
    for column in DATE_COLUMNS:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            continue
        parsed = pd.to_datetime(out[column], format=params.date_format, errors="coerce")
        invalid = parsed.isna() & out[column].notna()
        out[column] = parsed
        if column in NULLABLE_DATE_COLUMNS:
            set_null(column, invalid, f"fecha_invalida_{column}", f"{column} inexistente → nulo")
        else:
            invalid_dates |= invalid | parsed.isna()
    drop(invalid_dates, "fecha_invalida", "FFP_DATE, FIRST_FLIGHT_DATE o LOAD_TIME inválida")
    drop(
        out["FFP_DATE"] > out["LOAD_TIME"],
        "alta_posterior_al_corte",
        "FFP_DATE posterior a LOAD_TIME",
    )

    # 3. Ingresos.
    drop(
        out["SUM_YR_1"].isna() | out["SUM_YR_2"].isna(),
        "ingreso_nulo",
        "SUM_YR_1 o SUM_YR_2 nulo",
    )
    drop(
        (out["SUM_YR_1"] + out["SUM_YR_2"] == 0) & (out["SEG_KM_SUM"] > 0),
        "ingreso_cero_con_vuelos",
        "Ingreso total 0 con km volados (vuelos de canje con puntos)",
    )

    # 4. Edad fuera de rango → nulo (no entra al clustering).
    set_null(
        "AGE",
        ~out["AGE"].between(params.age_min, params.age_max) & out["AGE"].notna(),
        "edad_fuera_de_rango",
        f"AGE fuera de [{params.age_min}, {params.age_max}] → nulo",
    )

    out = out.reset_index(drop=True)
    clean_schema(params).validate(out)
    return out, lineage


def lineage_report(n_input: int, n_output: int, lineage: list[LineageStep]) -> dict:
    """Resumen serializable del lineage (se muestra en la página Datos del sitio)."""
    return {
        "input_rows": n_input,
        "output_rows": n_output,
        "dropped_rows": sum(s.rows for s in lineage if s.action == "drop"),
        "steps": [asdict(s) for s in lineage],
    }


def save_clean(df: pd.DataFrame, report: dict, clean_file: Path, lineage_file: Path) -> None:
    clean_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(clean_file, index=False)
    lineage_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def load_clean(clean_file: Path) -> pd.DataFrame:
    return pd.read_parquet(clean_file)
