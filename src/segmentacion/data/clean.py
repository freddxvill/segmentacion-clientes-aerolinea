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

# Sufijos administrativos en pinyin (省 = provincia, 市 = ciudad/municipio) que duplican categorías:
# "guangdongsheng" y "guangdong" son la misma provincia.
ADMIN_SUFFIXES: dict[str, tuple[str, ...]] = {
    "WORK_PROVINCE": ("sheng", "shi"),
    "WORK_CITY": ("shi",),
}


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

    # 1. Texto: ancho completo → ASCII (NFKC), espacios, mayúsculas/minúsculas, sufijos
    # administrativos y placeholders ("." "-" ...) → nulo.
    for column in TEXT_COLUMNS:
        text = out[column].astype("str").str.normalize("NFKC").str.strip()
        out[column] = text.mask(out[column].isna())
    for column, suffixes in ADMIN_SUFFIXES.items():
        out[column] = merge_suffix_variants(out[column].str.lower(), suffixes)
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
    # FIRST_FLIGHT_DATE solo se usa en el EDA: los valores imposibles (1905, posteriores al corte)
    # se anulan sin descartar al socio.
    first_flight = out["FIRST_FLIGHT_DATE"]
    set_null(
        "FIRST_FLIGHT_DATE",
        (first_flight < out["FFP_DATE"]) | (first_flight > out["LOAD_TIME"]),
        "primer_vuelo_fuera_de_ventana",
        "FIRST_FLIGHT_DATE anterior al alta o posterior al corte → nulo",
    )
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


def merge_suffix_variants(values: pd.Series, suffixes: tuple[str, ...]) -> pd.Series:
    """Quita un sufijo solo si la forma sin él ya existe como categoría.

    Así "guangzhoushi" → "guangzhou", pero "huangshi" (una ciudad real) se conserva.
    """
    known = set(values.dropna())

    def base(value: str) -> str:
        for suffix in suffixes:
            stem = value.removesuffix(suffix)
            if stem != value and stem in known:
                return stem
        return value

    return values.map(base, na_action="ignore")


def lineage_report(n_input: int, n_output: int, lineage: list[LineageStep]) -> dict:
    """Resumen serializable del lineage (se muestra en la página Datos del sitio)."""
    return {
        "input_rows": n_input,
        "output_rows": n_output,
        "dropped_rows": sum(s.rows for s in lineage if s.action == "drop"),
        "steps": [asdict(s) for s in lineage],
    }


def null_summary(raw: pd.DataFrame, cleaned: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Nulos por columna antes y después de limpiar (los placeholders pasan a contar como nulos)."""
    return {
        "raw": {c: int(n) for c, n in raw.isna().sum().items()},
        "clean": {c: int(n) for c, n in cleaned.isna().sum().items()},
    }


def save_clean(df: pd.DataFrame, report: dict, clean_file: Path, lineage_file: Path) -> None:
    clean_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(clean_file, index=False)
    lineage_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def load_clean(clean_file: Path) -> pd.DataFrame:
    return pd.read_parquet(clean_file)
