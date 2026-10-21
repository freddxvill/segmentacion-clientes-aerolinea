"""Lectura del CSV crudo `flight.csv`."""

import csv
from pathlib import Path

import pandas as pd

# Diccionario esperado (ver docs/data-card.md). Se confirma contra el archivo real en la Fase 1.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "MEMBER_NO",
    "FFP_DATE",
    "FIRST_FLIGHT_DATE",
    "GENDER",
    "FFP_TIER",
    "WORK_CITY",
    "WORK_PROVINCE",
    "WORK_COUNTRY",
    "AGE",
    "LOAD_TIME",
    "FLIGHT_COUNT",
    "SUM_YR_1",
    "SUM_YR_2",
    "SEG_KM_SUM",
    "LAST_FLIGHT_DATE",
    "LAST_TO_END",
    "AVG_INTERVAL",
    "MAX_INTERVAL",
    "avg_discount",
    "BP_SUM",
    "EXCHANGE_COUNT",
    "Points_Sum",
    "Point_NotFlight",
)

# Las ciudades/provincias pueden venir en caracteres chinos; se prueba UTF-8 y luego GB18030.
ENCODINGS: tuple[str, ...] = ("utf-8-sig", "gb18030")


def detect_encoding(path: Path) -> str:
    """Devuelve la primera codificación de `ENCODINGS` que decodifica el archivo completo."""
    data = path.read_bytes()
    for encoding in ENCODINGS:
        try:
            data.decode(encoding)
        except UnicodeDecodeError:
            continue
        return encoding
    raise ValueError(f"No se pudo decodificar {path} con ninguna de: {', '.join(ENCODINGS)}")


def read_header(path: Path) -> list[str]:
    """Lee solo la fila de encabezados del CSV."""
    with path.open(encoding=detect_encoding(path), newline="") as f:
        return next(csv.reader(f))


def load_raw(path: Path) -> pd.DataFrame:
    """Carga el CSV crudo completo, sin transformar tipos de fecha."""
    return pd.read_csv(path, encoding=detect_encoding(path))
