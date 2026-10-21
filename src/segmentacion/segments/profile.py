"""Perfil de cada segmento y pruebas de que difieren en valor real (Fase 9).

- **Centroides en z-score:** los centros de K-Means ya están en el espacio transformado y
  estandarizado (Yeo-Johnson + StandardScaler), así que 0 = socio medio y ±1 = una desviación.
- **KPIs:** tamaño, ingreso, tendencia, nivel del programa y perfil demográfico, en unidades
  originales (variables de perfilado que no entraron al modelo).
- **Validación externa:** V de Cramér (segmento × variable categórica) y Kruskal-Wallis con su
  tamaño de efecto ε² (segmento × variable numérica). Con 62.000 socios casi cualquier diferencia
  es «significativa», por eso se informa siempre el tamaño del efecto.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

TREND = "T"  # columna de la tendencia de ingresos (z-score) junto a las LRFMC

# Umbrales convencionales de magnitud (Cohen para V con 2 grados de libertad mínimos; ε² igual
# que η²). Se aplican como «al menos este valor».
CRAMERS_V_THRESHOLDS = {"pequeño": 0.07, "mediano": 0.21, "grande": 0.35}
EPSILON2_THRESHOLDS = {"pequeño": 0.01, "mediano": 0.06, "grande": 0.14}


def centroids_z(centers: np.ndarray, columns: Sequence[str]) -> pd.DataFrame:
    """Centros de K-Means (ya en z-score) como tabla `cluster × feature`."""
    return pd.DataFrame(centers, columns=list(columns)).rename_axis("cluster")


def trend_z(labels: np.ndarray, revenue_trend: pd.Series) -> pd.Series:
    """Tendencia de ingresos media de cada clúster, estandarizada como las features."""
    values = pd.Series(np.asarray(revenue_trend, dtype=float))
    means = values.groupby(np.asarray(labels)).mean()
    return ((means - values.mean()) / values.std(ddof=0)).rename(TREND).rename_axis("cluster")


def segment_kpis(labels: np.ndarray, lrfmc: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    """KPIs de negocio por clúster (filas alineadas con `labels`).

    - `members`, `pct`: tamaño.
    - `revenue_median`, `revenue_share`: ingreso típico y % del ingreso total que aporta.
    - `trend_median`, `falling_pct`: tendencia año 2 vs año 1 y % de socios con caída fuerte
      (tendencia < −0,2: el año reciente ingresa menos de dos tercios que el anterior).
    - Medianas de L, R, F, M, C y de km por vuelo; % de socios en niveles altos del programa
      (`FFP_TIER` > 4), edad mediana, % de mujeres y canjes medios.
    """
    labels = np.asarray(labels)
    data = pd.concat([lrfmc.reset_index(drop=True), profile.reset_index(drop=True)], axis=1).assign(
        cluster=labels
    )
    groups = data.groupby("cluster")
    known_gender = data["GENDER"].notna()
    table = pd.DataFrame(
        {
            "members": groups.size(),
            "pct": groups.size() / len(data),
            "revenue_median": groups["revenue_total"].median(),
            "revenue_share": groups["revenue_total"].sum() / data["revenue_total"].sum(),
            "trend_median": groups["revenue_trend"].median(),
            "falling_pct": (data["revenue_trend"] < -0.2).groupby(labels).mean(),
            **{f"{c}_median": groups[c].median() for c in ("L", "R", "F", "M", "C")},
            "km_per_flight_median": groups["km_per_flight"].median(),
            "elite_pct": (data["FFP_TIER"] > 4).groupby(labels).mean(),
            "age_median": groups["AGE"].median(),
            "female_pct": (data.loc[known_gender, "GENDER"] == "Female")
            .groupby(labels[known_gender.to_numpy()])
            .mean(),
            "exchanges_mean": groups["EXCHANGE_COUNT"].mean(),
        }
    )
    return table.rename_axis("cluster")


def magnitude(value: float, thresholds: dict[str, float]) -> str:
    """Etiqueta de magnitud del efecto: «despreciable», «pequeño», «mediano» o «grande»."""
    label = "despreciable"
    for name, threshold in thresholds.items():
        if value >= threshold:
            label = name
    return label


def cramers_v(labels: np.ndarray, other: pd.Series) -> dict[str, float]:
    """Asociación entre los segmentos y una variable categórica (se ignoran los nulos).

    V = √(χ² / (n · (min(filas, columnas) − 1))): 0 = independencia, 1 = asociación perfecta.
    """
    mask = pd.Series(other).notna().to_numpy()
    table = pd.crosstab(np.asarray(labels)[mask], pd.Series(other).to_numpy()[mask])
    # Sin corrección de Yates: en tablas 2×2 rebajaría χ² y V no llegaría a 1.
    chi2, p_value, dof, _ = stats.chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
    return {"statistic": float(chi2), "p_value": float(p_value), "dof": int(dof), "effect": v}


def kruskal_epsilon(labels: np.ndarray, values: pd.Series) -> dict[str, float]:
    """Kruskal-Wallis entre segmentos y su tamaño de efecto ε² = H / (n − 1).

    ε² es la proporción de la variabilidad de los rangos explicada por el segmento (0–1).
    """
    values = pd.Series(np.asarray(values, dtype=float))
    mask = values.notna().to_numpy()
    groups = [g.to_numpy() for _, g in values[mask].groupby(np.asarray(labels)[mask])]
    h, p_value = stats.kruskal(*groups)
    n = int(mask.sum())
    return {
        "statistic": float(h),
        "p_value": float(p_value),
        "dof": len(groups) - 1,
        "effect": float(h / (n - 1)),
    }


def association_tests(labels: np.ndarray, profile: pd.DataFrame) -> pd.DataFrame:
    """Validación externa de los segmentos con variables que no entraron al modelo.

    Columnas: `[variable, test, statistic, dof, p_value, effect_name, effect, magnitude]`.
    """
    rows = []
    for column in ("FFP_TIER", "GENDER"):
        result = cramers_v(labels, profile[column])
        rows.append(
            {
                "variable": column,
                "test": "Chi-cuadrado",
                **result,
                "effect_name": "V de Cramér",
                "magnitude": magnitude(result["effect"], CRAMERS_V_THRESHOLDS),
            }
        )
    for column in ("revenue_total", "revenue_trend", "AGE"):
        result = kruskal_epsilon(labels, profile[column])
        rows.append(
            {
                "variable": column,
                "test": "Kruskal-Wallis",
                **result,
                "effect_name": "ε²",
                "magnitude": magnitude(result["effect"], EPSILON2_THRESHOLDS),
            }
        )
    columns = [
        "variable",
        "test",
        "statistic",
        "dof",
        "p_value",
        "effect_name",
        "effect",
        "magnitude",
    ]
    return pd.DataFrame(rows)[columns]
