"""Reglas explícitas centroide → nombre comercial del segmento (Fase 9).

Cada nombre es un **arquetipo** definido por los pesos que da a las features del centroide (en
z-score): por ejemplo, un VIP vuela mucho (F↑, M↑) y hace poco (R↓). La puntuación de un clúster
para un arquetipo es el producto escalar de su centroide con esos pesos. Los nombres se reparten
con una asignación óptima (algoritmo húngaro) que maximiza la puntuación total, así que dos
clústeres nunca reciben el mismo nombre aunque se parezcan.

Los arquetipos parten de las hipótesis del plan (§4.8) y se ajustaron tras ver los centroides:
«Dormidos / bajo valor» pasó a «Veteranos ocasionales», porque ese grupo vuela poco pero no está
inactivo (su recencia es la típica).
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from segmentacion.segments.profile import TREND


@dataclass(frozen=True)
class Archetype:
    name: str
    weights: dict[str, float]  # feature (L, R, F, M, C o T = tendencia) → peso
    rule: str  # la regla en palabras, para el sitio
    description: str  # quiénes son, en una frase (sin cifras: las cifras salen de los datos)


# En orden de valor para el negocio: así se ordenan los segmentos en tablas y gráficos.
ARCHETYPES: tuple[Archetype, ...] = (
    Archetype(
        "VIP",
        {"F": 1, "M": 1, "R": -1, "C": 0.5},
        "F↑ M↑ R↓ (C↑)",
        "Vuelan mucho, lo han hecho hace poco y pagan tarifas algo por encima de la media: "
        "concentran la mayor parte del ingreso.",
    ),
    Archetype(
        "Nuevos con potencial",
        {"L": -1, "R": -1, "F": 0.5},
        "L↓ R↓ (F↑)",
        "Socios recientes y activos, con más vuelos que la media: el semillero de los futuros VIP.",
    ),
    Archetype(
        "Veteranos ocasionales",
        {"L": 1, "F": -0.5, "M": -0.5},
        "L↑ (F↓ M↓)",
        "Socios antiguos que siguen volando, pero poco: una fidelidad de largo plazo con baja "
        "frecuencia.",
    ),
    Archetype(
        "En riesgo de abandono",
        {"R": 1, TREND: -1, "C": 0.5},
        "R↑ y tendencia de ingreso↓ (C↑)",
        "Llevan muchos meses sin volar, sus ingresos caen y pagaban tarifas casi completas: "
        "valor que se está perdiendo.",
    ),
    Archetype(
        "Cazadores de ofertas",
        {"C": -1, "F": -0.5},
        "C↓ (F↓)",
        "Compran con mucho descuento y vuelan poco; la mayoría lleva meses sin volar.",
    ),
)
SEGMENT_ORDER: tuple[str, ...] = tuple(a.name for a in ARCHETYPES)


def archetype_scores(
    centroids: pd.DataFrame, archetypes: Sequence[Archetype] = ARCHETYPES
) -> pd.DataFrame:
    """Puntuación de cada clúster (filas) para cada arquetipo (columnas).

    Las features que un arquetipo no menciona pesan 0; las que faltan en `centroids` también.
    `archetypes` permite probar pesos alternativos (análisis de sensibilidad).
    """
    weights = pd.DataFrame({a.name: a.weights for a in archetypes}).T
    weights = weights.reindex(columns=centroids.columns).fillna(0.0)
    return pd.DataFrame(
        centroids.to_numpy(dtype=float) @ weights.to_numpy().T,
        index=centroids.index,
        columns=[a.name for a in archetypes],
    )


def name_segments(
    centroids: pd.DataFrame, archetypes: Sequence[Archetype] = ARCHETYPES
) -> dict[int, str]:
    """Nombre único para cada clúster: asignación que maximiza la puntuación total."""
    if len(centroids) > len(archetypes):
        raise ValueError(
            f"{len(centroids)} clústeres y solo {len(archetypes)} arquetipos: hay que definir "
            "más reglas de nombre."
        )
    scores = archetype_scores(centroids, archetypes)
    rows, cols = linear_sum_assignment(-scores.to_numpy())
    return {int(scores.index[r]): scores.columns[c] for r, c in zip(rows, cols, strict=True)}


def ordered_segments(names: dict[int, str]) -> list[str]:
    """Los nombres asignados, en el orden de `SEGMENT_ORDER`."""
    return [name for name in SEGMENT_ORDER if name in names.values()]


def archetype(name: str) -> Archetype:
    return next(a for a in ARCHETYPES if a.name == name)


def assignment_margin(centroids: pd.DataFrame, names: dict[int, str]) -> pd.Series:
    """Ventaja del nombre asignado sobre el mejor nombre alternativo de cada clúster.

    Un margen grande indica que el nombre no depende de pequeños cambios en los pesos.
    """
    scores = archetype_scores(centroids)
    margins = {}
    for cluster, name in names.items():
        row = scores.loc[cluster]
        margins[cluster] = float(row[name] - row.drop(name).max())
    return pd.Series(margins, name="margin").rename_axis("cluster").sort_index()


def perturbed_archetypes(scale: float, rng: np.random.Generator) -> tuple[Archetype, ...]:
    """Arquetipos con cada peso multiplicado por un factor aleatorio en [1 − scale, 1 + scale]."""
    return tuple(
        replace(a, weights={f: w * rng.uniform(1 - scale, 1 + scale) for f, w in a.weights.items()})
        for a in ARCHETYPES
    )


def perturbation_agreement(
    centroids: pd.DataFrame, runs: int = 500, scale: float = 0.3, random_state: int = 42
) -> float:
    """% de perturbaciones de los pesos (±`scale`) que dejan la misma asignación de nombres."""
    rng = np.random.default_rng(random_state)
    names = name_segments(centroids)
    same = [
        name_segments(centroids, perturbed_archetypes(scale, rng)) == names for _ in range(runs)
    ]
    return float(np.mean(same))
