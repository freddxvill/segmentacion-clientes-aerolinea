"""Baseline «sin ML»: scoring LRFMC por quintiles y reglas de segmento (Fase 7).

Es la segmentación que haría un equipo de marketing con una hoja de cálculo: cada métrica se
puntúa de 1 a 5 según su quintil y los segmentos salen de reglas fijas sobre las puntuaciones,
al estilo de las matrices RFM clásicas. Sirve de referencia para ver qué aporta K-Means.
"""

import numpy as np
import pandas as pd

from segmentacion.features.lrfmc import LRFMC_COLUMNS

# Métricas donde un valor bajo es mejor: pocos meses sin volar = cliente reciente.
LOWER_IS_BETTER: tuple[str, ...] = ("R",)

# En el orden en que se evalúan las reglas (la primera que se cumple asigna el segmento).
BASELINE_SEGMENTS: tuple[str, ...] = (
    "Campeones",
    "Leales",
    "En riesgo",
    "Hibernando",
    "Potenciales",
)


def quintile_scores(lrfmc: pd.DataFrame) -> pd.DataFrame:
    """Puntuación 1–5 de cada métrica LRFMC según su quintil (5 = mejor).

    Se usa el rango percentil con empates promediados: los socios con el mismo valor (p. ej. el
    mismo número de vuelos) reciben siempre la misma puntuación, aunque los quintiles dejen de
    tener exactamente el 20 % de los socios.
    """
    scores = {}
    for column in LRFMC_COLUMNS:
        rank = lrfmc[column].rank(pct=True, method="average")
        score = np.ceil(rank * 5).clip(1, 5).astype(int)
        scores[column] = 6 - score if column in LOWER_IS_BETTER else score
    return pd.DataFrame(scores, index=lrfmc.index)


def baseline_segments(scores: pd.DataFrame) -> pd.Series:
    """Segmento de cada socio a partir de sus puntuaciones (reglas RFM clásicas).

    El valor (`FM`) es la media redondeada de las puntuaciones de frecuencia y kilómetros:

    1. **Campeones:** recientes (R ≥ 4) y de alto valor (FM ≥ 4).
    2. **Leales:** alto valor (FM ≥ 4) aunque no vuelen hace poco.
    3. **En riesgo:** valor medio o alto (FM ≥ 3) pero sin volar hace tiempo (R ≤ 2).
    4. **Hibernando:** sin volar hace tiempo y de bajo valor.
    5. **Potenciales:** el resto (actividad y valor intermedios).
    """
    value = np.floor((scores["F"] + scores["M"]) / 2 + 0.5)  # redondeo «comercial» de x,5
    recent, inactive = scores["R"] >= 4, scores["R"] <= 2
    conditions = [recent & (value >= 4), value >= 4, inactive & (value >= 3), inactive]
    labels = np.select(conditions, list(BASELINE_SEGMENTS[:-1]), default=BASELINE_SEGMENTS[-1])
    return pd.Series(labels, index=scores.index, name="baseline_segment")
