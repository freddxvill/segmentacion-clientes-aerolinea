"""Modelo principal: K-Means con el *k* elegido y su estabilidad (Fase 7).

La estabilidad es el criterio principal de calidad del proyecto: si cambiar la semilla (o los
socios con los que se entrena) cambia los segmentos, no se pueden usar para decidir acciones.
Se mide con el ARI (*Adjusted Rand Index*: 1 = mismas particiones, 0 = coincidencia al azar).
"""

from collections.abc import Sequence
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def fit_kmeans(features: pd.DataFrame, k: int, random_state: int = 42) -> KMeans:
    return KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(features)


def stability_ari(features: pd.DataFrame, k: int, seeds: Sequence[int]) -> pd.DataFrame:
    """ARI entre cada par de modelos entrenados con semillas distintas.

    Devuelve `[seed_a, seed_b, ari]` con una fila por par (10 semillas → 45 pares).
    """
    labels = {seed: fit_kmeans(features, k, seed).labels_ for seed in seeds}
    rows = [
        {"seed_a": a, "seed_b": b, "ari": adjusted_rand_score(labels[a], labels[b])}
        for a, b in combinations(seeds, 2)
    ]
    return pd.DataFrame(rows, columns=["seed_a", "seed_b", "ari"])


def subsample_stability(
    features: pd.DataFrame,
    k: int,
    runs: int = 10,
    fraction: float = 0.8,
    random_state: int = 42,
) -> pd.DataFrame:
    """ARI frente al modelo completo de modelos entrenados con un `fraction` de los socios.

    Cada modelo asigna a **todos** los socios (también a los que no vio), así que el ARI mide si
    los segmentos dependen de qué socios se usaron para entrenar. Devuelve `[run, ari]`.
    """
    reference = fit_kmeans(features, k, random_state).labels_
    rng = np.random.default_rng(random_state)
    size = round(fraction * len(features))
    rows = []
    for run in range(runs):
        idx = rng.choice(len(features), size=size, replace=False)
        model = fit_kmeans(features.iloc[idx], k, random_state + run + 1)
        rows.append({"run": run, "ari": adjusted_rand_score(reference, model.predict(features))})
    return pd.DataFrame(rows, columns=["run", "ari"])


def save_model(model: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> object:
    return joblib.load(path)
