"""Selección del número de segmentos *k* (Fase 6).

Para cada *k* se entrena K-Means sobre todos los socios y se calculan inercia (Elbow),
Calinski-Harabasz y Davies-Bouldin con los datos completos. El silhouette es O(n²), así que se
calcula sobre una muestra estratificada por clúster.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
)

from segmentacion.models.kmeans import fit_kmeans


def stratified_sample(labels: np.ndarray, size: int, random_state: int = 42) -> np.ndarray:
    """Posiciones de una muestra que conserva la proporción de cada clúster."""
    labels = np.asarray(labels)
    if size >= len(labels):
        return np.arange(len(labels))
    rng = np.random.default_rng(random_state)
    picked = []
    for cluster in np.unique(labels):
        members = np.flatnonzero(labels == cluster)
        n = max(1, round(size * len(members) / len(labels)))
        picked.append(rng.choice(members, size=min(n, len(members)), replace=False))
    return np.sort(np.concatenate(picked))


def evaluate_k_range(
    features: pd.DataFrame,
    ks: Sequence[int] = range(2, 11),
    sample_size: int = 10_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Métricas de K-Means para cada *k*.

    Columnas: `[k, inertia, silhouette, negative_pct, ch, db, min_cluster_pct]`, donde
    `negative_pct` es la proporción de socios con silhouette < 0 (más cerca de otro clúster).
    """
    rows = []
    for k in ks:
        model = fit_kmeans(features, k, random_state)
        labels = model.labels_
        sample = stratified_sample(labels, sample_size, random_state)
        values = silhouette_samples(features.iloc[sample], labels[sample])
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": values.mean(),
                "negative_pct": (values < 0).mean(),
                "ch": calinski_harabasz_score(features, labels),
                "db": davies_bouldin_score(features, labels),
                "min_cluster_pct": np.bincount(labels).min() / len(labels),
            }
        )
    return pd.DataFrame(rows)


def find_elbow(ks: Sequence[int], inertias: Sequence[float]) -> int:
    """Codo de la curva: el punto más alejado de la recta entre el primer y el último *k*.

    Ambos ejes se normalizan a [0, 1] para que la distancia no dependa de sus escalas.
    """
    x = np.asarray(ks, dtype=float)
    y = np.asarray(inertias, dtype=float)
    if len(x) < 3:
        raise ValueError("Se necesitan al menos 3 valores de k para detectar un codo.")
    x_norm = (x - x[0]) / (x[-1] - x[0])
    y_norm = (y - y[-1]) / (y[0] - y[-1])
    # Recta de (0, 1) a (1, 0): x + y - 1 = 0 → distancia ∝ |x + y - 1|.
    distance = np.abs(x_norm + y_norm - 1) / np.sqrt(2)
    return int(x[np.argmax(distance)])


def silhouette_profile(
    features: pd.DataFrame,
    labels: np.ndarray,
    sample_size: int = 10_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Silhouette de cada socio de una muestra estratificada: `[cluster, silhouette]`."""
    labels = np.asarray(labels)
    sample = stratified_sample(labels, sample_size, random_state)
    values = silhouette_samples(features.iloc[sample], labels[sample])
    return pd.DataFrame({"cluster": labels[sample], "silhouette": values})


def silhouette_curve(profile: pd.DataFrame, points: int = 100) -> pd.DataFrame:
    """Reduce un silhouette plot a `points` cuantiles por clúster (para gráficos ligeros).

    Devuelve `[cluster, posicion, silhouette, socios, media]`: `posicion` va de 0 a 1 dentro de
    cada clúster, ordenada de mayor a menor silhouette.
    """
    rows = []
    for cluster, group in profile.groupby("cluster"):
        q = np.linspace(0, 1, min(points, len(group)))
        values = np.quantile(group["silhouette"], 1 - q)
        rows.append(
            pd.DataFrame(
                {
                    "cluster": cluster,
                    "posicion": q,
                    "silhouette": values,
                    "socios": len(group),
                    "media": group["silhouette"].mean(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def cluster_medians(features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Mediana de cada feature (en unidades originales) y % de socios por clúster."""
    labels = np.asarray(labels)
    table = features.groupby(labels).median()
    table["pct"] = pd.Series(labels).value_counts(normalize=True).sort_index().to_numpy()
    return table.rename_axis("cluster").reset_index()
