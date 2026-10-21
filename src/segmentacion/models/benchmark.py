"""Comparación ligera de K-Means con otros algoritmos (Fase 7).

- **GMM:** clústeres elípticos y probabilidad de pertenencia; el tipo de covarianza se elige por
  BIC.
- **Agglomerative (Ward):** jerárquico; su memoria es O(n²), así que se entrena con una muestra y
  el resto de los socios se asigna al centroide más cercano de cada grupo.
- **DBSCAN:** grupos por densidad; `eps` sale del codo del gráfico *k-distance* y los socios
  que no caen en ninguna zona densa quedan como ruido (etiqueta −1): los clientes atípicos.

Todas las métricas se calculan sobre las mismas features escaladas que usa K-Means y, para el
silhouette (O(n²)), sobre la misma muestra de socios para todos los modelos.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from segmentacion.models.k_selection import find_elbow

NOISE = -1


def fit_gmm(
    features: pd.DataFrame,
    k: int,
    covariance_types: Sequence[str] = ("full", "diag"),
    n_init: int = 3,
    random_state: int = 42,
) -> tuple[GaussianMixture, pd.DataFrame]:
    """GMM con `k` componentes y el tipo de covarianza de menor BIC.

    Devuelve el mejor modelo y la tabla `[covariance_type, bic]` de todos los probados.
    """
    models = {
        cov: GaussianMixture(
            n_components=k, covariance_type=cov, n_init=n_init, random_state=random_state
        ).fit(features)
        for cov in covariance_types
    }
    bic = pd.DataFrame(
        {"covariance_type": list(models), "bic": [m.bic(features) for m in models.values()]}
    )
    best = bic.loc[bic["bic"].idxmin(), "covariance_type"]
    return models[best], bic


def fit_agglomerative(
    features: pd.DataFrame, k: int, sample_size: int = 10_000, random_state: int = 42
) -> np.ndarray:
    """Ward sobre una muestra; el resto de los socios va al centroide más cercano."""
    x = features.to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    sample = rng.choice(len(x), size=min(sample_size, len(x)), replace=False)
    sample_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x[sample])
    centroids = np.vstack([x[sample][sample_labels == c].mean(axis=0) for c in range(k)])
    distances = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    labels = distances.argmin(axis=1)
    labels[sample] = sample_labels  # los socios de la muestra conservan su grupo de Ward
    return labels


def k_distance(features: pd.DataFrame, min_samples: int = 10) -> np.ndarray:
    """Distancia de cada socio a su `min_samples`-ésimo vecino (contándose a sí mismo), ordenada.

    Es la misma convención de DBSCAN: un punto es *core* si tiene `min_samples` puntos (él
    incluido) a distancia ≤ `eps`.
    """
    distances, _ = NearestNeighbors(n_neighbors=min_samples).fit(features).kneighbors(features)
    return np.sort(distances[:, -1])


def k_distance_curve(distances: np.ndarray, points: int = 201) -> pd.DataFrame:
    """Reduce la curva k-distance a `points` cuantiles: `[cuantil, distancia]`."""
    q = np.linspace(0, 1, points)
    return pd.DataFrame({"cuantil": q, "distancia": np.quantile(distances, q)})


def choose_eps(curve: pd.DataFrame) -> float:
    """`eps` de DBSCAN: el codo de la curva k-distance (donde la distancia se dispara)."""
    knee = find_elbow(range(len(curve)), curve["distancia"])
    return float(curve["distancia"].iloc[knee])


def fit_dbscan(features: pd.DataFrame, eps: float, min_samples: int = 10) -> np.ndarray:
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(features)


def clustering_metrics(
    features: pd.DataFrame,
    labels: np.ndarray,
    reference: np.ndarray,
    sample: np.ndarray,
) -> dict[str, float]:
    """Métricas comparables entre modelos.

    El ruido de DBSCAN (−1) cuenta en `noise_pct` y en el ARI (es una asignación más), pero no
    en el silhouette, Davies-Bouldin ni en los tamaños de clúster. Con menos de dos clústeres el
    silhouette y Davies-Bouldin no están definidos (NaN).
    """
    labels = np.asarray(labels)
    clustered = labels != NOISE
    sizes = np.unique(labels[clustered], return_counts=True)[1] / len(labels)
    n_clusters = len(sizes)
    in_sample = sample[clustered[sample]]
    silhouette = davies_bouldin = np.nan
    if n_clusters >= 2:
        silhouette = silhouette_score(features.iloc[in_sample], labels[in_sample])
        davies_bouldin = davies_bouldin_score(features[clustered], labels[clustered])
    return {
        "n_clusters": n_clusters,
        "noise_pct": float((~clustered).mean()),
        "silhouette": float(silhouette),
        "db": float(davies_bouldin),
        "min_cluster_pct": float(sizes.min()) if n_clusters else np.nan,
        "max_cluster_pct": float(sizes.max()) if n_clusters else np.nan,
        "ari_kmeans": float(adjusted_rand_score(reference, labels)),
    }


@dataclass(frozen=True)
class BenchmarkResult:
    table: pd.DataFrame  # una fila por modelo
    labels: dict[str, np.ndarray]  # etiquetas de cada modelo para todos los socios
    gmm: GaussianMixture
    gmm_bic: pd.DataFrame
    k_distance: pd.DataFrame
    eps: float


def run_benchmark(
    features: pd.DataFrame,
    kmeans_labels: np.ndarray,
    baseline_labels: np.ndarray,
    k: int,
    *,
    kmeans_seconds: float = np.nan,
    covariance_types: Sequence[str] = ("full", "diag"),
    gmm_n_init: int = 3,
    agglomerative_sample_size: int = 10_000,
    dbscan_min_samples: int = 10,
    silhouette_sample_size: int = 10_000,
    random_state: int = 42,
) -> BenchmarkResult:
    """Entrena los modelos alternativos y arma la tabla comparativa.

    Columnas de `table`: `[model, n_clusters, noise_pct, silhouette, db, min_cluster_pct,
    max_cluster_pct, ari_kmeans, seconds]`. Los segundos de K-Means se reciben de quien lo
    entrenó; el baseline no tiene costo de entrenamiento relevante.
    """
    timings: dict[str, float] = {"K-Means": kmeans_seconds, "Quintiles LRFMC": np.nan}
    labels: dict[str, np.ndarray] = {
        "K-Means": np.asarray(kmeans_labels),
        "Quintiles LRFMC": pd.factorize(np.asarray(baseline_labels), sort=True)[0],
    }

    def timed(name: str, fit: Callable[[], np.ndarray]) -> None:
        start = time.perf_counter()
        labels[name] = np.asarray(fit())
        timings[name] = time.perf_counter() - start

    fitted: dict[str, object] = {}

    def gmm_labels() -> np.ndarray:
        model, bic = fit_gmm(features, k, covariance_types, gmm_n_init, random_state)
        fitted.update(model=model, bic=bic)
        return model.predict(features)

    timed("GMM", gmm_labels)
    timed(
        "Agglomerative (Ward)",
        lambda: fit_agglomerative(features, k, agglomerative_sample_size, random_state),
    )

    def dbscan_labels() -> np.ndarray:
        # El tiempo incluye el cálculo de k-distance: es parte de ajustar el modelo.
        curve = k_distance_curve(k_distance(features, dbscan_min_samples))
        eps = choose_eps(curve)
        fitted.update(curve=curve, eps=eps)
        return fit_dbscan(features, eps, dbscan_min_samples)

    timed("DBSCAN", dbscan_labels)

    rng = np.random.default_rng(random_state)
    sample = np.sort(
        rng.choice(len(features), size=min(silhouette_sample_size, len(features)), replace=False)
    )
    rows = [
        {"model": name, **clustering_metrics(features, values, labels["K-Means"], sample)}
        | {"seconds": timings[name]}
        for name, values in labels.items()
    ]
    return BenchmarkResult(
        table=pd.DataFrame(rows),
        labels=labels,
        gmm=fitted["model"],
        gmm_bic=fitted["bic"],
        k_distance=fitted["curve"],
        eps=fitted["eps"],
    )
