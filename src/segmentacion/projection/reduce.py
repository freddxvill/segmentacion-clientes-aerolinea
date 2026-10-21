"""Reducción de dimensionalidad para **visualizar** los segmentos (Fase 8).

Las proyecciones solo sirven para mirar los clústeres en 2D/3D: K-Means se entrenó sobre las
cinco features escaladas y aquí no se clusteriza nada.

- **PCA:** lineal; se ajusta con todos los socios y reporta la varianza explicada y los
  *loadings* (peso de cada feature en cada componente).
- **t-SNE:** no lineal, conserva vecindarios locales pero no distancias globales. Es O(n²) en
  memoria y lento, así que se calcula sobre una muestra estratificada por clúster.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness


@dataclass(frozen=True)
class PCAResult:
    scores: pd.DataFrame  # [PC1, …, PCn], mismo índice que las features
    explained: pd.DataFrame  # [component, variance, cumulative] de todas las componentes
    loadings: pd.DataFrame  # filas = componentes proyectadas, columnas = features


def pca_project(features: pd.DataFrame, n: int = 2) -> PCAResult:
    """Proyección PCA a `n` componentes.

    La varianza explicada se reporta para **todas** las componentes (no solo las `n` usadas),
    para ver cuánta información deja fuera la proyección.
    """
    if not 1 <= n <= features.shape[1]:
        raise ValueError(f"n debe estar entre 1 y {features.shape[1]}.")
    pca = PCA(random_state=0).fit(features)
    names = [f"PC{i}" for i in range(1, pca.n_components_ + 1)]
    ratio = pca.explained_variance_ratio_
    explained = pd.DataFrame(
        {"component": names, "variance": ratio, "cumulative": np.cumsum(ratio)}
    )
    loadings = pd.DataFrame(pca.components_[:n], index=names[:n], columns=features.columns)
    scores = pd.DataFrame(pca.transform(features)[:, :n], index=features.index, columns=names[:n])
    return PCAResult(scores=scores, explained=explained, loadings=loadings)


def tsne_project(
    features: pd.DataFrame,
    sample: Sequence[int],
    perplexity: float = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """t-SNE 2D de los socios en las posiciones `sample`: `[position, TSNE1, TSNE2]`.

    Con la misma semilla e inicialización PCA el resultado es reproducible.
    """
    sample = np.asarray(sample)
    x = features.iloc[sample].to_numpy(dtype=float)
    embedding = TSNE(
        n_components=2,
        perplexity=min(perplexity, (len(x) - 1) / 3),  # t-SNE exige perplexity < n
        init="pca",
        random_state=random_state,
    ).fit_transform(x)
    embedding = embedding.astype(float)  # TSNE devuelve float32
    return pd.DataFrame({"position": sample, "TSNE1": embedding[:, 0], "TSNE2": embedding[:, 1]})


def neighborhood_preservation(
    features: pd.DataFrame, embedding: pd.DataFrame, n_neighbors: int = 10
) -> float:
    """*Trustworthiness* (0–1): cuánto respeta la proyección a los vecinos originales.

    1 = los vecinos cercanos en 2D también lo son en el espacio LRFMC de 5 dimensiones.
    Es O(n²): pasar una muestra de unos pocos miles de socios.
    """
    return float(trustworthiness(features, embedding, n_neighbors=n_neighbors))
