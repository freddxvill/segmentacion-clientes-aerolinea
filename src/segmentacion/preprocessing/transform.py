"""Transformación de asimetría y escalado de las features LRFMC (Fase 5).

El preprocesamiento es un `Pipeline` de scikit-learn (transformación → escalado) que se
serializa con joblib, para aplicar exactamente lo mismo al entrenar y al asignar segmentos.
"""

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

Transform = Literal["none", "log1p", "yeo-johnson"]
Scaler = Literal["standard", "robust"]
TRANSFORMS: tuple[Transform, ...] = ("none", "log1p", "yeo-johnson")
SCALERS: tuple[Scaler, ...] = ("standard", "robust")


def _transform_step(transform: Transform):
    if transform == "none":
        return "passthrough"
    if transform == "log1p":
        return FunctionTransformer(np.log1p, inverse_func=np.expm1, feature_names_out="one-to-one")
    if transform == "yeo-johnson":
        # Sin estandarizar aquí: el escalado es un paso aparte para poder compararlo.
        return PowerTransformer(method="yeo-johnson", standardize=False)
    raise ValueError(f"Transformación desconocida: {transform!r}")


def _scaler_step(scaler: Scaler):
    if scaler == "standard":
        return StandardScaler()
    if scaler == "robust":
        return RobustScaler()
    raise ValueError(f"Escalador desconocido: {scaler!r}")


def make_preprocessor(
    transform: Transform = "yeo-johnson", scaler: Scaler = "standard"
) -> Pipeline:
    """Pipeline `transformación → escalado` que devuelve DataFrames con las mismas columnas."""
    pipeline = Pipeline(
        [("transform", _transform_step(transform)), ("scale", _scaler_step(scaler))]
    )
    return pipeline.set_output(transform="pandas")


def skewness(data: pd.DataFrame) -> pd.Series:
    """Asimetría (Fisher-Pearson, sesgada) de cada columna."""
    return pd.Series(stats.skew(data.to_numpy(dtype=float), axis=0), index=data.columns)


def compare_preprocessors(
    features: pd.DataFrame,
    ks: tuple[int, ...] = (4, 5, 6),
    sample_size: int = 10_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compara cada combinación transformación × escalado.

    Por combinación: asimetría de cada feature tras transformar, cuántas quedan con |skew| < 1,
    el desequilibrio de pesos entre features (`std_ratio`) y, para cada *k* fijo, el silhouette
    de K-Means (sobre una muestra, por su costo O(n²)) y el tamaño del clúster más pequeño.
    """
    rows = []
    for transform in TRANSFORMS:
        for scaler in SCALERS:
            scaled = make_preprocessor(transform, scaler).fit_transform(features)
            skew = skewness(scaled)
            row: dict[str, object] = {"transform": transform, "scaler": scaler}
            row.update({f"skew_{c}": s for c, s in skew.items()})
            row["n_abs_skew_lt_1"] = int((skew.abs() < 1).sum())
            row["max_abs_skew"] = float(skew.abs().max())
            # Con desviaciones distintas, la feature más dispersa pesa más en la distancia de
            # K-Means aunque no sea la más relevante.
            spread = scaled.std(ddof=0)
            row["std_ratio"] = float(spread.max() / spread.min())
            for k in ks:
                labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(
                    scaled
                )
                row[f"min_cluster_pct_k{k}"] = float(np.bincount(labels).min() / len(labels))
                row[f"silhouette_k{k}"] = silhouette_score(
                    scaled,
                    labels,
                    sample_size=min(sample_size, len(scaled)),
                    random_state=random_state,
                )
            row["silhouette_mean"] = float(np.mean([row[f"silhouette_k{k}"] for k in ks]))
            rows.append(row)
    return pd.DataFrame(rows)


def save_preprocessor(preprocessor: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path: Path) -> Pipeline:
    return joblib.load(path)
