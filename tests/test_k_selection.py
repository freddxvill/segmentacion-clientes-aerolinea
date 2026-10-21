import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from segmentacion.models.k_selection import (
    cluster_medians,
    evaluate_k_range,
    find_elbow,
    fit_kmeans,
    silhouette_curve,
    silhouette_profile,
    stratified_sample,
)


@pytest.fixture(scope="module")
def blobs() -> pd.DataFrame:
    """Cuatro grupos bien separados en 5 dimensiones (como las features LRFMC)."""
    x, _ = make_blobs(n_samples=2_000, centers=4, n_features=5, cluster_std=1.0, random_state=7)
    return pd.DataFrame(x, columns=list("LRFMC"))


@pytest.fixture(scope="module")
def blob_metrics(blobs) -> pd.DataFrame:
    return evaluate_k_range(blobs, range(2, 11), sample_size=1_000, random_state=0)


def test_find_elbow_on_blobs_returns_true_k(blob_metrics):
    assert find_elbow(blob_metrics["k"], blob_metrics["inertia"]) == 4


def test_silhouette_max_on_blobs(blob_metrics):
    assert blob_metrics.loc[blob_metrics["silhouette"].idxmax(), "k"] == 4


def test_complementary_metrics_agree_on_blobs(blob_metrics):
    assert blob_metrics.loc[blob_metrics["ch"].idxmax(), "k"] == 4
    assert blob_metrics.loc[blob_metrics["db"].idxmin(), "k"] == 4


def test_metrics_dataframe_columns(blob_metrics):
    assert list(blob_metrics.columns) == [
        "k",
        "inertia",
        "silhouette",
        "negative_pct",
        "ch",
        "db",
        "min_cluster_pct",
    ]
    assert blob_metrics["k"].tolist() == list(range(2, 11))
    assert blob_metrics["inertia"].is_monotonic_decreasing


def test_find_elbow_hand_calculated():
    # La caída se frena en k = 3: es el punto más alejado de la recta entre extremos.
    assert find_elbow([1, 2, 3, 4, 5], [100, 40, 15, 12, 10]) == 3


def test_find_elbow_requires_three_points():
    with pytest.raises(ValueError, match="al menos 3"):
        find_elbow([2, 3], [10, 5])


def test_stratified_sample_keeps_proportions():
    labels = np.array([0] * 800 + [1] * 150 + [2] * 50)
    sample = stratified_sample(labels, 100, random_state=0)
    counts = np.bincount(labels[sample])
    assert counts.tolist() == [80, 15, 5]
    assert len(np.unique(sample)) == len(sample)


def test_stratified_sample_returns_all_when_size_exceeds_data():
    labels = np.array([0, 1, 1])
    assert stratified_sample(labels, 10).tolist() == [0, 1, 2]


def test_silhouette_curve_is_sorted_and_compact(blobs):
    labels = fit_kmeans(blobs, 4, random_state=0).labels_
    profile = silhouette_profile(blobs, labels, sample_size=800, random_state=0)
    curve = silhouette_curve(profile, points=50)
    assert set(curve["cluster"]) == {0, 1, 2, 3}
    assert curve.groupby("cluster").size().max() == 50
    for _, group in curve.groupby("cluster"):
        assert group["silhouette"].is_monotonic_decreasing
    assert curve["socios"].groupby(curve["cluster"]).first().sum() == len(profile)


def test_cluster_medians_hand_calculated():
    features = pd.DataFrame({"F": [1, 3, 10, 20, 30], "M": [100, 300, 1000, 2000, 3000]})
    table = cluster_medians(features, np.array([0, 0, 1, 1, 1]))
    assert table["F"].tolist() == [2, 20]
    assert table["M"].tolist() == [200, 2000]
    assert table["pct"].tolist() == pytest.approx([0.4, 0.6])
