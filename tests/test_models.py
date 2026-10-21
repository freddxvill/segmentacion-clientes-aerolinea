import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from segmentacion.models.baseline import BASELINE_SEGMENTS, baseline_segments, quintile_scores
from segmentacion.models.benchmark import (
    NOISE,
    choose_eps,
    clustering_metrics,
    fit_agglomerative,
    fit_dbscan,
    fit_gmm,
    k_distance,
    k_distance_curve,
    run_benchmark,
)
from segmentacion.models.kmeans import (
    fit_kmeans,
    load_model,
    save_model,
    stability_ari,
    subsample_stability,
)


@pytest.fixture(scope="module")
def blobs() -> pd.DataFrame:
    """Cuatro grupos bien separados en 5 dimensiones (como las features LRFMC)."""
    x, _ = make_blobs(n_samples=1_000, centers=4, n_features=5, cluster_std=1.0, random_state=7)
    return pd.DataFrame(x, columns=list("LRFMC"))


@pytest.fixture(scope="module")
def blobs_with_outliers(blobs) -> pd.DataFrame:
    """Los blobs + 5 socios muy alejados de todo grupo."""
    outliers = pd.DataFrame(
        np.full((5, 5), 60.0) + np.arange(5)[:, None] * 10, columns=list("LRFMC")
    )
    return pd.concat([blobs, outliers], ignore_index=True)


def test_kmeans_reproducible_with_seed(blobs):
    a = fit_kmeans(blobs, 4, random_state=3).labels_
    b = fit_kmeans(blobs, 4, random_state=3).labels_
    np.testing.assert_array_equal(a, b)


def test_labels_length_matches_input(blobs):
    assert len(fit_kmeans(blobs, 4).labels_) == len(blobs)
    assert len(fit_agglomerative(blobs, 4, sample_size=300)) == len(blobs)


def test_stability_ari_high_on_blobs(blobs):
    pairs = stability_ari(blobs, 4, seeds=range(5))
    assert len(pairs) == 10  # 5 semillas → 10 pares
    assert list(pairs.columns) == ["seed_a", "seed_b", "ari"]
    assert pairs["ari"].mean() >= 0.9


def test_stability_ari_detects_wrong_k(blobs):
    # Con k = 7 sobre 4 grupos reales, los grupos sobrantes se parten al azar según la semilla.
    assert stability_ari(blobs, 7, seeds=range(5))["ari"].mean() < 0.9


def test_subsample_stability_high_on_blobs(blobs):
    runs = subsample_stability(blobs, 4, runs=4, fraction=0.7, random_state=0)
    assert runs["run"].tolist() == [0, 1, 2, 3]
    assert runs["ari"].min() >= 0.9


def test_model_roundtrip_joblib(blobs, tmp_path):
    model = fit_kmeans(blobs, 4)
    path = tmp_path / "models" / "kmeans.joblib"
    save_model(model, path)
    np.testing.assert_array_equal(load_model(path).predict(blobs), model.labels_)


def test_baseline_scores_between_1_and_5():
    rng = np.random.default_rng(0)
    lrfmc = pd.DataFrame(rng.lognormal(size=(500, 5)), columns=list("LRFMC"))
    scores = quintile_scores(lrfmc)
    assert scores.isin(range(1, 6)).all().all()
    # Sin empates, cada quintil tiene exactamente el 20 % de los socios.
    assert (scores.apply(pd.Series.value_counts) == 100).all().all()


def test_baseline_recency_is_inverted():
    lrfmc = pd.DataFrame({c: np.arange(1, 11, dtype=float) for c in "LRFMC"})
    scores = quintile_scores(lrfmc)
    # Más meses sin volar = peor: el socio con menor R recibe la puntuación más alta.
    assert scores["R"].iloc[0] == 5 and scores["R"].iloc[-1] == 1
    assert scores["F"].iloc[0] == 1 and scores["F"].iloc[-1] == 5


def test_baseline_ties_get_same_score():
    lrfmc = pd.DataFrame({c: [2.0] * 6 + [1.0, 3.0, 4.0, 5.0] for c in "LRFMC"})
    scores = quintile_scores(lrfmc)
    assert scores["F"].iloc[:6].nunique() == 1


def test_baseline_segment_rules_hand_calculated():
    scores = pd.DataFrame(
        {
            "L": [3, 3, 3, 3, 3],
            "R": [5, 2, 1, 1, 3],
            "F": [5, 4, 3, 1, 3],
            "M": [4, 5, 2, 2, 2],
            "C": [3, 3, 3, 3, 3],
        }
    )
    # FM = 4,5→5 · 4,5→5 · 2,5→3 · 1,5→2 · 2,5→3
    expected = ["Campeones", "Leales", "En riesgo", "Hibernando", "Potenciales"]
    assert baseline_segments(scores).tolist() == expected
    assert set(expected) == set(BASELINE_SEGMENTS)


def test_gmm_selects_lowest_bic(blobs):
    model, bic = fit_gmm(blobs, 4, n_init=1, random_state=0)
    assert set(bic["covariance_type"]) == {"full", "diag"}
    assert model.covariance_type == bic.loc[bic["bic"].idxmin(), "covariance_type"]


def test_agglomerative_recovers_blobs(blobs):
    reference = fit_kmeans(blobs, 4).labels_
    labels = fit_agglomerative(blobs, 4, sample_size=300, random_state=0)
    metrics = clustering_metrics(blobs, labels, reference, np.arange(len(blobs)))
    assert metrics["ari_kmeans"] >= 0.95


def test_dbscan_marks_outliers_as_minus_one(blobs_with_outliers):
    curve = k_distance_curve(k_distance(blobs_with_outliers, min_samples=10))
    eps = choose_eps(curve)
    labels = fit_dbscan(blobs_with_outliers, eps, min_samples=10)
    assert (labels[-5:] == NOISE).all()
    assert (labels[:-5] != NOISE).mean() > 0.9


def test_k_distance_is_sorted(blobs):
    distances = k_distance(blobs, min_samples=5)
    assert len(distances) == len(blobs)
    assert np.all(np.diff(distances) >= 0)


def test_clustering_metrics_ignore_noise_except_in_ari():
    features = pd.DataFrame({"x": [0.0, 0.1, 5.0, 5.1, 50.0]})
    labels = np.array([0, 0, 1, 1, NOISE])
    metrics = clustering_metrics(features, labels, labels, np.arange(5))
    assert metrics["n_clusters"] == 2
    assert metrics["noise_pct"] == pytest.approx(0.2)
    assert metrics["min_cluster_pct"] == pytest.approx(0.4)
    assert metrics["ari_kmeans"] == pytest.approx(1.0)
    assert metrics["silhouette"] > 0.9


def test_clustering_metrics_single_cluster_is_nan():
    features = pd.DataFrame({"x": [0.0, 0.1, 0.2, 9.0]})
    metrics = clustering_metrics(features, np.array([0, 0, 0, NOISE]), np.zeros(4), np.arange(4))
    assert metrics["n_clusters"] == 1
    assert np.isnan(metrics["silhouette"]) and np.isnan(metrics["db"])


def test_run_benchmark_table_contract(blobs):
    kmeans = fit_kmeans(blobs, 4).labels_
    baseline = np.where(blobs["L"] > blobs["L"].median(), "Alto", "Bajo")
    result = run_benchmark(
        blobs,
        kmeans,
        baseline,
        4,
        kmeans_seconds=0.1,
        gmm_n_init=1,
        agglomerative_sample_size=300,
        silhouette_sample_size=500,
        random_state=0,
    )
    assert result.table["model"].tolist() == [
        "K-Means",
        "Quintiles LRFMC",
        "GMM",
        "Agglomerative (Ward)",
        "DBSCAN",
    ]
    assert list(result.table.columns) == [
        "model",
        "n_clusters",
        "noise_pct",
        "silhouette",
        "db",
        "min_cluster_pct",
        "max_cluster_pct",
        "ari_kmeans",
        "seconds",
    ]
    by_model = result.table.set_index("model")
    assert by_model.loc["K-Means", "ari_kmeans"] == pytest.approx(1.0)
    assert by_model.loc["GMM", "ari_kmeans"] >= 0.9  # grupos esféricos: GMM coincide
    assert all(len(v) == len(blobs) for v in result.labels.values())
    assert result.eps > 0
