import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from segmentacion.models.k_selection import stratified_sample
from segmentacion.projection.reduce import (
    neighborhood_preservation,
    pca_project,
    tsne_project,
)


@pytest.fixture(scope="module")
def blobs() -> tuple[pd.DataFrame, np.ndarray]:
    x, y = make_blobs(n_samples=600, centers=3, n_features=5, random_state=1)
    return pd.DataFrame(x, columns=list("LRFMC")), y


def test_pca_reports_explained_variance(blobs):
    features, _ = blobs
    result = pca_project(features, n=2)
    assert result.explained["component"].tolist() == ["PC1", "PC2", "PC3", "PC4", "PC5"]
    assert result.explained["variance"].is_monotonic_decreasing
    assert result.explained["cumulative"].iloc[-1] == pytest.approx(1)
    assert list(result.scores.columns) == ["PC1", "PC2"]
    assert result.scores.index.equals(features.index)
    assert result.loadings.shape == (2, 5)
    # Cada componente es un vector unitario en el espacio de las features.
    assert np.linalg.norm(result.loadings, axis=1) == pytest.approx([1, 1])


def test_pca_3d_explains_more_than_2d(blobs):
    features, _ = blobs
    two, three = pca_project(features, 2), pca_project(features, 3)
    assert list(three.scores.columns) == ["PC1", "PC2", "PC3"]
    np.testing.assert_allclose(three.scores[["PC1", "PC2"]], two.scores)


def test_pca_rejects_invalid_n(blobs):
    features, _ = blobs
    with pytest.raises(ValueError, match="entre 1 y 5"):
        pca_project(features, n=6)


def test_tsne_reproducible_with_fixed_seed(blobs):
    features, labels = blobs
    sample = stratified_sample(labels, 200, random_state=0)
    a = tsne_project(features, sample, perplexity=20, random_state=7)
    b = tsne_project(features, sample, perplexity=20, random_state=7)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == ["position", "TSNE1", "TSNE2"]
    assert a["position"].tolist() == sample.tolist()
    assert a[["TSNE1", "TSNE2"]].dtypes.eq(float).all()


def test_tsne_sample_is_stratified_by_cluster(blobs):
    _, labels = blobs
    labels = np.where(np.arange(len(labels)) < 60, 9, labels)  # un clúster pequeño (10 %)
    sample = stratified_sample(labels, 100, random_state=0)
    share = pd.Series(labels[sample]).value_counts(normalize=True)
    expected = pd.Series(labels).value_counts(normalize=True)
    assert (share - expected).abs().max() < 0.02


def test_tsne_handles_tiny_samples(blobs):
    features, _ = blobs
    # perplexity se ajusta al tamaño de la muestra (t-SNE exige perplexity < n).
    result = tsne_project(features, np.arange(20), perplexity=30, random_state=0)
    assert len(result) == 20


def test_neighborhood_preservation_separated_blobs(blobs):
    features, labels = blobs
    sample = stratified_sample(labels, 200, random_state=0)
    embedding = tsne_project(features, sample, perplexity=20, random_state=0)
    score = neighborhood_preservation(features.iloc[sample], embedding[["TSNE1", "TSNE2"]])
    assert 0.8 < score <= 1
