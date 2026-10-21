import json

import pandas as pd
import pytest

from segmentacion.pipeline import main


def test_pipeline_clean_step_writes_artifacts(tmp_config, flight_csv, capsys):
    assert main(["--step", "clean"], config=tmp_config) == 0
    assert tmp_config.clean_file.exists()
    report = json.loads(tmp_config.lineage_file.read_text(encoding="utf-8"))
    assert report["output_rows"] == 200
    assert set(report["null_counts"]) == {"raw", "clean"}
    assert "clean: 200 → 200 filas" in capsys.readouterr().out


def test_pipeline_fails_cleanly_without_raw_file(tmp_config, capsys):
    assert main(["--step", "clean"], config=tmp_config) == 1
    assert "poe download" in capsys.readouterr().err


def test_pipeline_all_writes_lrfmc_and_profile(tmp_config, flight_csv, capsys):
    assert main(["--step", "all"], config=tmp_config) == 0
    lrfmc = pd.read_parquet(tmp_config.lrfmc_file)
    profile = pd.read_parquet(tmp_config.profile_file)
    assert list(lrfmc.columns) == ["MEMBER_NO", "L", "R", "F", "M", "C"]
    assert len(lrfmc) == len(profile) == 200
    assert "features: 200 socios" in capsys.readouterr().out
    scaled = pd.read_parquet(tmp_config.scaled_file)
    assert scaled[["L", "R", "F", "M", "C"]].mean().abs().max() < 1e-9
    assert tmp_config.preprocessor_file.exists()
    assert len(pd.read_csv(tmp_config.transform_comparison_file)) == 6
    k_table = pd.read_csv(tmp_config.k_selection_file)
    assert k_table["k"].tolist() == list(range(2, 11))
    assert k_table["elbow"].sum() == 1
    assert set(pd.read_parquet(tmp_config.silhouette_curves_file)["k"]) == {3, 4, 5, 6, 7}
    assert len(pd.read_csv(tmp_config.k_preview_file)) == tmp_config.k_selection.chosen_k


def test_pipeline_end_to_end_on_fixture(tmp_config, flight_csv, capsys):
    assert main(["--step", "all"], config=tmp_config) == 0
    k = tmp_config.k_selection.chosen_k
    segments = pd.read_parquet(tmp_config.segments_file)
    assert len(segments) == 200
    assert segments["MEMBER_NO"].is_unique
    assert set(segments["cluster"]) == set(range(k))
    assert segments["gmm_proba"].between(0, 1).all()
    assert segments.filter(like="score_").isin(range(1, 6)).all().all()

    metrics = json.loads(tmp_config.metrics_file.read_text(encoding="utf-8"))
    assert metrics["k"] == k and metrics["members"] == 200
    assert sum(metrics["kmeans"]["cluster_pct"]) == pytest.approx(1)
    assert set(metrics["quality"]) == {"stability_ok", "silhouette_ok", "segment_size_ok"}
    assert -1 <= metrics["stability"]["seed_ari_min"] <= metrics["stability"]["seed_ari_mean"] <= 1

    stability = pd.read_csv(tmp_config.stability_file)
    seeds = tmp_config.modeling.stability_seeds
    assert (stability["method"] == "semillas").sum() == seeds * (seeds - 1) // 2
    benchmark = pd.read_csv(tmp_config.benchmark_file)
    assert len(benchmark) == 5
    assert tmp_config.kmeans_file.exists() and tmp_config.gmm_file.exists()
    assert "clustering: k = 5" in capsys.readouterr().out

    explained = pd.read_csv(tmp_config.pca_variance_file)
    assert explained["cumulative"].iloc[-1] == pytest.approx(1)
    assert len(pd.read_csv(tmp_config.pca_loadings_file)) == 3
    points = pd.read_parquet(tmp_config.pca_projection_file)
    assert {"MEMBER_NO", "cluster", "PC1", "PC2", "PC3"} <= set(points.columns)
    tsne = pd.read_parquet(tmp_config.tsne_file)
    assert sorted(tsne["perplexity"].unique()) == list(tmp_config.projection.tsne_perplexities)
    projection = json.loads(tmp_config.projection_metrics_file.read_text(encoding="utf-8"))
    assert all(0 <= v <= 1 for v in projection["trustworthiness"].values())


def test_features_step_requires_clean_file(tmp_config, capsys):
    assert main(["--step", "features"], config=tmp_config) == 1
    assert "poe clean" in capsys.readouterr().err
