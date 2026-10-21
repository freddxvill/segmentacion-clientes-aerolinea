from segmentacion.config import PROJECT_ROOT, get_config


def test_project_root_contains_pyproject():
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_paths_are_absolute_and_inside_project():
    config = get_config()
    assert config.paths.raw_dir.is_absolute()
    assert config.paths.raw_dir.is_relative_to(PROJECT_ROOT)


def test_data_params_match_spec():
    config = get_config()
    assert config.raw_file.name == "flight.csv"
    assert config.data.max_raw_size_mb == 20
    assert config.data.kaggle_competition == "sa-customer-segmentation"


def test_preprocessing_params_loaded_from_yaml():
    preprocessing = get_config().preprocessing
    assert preprocessing.transform in {"none", "log1p", "yeo-johnson"}
    assert preprocessing.scaler in {"standard", "robust"}
    assert preprocessing.compare_ks == (4, 5, 6)
    assert get_config().preprocessor_file.parent == get_config().paths.models_dir


def test_k_selection_params_loaded_from_yaml():
    params = get_config().k_selection
    assert list(params.ks) == list(range(2, 11))
    assert params.chosen_k in params.ks
    assert set(params.silhouette_plot_ks) <= set(params.ks)


def test_quality_thresholds_prioritize_stability():
    quality = get_config().quality
    assert quality.min_stability_ari == 0.90
    assert quality.min_silhouette == 0.20
    assert quality.min_segment_pct == 0.03
