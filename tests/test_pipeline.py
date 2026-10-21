import json

import pandas as pd

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


def test_features_step_requires_clean_file(tmp_config, capsys):
    assert main(["--step", "features"], config=tmp_config) == 1
    assert "poe clean" in capsys.readouterr().err
