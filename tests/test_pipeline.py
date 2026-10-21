import json

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
