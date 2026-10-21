import hashlib
import subprocess
import zipfile

import pandas as pd
import pytest

from segmentacion.data import download
from segmentacion.data.download import (
    META_END,
    META_START,
    DataAcquisitionError,
    acquire,
    check_size,
    count_rows,
    extract_csvs,
    main,
    merge_sources,
    sha256sum,
    update_data_card,
    verify_raw_file,
)
from segmentacion.data.load import detect_encoding, read_header


@pytest.fixture
def fake_kaggle(monkeypatch):
    monkeypatch.setattr(download.shutil, "which", lambda _: "kaggle")


def fake_runner_writing_zip(files: dict[str, str], returncode: int = 0):
    """Simula `kaggle competitions download` creando un ZIP en el directorio `-p`."""

    def runner(cmd, **_):
        dest = cmd[cmd.index("-p") + 1]
        if returncode == 0:
            with zipfile.ZipFile(f"{dest}/competition.zip", "w") as zf:
                for name, content in files.items():
                    zf.writestr(f"nested/{name}", content)
                zf.writestr("README.txt", "ignorar")
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr="401 Unauthorized")

    return runner


def test_check_size_accepts_small_file(tmp_path):
    path = tmp_path / "a.csv"
    path.write_bytes(b"x" * 1024)
    assert check_size(path, max_mb=20) == pytest.approx(1 / 1024)


def test_check_size_rejects_file_over_limit(tmp_path):
    path = tmp_path / "a.csv"
    path.write_bytes(b"x" * (2 * 1024 * 1024))
    with pytest.raises(DataAcquisitionError, match="límite"):
        check_size(path, max_mb=1)


def test_sha256sum_matches_hashlib(tmp_path):
    path = tmp_path / "a.csv"
    path.write_bytes(b"hola mundo")
    assert sha256sum(path, chunk_size=3) == hashlib.sha256(b"hola mundo").hexdigest()


def test_extract_csvs_flattens_paths_and_skips_non_csv(tmp_path):
    zip_path = tmp_path / "x.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../../evil/flight.csv", "a,b\n1,2\n")
        zf.writestr("notes.txt", "no")
    out = tmp_path / "out"
    out.mkdir()
    assert extract_csvs(zip_path, out) == [out / "flight.csv"]
    assert (out / "flight.csv").read_text() == "a,b\n1,2\n"


def test_detect_encoding_supports_chinese_gb18030(tmp_path):
    path = tmp_path / "a.csv"
    path.write_bytes("WORK_CITY\n广州\n".encode("gb18030"))
    assert detect_encoding(path) == "gb18030"
    assert read_header(path) == ["WORK_CITY"]


def test_verify_raw_file_reports_rows_and_columns(flight_csv):
    report = verify_raw_file(flight_csv, max_mb=20)
    assert report.n_rows == 200
    assert len(report.columns) == 23
    assert report.extra_columns == []


def test_verify_raw_file_fails_if_missing(tmp_path):
    with pytest.raises(DataAcquisitionError, match="No existe"):
        verify_raw_file(tmp_path / "flight.csv", max_mb=20)


def test_verify_raw_file_fails_on_missing_columns(flight_csv, flight_df):
    flight_df.drop(columns=["avg_discount"]).to_csv(flight_csv, index=False)
    with pytest.raises(DataAcquisitionError, match="avg_discount"):
        verify_raw_file(flight_csv, max_mb=20)


def test_update_data_card_replaces_block_idempotently(tmp_path):
    card = tmp_path / "card.md"
    card.write_text(f"# Título\n\n{META_START}\nviejo\n{META_END}\n\n## Fin\n", encoding="utf-8")
    block = f"{META_START}\nnuevo\n{META_END}"
    update_data_card(card, block)
    update_data_card(card, block)
    text = card.read_text(encoding="utf-8")
    assert text.count(META_START) == 1
    assert "nuevo" in text and "viejo" not in text
    assert text.startswith("# Título") and text.rstrip().endswith("## Fin")


def test_download_without_kaggle_cli_gives_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr(download.shutil, "which", lambda _: None)
    with pytest.raises(DataAcquisitionError, match="Kaggle CLI"):
        download.download_competition("comp", tmp_path)


def test_download_failure_explains_credentials_and_rules(fake_kaggle, tmp_path):
    runner = fake_runner_writing_zip({}, returncode=1)
    with pytest.raises(DataAcquisitionError, match="reglas") as exc:
        download.download_competition("comp", tmp_path, runner=runner)
    assert "401 Unauthorized" in str(exc.value)


def test_count_rows_excludes_header_and_blank_lines(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("a,b\n1,2\n3,4\n\n", encoding="utf-8")
    assert count_rows(path) == 2


def test_merge_sources_keeps_single_header_and_original_text(tmp_path):
    train, test = tmp_path / "train.csv", tmp_path / "test.csv"
    train.write_bytes(b"A,B\n1,0.961639043\n")
    test.write_bytes(b"A,B\n2,\n")  # sin salto de línea final y con un nulo
    dest = tmp_path / "all.csv"
    merge_sources([train, test], dest)
    assert dest.read_bytes() == b"A,B\n1,0.961639043\n2,\n"


def test_merge_sources_rejects_different_headers(tmp_path):
    (tmp_path / "a.csv").write_text("A,B\n1,2\n")
    (tmp_path / "b.csv").write_text("A,C\n1,2\n")
    with pytest.raises(DataAcquisitionError, match="encabezados"):
        merge_sources([tmp_path / "a.csv", tmp_path / "b.csv"], tmp_path / "all.csv")


def test_merge_sources_lists_available_csvs_when_missing(tmp_path):
    (tmp_path / "train.csv").write_text("A\n1\n")
    with pytest.raises(DataAcquisitionError, match="train.csv"):
        merge_sources([tmp_path / "flight_train.csv"], tmp_path / "flight.csv")


def test_verify_raw_file_rejects_duplicate_member_no(flight_csv, flight_df):
    pd.concat([flight_df, flight_df.head(3)]).to_csv(flight_csv, index=False)
    with pytest.raises(DataAcquisitionError, match="3 MEMBER_NO duplicados"):
        verify_raw_file(flight_csv, max_mb=20)


def test_acquire_downloads_merges_and_updates_card(fake_kaggle, monkeypatch, tmp_config, flight_df):
    files = {
        "flight_train.csv": flight_df.iloc[:150].to_csv(index=False),
        "flight_test.csv": flight_df.iloc[150:].to_csv(index=False),
        "sample.csv": "MEMBER_NO,FLIGHT_COUNT\n1,2\n",
    }
    runner = fake_runner_writing_zip(files)
    monkeypatch.setattr(download, "download_competition", _bind_runner(runner))

    report = acquire(tmp_config)

    assert report.n_rows == len(flight_df)
    assert not list(tmp_config.paths.raw_dir.glob("*.zip"))
    card = tmp_config.paths.data_card.read_text(encoding="utf-8")
    assert report.sha256 in card
    assert "`flight_train.csv` (150) + `flight_test.csv` (50)" in card


def test_acquire_fails_if_merged_file_is_stale(tmp_config, flight_df, flight_csv):
    flight_df.iloc[:150].to_csv(tmp_config.source_paths[0], index=False)
    flight_df.iloc[150:190].to_csv(tmp_config.source_paths[1], index=False)
    with pytest.raises(DataAcquisitionError, match="--force"):
        acquire(tmp_config, verify_only=True)


def test_acquire_skips_download_when_file_exists(monkeypatch, tmp_config, flight_csv):
    def fail(*_, **__):
        raise AssertionError("no debería descargar")

    monkeypatch.setattr(download, "download_competition", fail)
    assert acquire(tmp_config).n_rows == 200


def test_main_verify_only_returns_error_code_when_missing(tmp_config, capsys):
    assert main(["--verify-only"], config=tmp_config) == 1
    assert "Faltan archivos fuente" in capsys.readouterr().err


def test_main_verify_only_ok(tmp_config, flight_csv, capsys):
    assert main(["--verify-only"], config=tmp_config) == 0
    assert "200 filas" in capsys.readouterr().out


def _bind_runner(runner):
    original = download.download_competition

    def bound(competition, dest_dir):
        return original(competition, dest_dir, runner=runner)

    return bound
