"""Adquisición del dataset (Fase 1): descarga desde Kaggle, verificación y ficha del dataset.

Uso:
    uv run poe download       # descarga + verificación + actualización de docs/data-card.md
    uv run poe verify-data    # solo verifica un data/raw/flight.csv ya presente
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from segmentacion.config import Config, get_config
from segmentacion.data.load import EXPECTED_COLUMNS, detect_encoding, load_raw, read_header

META_START = "<!-- META:START -->"
META_END = "<!-- META:END -->"

Runner = Callable[..., subprocess.CompletedProcess]


class DataAcquisitionError(RuntimeError):
    """Error de descarga o verificación del dataset, con un mensaje accionable."""


@dataclass(frozen=True)
class RawFileReport:
    path: Path
    size_mb: float
    sha256: str
    encoding: str
    n_rows: int
    columns: list[str]
    extra_columns: list[str]
    checked_on: date


def download_competition(
    competition: str, dest_dir: Path, runner: Runner = subprocess.run
) -> list[Path]:
    """Descarga los archivos de una competición de Kaggle a `dest_dir` y devuelve los ZIP."""
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise DataAcquisitionError(
            "No se encontró el Kaggle CLI. Ejecuta `uv sync` y vuelve a intentarlo."
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = [kaggle, "competitions", "download", "-c", competition, "-p", str(dest_dir)]
    result = runner(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise DataAcquisitionError(
            f"Falló la descarga de Kaggle:\n{(result.stderr or result.stdout).strip()}\n\n"
            "Revisa que:\n"
            "  1. Tengas credenciales (`uv run kaggle auth login` o ~/.kaggle/kaggle.json).\n"
            "  2. Hayas aceptado las reglas en "
            f"https://www.kaggle.com/competitions/{competition}/rules\n"
            f"Alternativa: descarga el archivo a mano, cópialo en {dest_dir} y ejecuta "
            "`uv run poe verify-data`."
        )
    return sorted(dest_dir.glob("*.zip"))


def extract_csvs(zip_path: Path, dest_dir: Path) -> list[Path]:
    """Extrae los CSV de un ZIP en `dest_dir` (aplanando rutas internas)."""
    extracted = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir() or not member.filename.lower().endswith(".csv"):
                continue
            # Path(...).name evita escribir fuera de dest_dir (path traversal).
            target = dest_dir / Path(member.filename).name
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(target)
    return extracted


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def check_size(path: Path, max_mb: float) -> float:
    """Valida la restricción de tamaño del proyecto y devuelve el tamaño en MB."""
    size = file_size_mb(path)
    if size >= max_mb:
        raise DataAcquisitionError(
            f"{path.name} pesa {size:.2f} MB y el límite del proyecto es {max_mb} MB."
        )
    return size


def sha256sum(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def count_rows(path: Path) -> int:
    """Filas de datos (sin encabezado) de un CSV sin saltos de línea dentro de campos."""
    with path.open("rb") as f:
        return sum(1 for line in f if line.strip()) - 1


def merge_sources(sources: Sequence[Path], dest: Path) -> None:
    """Concatena CSV con el mismo encabezado conservando el texto original (sin re-serializar)."""
    missing = [p.name for p in sources if not p.exists()]
    if missing:
        found = ", ".join(sorted(p.name for p in dest.parent.glob("*.csv"))) or "ningún CSV"
        raise DataAcquisitionError(
            f"Faltan archivos fuente: {missing} (en {dest.parent} hay: {found}). "
            "Revisa `data.source_files` en configs/params.yaml."
        )
    headers = {p.name: read_header(p) for p in sources}
    if len({tuple(h) for h in headers.values()}) > 1:
        raise DataAcquisitionError(f"Los archivos fuente tienen encabezados distintos: {headers}")
    with dest.open("wb") as out:
        for i, src in enumerate(sources):
            lines = src.read_bytes().splitlines(keepends=True)
            body = lines if i == 0 else lines[1:]
            out.writelines(line if line.endswith(b"\n") else line + b"\n" for line in body)


def verify_raw_file(path: Path, max_mb: float) -> RawFileReport:
    """Verifica existencia, tamaño, columnas y unicidad de `MEMBER_NO` del CSV crudo."""
    if not path.exists():
        raise DataAcquisitionError(
            f"No existe {path}. Ejecuta `uv run poe download` o copia el archivo a mano."
        )
    size = check_size(path, max_mb)
    df = load_raw(path)
    columns = list(df.columns)
    missing = [c for c in EXPECTED_COLUMNS if c not in columns]
    if missing:
        raise DataAcquisitionError(f"Faltan columnas esperadas en {path.name}: {missing}")
    n_duplicated = int(df["MEMBER_NO"].duplicated().sum())
    if n_duplicated:
        raise DataAcquisitionError(f"{path.name} tiene {n_duplicated} MEMBER_NO duplicados.")
    return RawFileReport(
        path=path,
        size_mb=round(size, 2),
        sha256=sha256sum(path),
        encoding=detect_encoding(path),
        n_rows=len(df),
        columns=columns,
        extra_columns=[c for c in columns if c not in EXPECTED_COLUMNS],
        checked_on=date.today(),
    )


def render_metadata_block(
    report: RawFileReport, competition: str, source_rows: Sequence[tuple[str, int]] = ()
) -> str:
    extra = ", ".join(f"`{c}`" for c in report.extra_columns) or "ninguna"
    sources = " + ".join(f"`{name}` ({rows:,})" for name, rows in source_rows) or "—"
    return "\n".join(
        [
            META_START,
            "| Campo | Valor |",
            "|---|---|",
            f"| Fuente | https://www.kaggle.com/competitions/{competition} |",
            f"| Archivos fuente (filas) | {sources} |",
            f"| Archivo | `{report.path.name}` |",
            f"| Fecha de verificación | {report.checked_on.isoformat()} |",
            f"| Tamaño | {report.size_mb} MB |",
            f"| SHA-256 | `{report.sha256}` |",
            f"| Codificación | `{report.encoding}` |",
            f"| Filas | {report.n_rows:,} |",
            f"| Columnas | {len(report.columns)} |",
            f"| Columnas no esperadas | {extra} |",
            META_END,
        ]
    )


def update_data_card(card_path: Path, block: str) -> None:
    """Reemplaza el bloque de metadatos entre marcadores (o lo agrega al final)."""
    text = card_path.read_text(encoding="utf-8") if card_path.exists() else ""
    if META_START in text and META_END in text:
        before, rest = text.split(META_START, 1)
        _, after = rest.split(META_END, 1)
        text = f"{before}{block}{after}"
    else:
        text = f"{text.rstrip()}\n\n{block}\n" if text else f"{block}\n"
    card_path.write_text(text, encoding="utf-8")


def acquire(config: Config, *, verify_only: bool = False, force: bool = False) -> RawFileReport:
    """Orquesta descarga (opcional), unión de fuentes, verificación y data card."""
    raw_file, sources = config.raw_file, config.source_paths
    have_data = raw_file.exists() or all(p.exists() for p in sources)
    if not verify_only and (force or not have_data):
        for zip_path in download_competition(config.data.kaggle_competition, raw_file.parent):
            extract_csvs(zip_path, raw_file.parent)
            zip_path.unlink()
    if force or not raw_file.exists():
        merge_sources(sources, raw_file)
    report = verify_raw_file(raw_file, config.data.max_raw_size_mb)
    source_rows = [(p.name, count_rows(p)) for p in sources if p.exists()]
    if source_rows and sum(rows for _, rows in source_rows) != report.n_rows:
        raise DataAcquisitionError(
            f"{raw_file.name} tiene {report.n_rows} filas, pero las fuentes suman "
            f"{sum(rows for _, rows in source_rows)}. Regenera con `uv run poe download --force`."
        )
    block = render_metadata_block(report, config.data.kaggle_competition, source_rows)
    update_data_card(config.paths.data_card, block)
    return report


def main(argv: Sequence[str] | None = None, config: Config | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify-only", action="store_true", help="No descarga; solo verifica.")
    parser.add_argument("--force", action="store_true", help="Descarga aunque el archivo exista.")
    args = parser.parse_args(argv)
    config = config or get_config()
    try:
        report = acquire(config, verify_only=args.verify_only, force=args.force)
    except DataAcquisitionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"OK: {report.path.name} · {report.size_mb} MB · {report.n_rows:,} filas · "
        f"{len(report.columns)} columnas · sha256={report.sha256[:12]}…\n"
        f"Ficha actualizada: {config.paths.data_card}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
