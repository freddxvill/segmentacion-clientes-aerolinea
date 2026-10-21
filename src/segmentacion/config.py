"""Carga de `configs/params.yaml` y resolución de rutas del proyecto."""

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARAMS_FILE = PROJECT_ROOT / "configs" / "params.yaml"


@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    models_dir: Path
    data_card: Path


@dataclass(frozen=True)
class DataParams:
    kaggle_competition: str
    source_files: tuple[str, ...]
    raw_filename: str
    max_raw_size_mb: float


@dataclass(frozen=True)
class Config:
    paths: Paths
    data: DataParams
    random_state: int

    @property
    def raw_file(self) -> Path:
        return self.paths.raw_dir / self.data.raw_filename

    @property
    def source_paths(self) -> list[Path]:
        return [self.paths.raw_dir / name for name in self.data.source_files]


def load_config(params_file: Path = DEFAULT_PARAMS_FILE, root: Path = PROJECT_ROOT) -> Config:
    """Lee el YAML de parámetros y devuelve un `Config` con rutas absolutas."""
    raw: dict[str, Any] = yaml.safe_load(params_file.read_text(encoding="utf-8"))
    paths = Paths(**{key: root / value for key, value in raw["paths"].items()})
    data = DataParams(**{**raw["data"], "source_files": tuple(raw["data"]["source_files"])})
    return Config(paths=paths, data=data, random_state=raw["random_state"])


@cache
def get_config() -> Config:
    """Config por defecto del proyecto (cacheada)."""
    return load_config()
