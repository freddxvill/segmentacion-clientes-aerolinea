"""Carga de `configs/params.yaml` y resolución de rutas del proyecto."""

from dataclasses import dataclass, field
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
    clean_filename: str
    lineage_filename: str
    lrfmc_filename: str = "lrfmc.parquet"
    profile_filename: str = "profile.parquet"
    scaled_filename: str = "lrfmc_scaled.parquet"
    transform_comparison_filename: str = "transform_comparison.csv"
    preprocessor_filename: str = "preprocessor.joblib"


@dataclass(frozen=True)
class CleaningParams:
    date_format: str = "%m/%d/%Y"
    age_min: int = 18
    age_max: int = 100
    text_placeholders: tuple[str, ...] = (".", "-", "。", "0", "`", "——", "nan")


@dataclass(frozen=True)
class PreprocessingParams:
    transform: str = "yeo-johnson"
    scaler: str = "standard"
    compare_ks: tuple[int, ...] = (4, 5, 6)
    silhouette_sample_size: int = 10_000


@dataclass(frozen=True)
class Config:
    paths: Paths
    data: DataParams
    cleaning: CleaningParams
    random_state: int
    preprocessing: PreprocessingParams = field(default_factory=PreprocessingParams)

    @property
    def raw_file(self) -> Path:
        return self.paths.raw_dir / self.data.raw_filename

    @property
    def source_paths(self) -> list[Path]:
        return [self.paths.raw_dir / name for name in self.data.source_files]

    @property
    def clean_file(self) -> Path:
        return self.paths.interim_dir / self.data.clean_filename

    @property
    def lineage_file(self) -> Path:
        return self.paths.interim_dir / self.data.lineage_filename

    @property
    def lrfmc_file(self) -> Path:
        return self.paths.processed_dir / self.data.lrfmc_filename

    @property
    def profile_file(self) -> Path:
        return self.paths.processed_dir / self.data.profile_filename

    @property
    def scaled_file(self) -> Path:
        return self.paths.processed_dir / self.data.scaled_filename

    @property
    def transform_comparison_file(self) -> Path:
        return self.paths.processed_dir / self.data.transform_comparison_filename

    @property
    def preprocessor_file(self) -> Path:
        return self.paths.models_dir / self.data.preprocessor_filename


def load_config(params_file: Path = DEFAULT_PARAMS_FILE, root: Path = PROJECT_ROOT) -> Config:
    """Lee el YAML de parámetros y devuelve un `Config` con rutas absolutas."""
    raw: dict[str, Any] = yaml.safe_load(params_file.read_text(encoding="utf-8"))
    paths = Paths(**{key: root / value for key, value in raw["paths"].items()})
    data = DataParams(**{**raw["data"], "source_files": tuple(raw["data"]["source_files"])})
    cleaning_raw = raw["cleaning"]
    cleaning = CleaningParams(
        **{**cleaning_raw, "text_placeholders": tuple(cleaning_raw["text_placeholders"])}
    )
    pre_raw = raw.get("preprocessing", {})
    preprocessing = PreprocessingParams(
        **{**pre_raw, "compare_ks": tuple(pre_raw.get("compare_ks", (4, 5, 6)))}
    )
    return Config(
        paths=paths,
        data=data,
        cleaning=cleaning,
        random_state=raw["random_state"],
        preprocessing=preprocessing,
    )


@cache
def get_config() -> Config:
    """Config por defecto del proyecto (cacheada)."""
    return load_config()
