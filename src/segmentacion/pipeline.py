"""CLI del pipeline: `python -m segmentacion.pipeline --step all`.

Cada fase agrega su paso a `STEPS`; `all` los ejecuta en orden.
"""

import argparse
import sys
from collections.abc import Callable, Sequence

from segmentacion.config import Config, get_config
from segmentacion.data.clean import clean, lineage_report, save_clean
from segmentacion.data.load import load_raw


def run_clean(config: Config) -> dict:
    """Fase 2: flight.csv → data/interim/flight_clean.parquet + lineage JSON."""
    if not config.raw_file.exists():
        raise FileNotFoundError(f"No existe {config.raw_file}. Ejecuta `uv run poe download`.")
    raw = load_raw(config.raw_file)
    df, lineage = clean(raw, config.cleaning)
    report = lineage_report(len(raw), len(df), lineage)
    save_clean(df, report, config.clean_file, config.lineage_file)
    for step in lineage:
        if step.rows:
            verb = "eliminadas" if step.action == "drop" else "anuladas"
            print(f"  · {step.description}: {step.rows:,} filas {verb}")
    print(
        f"clean: {report['input_rows']:,} → {report['output_rows']:,} filas "
        f"({report['dropped_rows']:,} eliminadas) → {config.clean_file.name}"
    )
    return report


STEPS: dict[str, Callable[[Config], object]] = {"clean": run_clean}


def main(argv: Sequence[str] | None = None, config: Config | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pipeline de segmentación LRFMC")
    parser.add_argument("--step", choices=[*STEPS, "all"], default="all")
    args = parser.parse_args(argv)
    config = config or get_config()
    steps = STEPS if args.step == "all" else {args.step: STEPS[args.step]}
    try:
        for step in steps.values():
            step(config)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
