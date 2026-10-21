"""CLI del pipeline: `python -m segmentacion.pipeline --step all`.

Cada fase agrega su paso a `STEPS`; `all` los ejecuta en orden.
"""

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score

from segmentacion.config import Config, get_config
from segmentacion.data.clean import clean, lineage_report, load_clean, null_summary, save_clean
from segmentacion.data.load import load_raw
from segmentacion.features.lrfmc import LRFMC_COLUMNS, build_lrfmc, build_profile_features
from segmentacion.models.baseline import baseline_segments, quintile_scores
from segmentacion.models.benchmark import run_benchmark
from segmentacion.models.k_selection import (
    cluster_medians,
    evaluate_k_range,
    find_elbow,
    silhouette_curve,
    silhouette_profile,
)
from segmentacion.models.kmeans import (
    fit_kmeans,
    save_model,
    stability_ari,
    subsample_stability,
)
from segmentacion.preprocessing.transform import (
    compare_preprocessors,
    make_preprocessor,
    save_preprocessor,
    skewness,
)


def run_clean(config: Config) -> dict:
    """Fase 2: flight.csv → data/interim/flight_clean.parquet + lineage JSON."""
    if not config.raw_file.exists():
        raise FileNotFoundError(f"No existe {config.raw_file}. Ejecuta `uv run poe download`.")
    raw = load_raw(config.raw_file)
    df, lineage = clean(raw, config.cleaning)
    report = lineage_report(len(raw), len(df), lineage)
    report["null_counts"] = null_summary(raw, df)
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


def run_features(config: Config) -> dict:
    """Fase 4: flight_clean.parquet → data/processed/lrfmc.parquet + profile.parquet."""
    if not config.clean_file.exists():
        raise FileNotFoundError(f"No existe {config.clean_file}. Ejecuta `uv run poe clean`.")
    df = load_clean(config.clean_file)
    lrfmc = build_lrfmc(df)
    profile = build_profile_features(df)
    config.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    lrfmc.to_parquet(config.lrfmc_file, index=False)
    profile.to_parquet(config.profile_file, index=False)
    print(
        f"features: {len(lrfmc):,} socios → {config.lrfmc_file.name} "
        f"(L, R, F, M, C) + {config.profile_file.name} ({profile.shape[1] - 1} variables)"
    )
    return {"rows": len(lrfmc)}


def run_transform(config: Config) -> dict:
    """Fase 5: compara transformaciones, ajusta la elegida y guarda el preprocesador."""
    if not config.lrfmc_file.exists():
        raise FileNotFoundError(f"No existe {config.lrfmc_file}. Ejecuta `uv run poe features`.")
    lrfmc = pd.read_parquet(config.lrfmc_file)
    features = lrfmc[list(LRFMC_COLUMNS)]
    params = config.preprocessing
    comparison = compare_preprocessors(
        features, params.compare_ks, params.silhouette_sample_size, config.random_state
    )
    comparison.to_csv(config.transform_comparison_file, index=False)

    preprocessor = make_preprocessor(params.transform, params.scaler)
    scaled = preprocessor.fit_transform(features)
    save_preprocessor(preprocessor, config.preprocessor_file)
    scaled.insert(0, "MEMBER_NO", lrfmc["MEMBER_NO"].to_numpy())
    scaled.to_parquet(config.scaled_file, index=False)

    before, after = skewness(features), skewness(scaled[list(LRFMC_COLUMNS)])
    for column in LRFMC_COLUMNS:
        print(f"  · skew {column}: {before[column]:+.2f} → {after[column]:+.2f}")
    print(
        f"transform: {params.transform} + {params.scaler} → {config.scaled_file.name}, "
        f"{config.preprocessor_file.name} (comparación en {config.transform_comparison_file.name})"
    )
    return {"rows": len(scaled)}


def run_k_selection(config: Config) -> dict:
    """Fase 6: métricas de K-Means para cada k + silhouette plots de los k candidatos."""
    if not config.scaled_file.exists():
        raise FileNotFoundError(f"No existe {config.scaled_file}. Ejecuta `uv run poe transform`.")
    features = pd.read_parquet(config.scaled_file)[list(LRFMC_COLUMNS)]
    original = pd.read_parquet(config.lrfmc_file)[list(LRFMC_COLUMNS)]
    params = config.k_selection
    ks = [k for k in params.ks if k < len(features)]
    table = evaluate_k_range(features, ks, params.silhouette_sample_size, config.random_state)
    elbow = find_elbow(table["k"], table["inertia"])
    table["elbow"] = table["k"] == elbow
    table.to_csv(config.k_selection_file, index=False)

    curves = []
    for k in [k for k in params.silhouette_plot_ks if k in ks]:
        labels = fit_kmeans(features, k, config.random_state).labels_
        profile = silhouette_profile(
            features, labels, params.silhouette_sample_size, config.random_state
        )
        curves.append(silhouette_curve(profile).assign(k=k))
        if k == params.chosen_k:
            cluster_medians(original, labels).to_csv(config.k_preview_file, index=False)
    pd.concat(curves, ignore_index=True).to_parquet(config.silhouette_curves_file, index=False)

    chosen = table.set_index("k").loc[params.chosen_k]
    print(f"  · codo de la inercia: k = {elbow}")
    print(
        f"  · k elegido = {params.chosen_k}: silhouette {chosen['silhouette']:.3f}, "
        f"Davies-Bouldin {chosen['db']:.3f}, clúster mínimo {chosen['min_cluster_pct']:.1%}"
    )
    print(f"k_selection: k = {ks[0]}…{ks[-1]} → {config.k_selection_file.name}")
    return {"elbow": elbow}


def run_clustering(config: Config) -> dict:
    """Fase 7: K-Means final + estabilidad + baseline por quintiles + benchmark."""
    if not config.scaled_file.exists():
        raise FileNotFoundError(f"No existe {config.scaled_file}. Ejecuta `uv run poe transform`.")
    scaled = pd.read_parquet(config.scaled_file)
    lrfmc = pd.read_parquet(config.lrfmc_file)
    features = scaled[list(LRFMC_COLUMNS)]
    k, params, quality = config.k_selection.chosen_k, config.modeling, config.quality

    start = time.perf_counter()
    kmeans = fit_kmeans(features, k, config.random_state)
    kmeans_seconds = time.perf_counter() - start
    seeds = stability_ari(features, k, list(range(params.stability_seeds)))
    subsamples = subsample_stability(
        features, k, params.subsample_runs, params.subsample_fraction, config.random_state
    )
    stability = pd.concat(
        [
            pd.DataFrame(
                {
                    "method": "semillas",
                    "run": seeds["seed_a"].astype(str) + "–" + seeds["seed_b"].astype(str),
                    "ari": seeds["ari"],
                }
            ),
            subsamples.assign(method="submuestras", run=subsamples["run"].astype(str)),
        ],
        ignore_index=True,
    )[["method", "run", "ari"]]

    scores = quintile_scores(lrfmc[list(LRFMC_COLUMNS)])
    baseline = baseline_segments(scores)
    bench = run_benchmark(
        features,
        kmeans.labels_,
        baseline.to_numpy(),
        k,
        kmeans_seconds=kmeans_seconds,
        covariance_types=params.gmm_covariance_types,
        gmm_n_init=params.gmm_n_init,
        agglomerative_sample_size=params.agglomerative_sample_size,
        dbscan_min_samples=params.dbscan_min_samples,
        silhouette_sample_size=params.silhouette_sample_size,
        random_state=config.random_state,
    )

    segments = pd.DataFrame(
        {
            "MEMBER_NO": lrfmc["MEMBER_NO"].to_numpy(),
            "cluster": kmeans.labels_,
            "baseline_segment": baseline.to_numpy(),
            "gmm_cluster": bench.labels["GMM"],
            "gmm_proba": bench.gmm.predict_proba(features).max(axis=1),
            "agglomerative_cluster": bench.labels["Agglomerative (Ward)"],
            "dbscan_outlier": bench.labels["DBSCAN"] == -1,
            **{f"score_{c}": scores[c].to_numpy() for c in LRFMC_COLUMNS},
        }
    )
    config.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    segments.to_parquet(config.segments_file, index=False)
    stability.to_csv(config.stability_file, index=False)
    bench.table.to_csv(config.benchmark_file, index=False)
    bench.k_distance.to_csv(config.k_distance_file, index=False)
    save_model(kmeans, config.kmeans_file)
    save_model(bench.gmm, config.gmm_file)

    kmeans_row = bench.table.set_index("model").loc["K-Means"]
    sizes = np.bincount(kmeans.labels_, minlength=k) / len(features)
    ari_mean = stability.groupby("method")["ari"].mean()
    ari_min = stability.groupby("method")["ari"].min()
    metrics = {
        "k": k,
        "members": len(features),
        "kmeans": {
            "inertia": float(kmeans.inertia_),
            "silhouette": float(kmeans_row["silhouette"]),
            "ch": float(calinski_harabasz_score(features, kmeans.labels_)),
            "db": float(kmeans_row["db"]),
            "cluster_pct": sizes.round(6).tolist(),
        },
        "stability": {
            "seeds": params.stability_seeds,
            "seed_ari_mean": float(ari_mean["semillas"]),
            "seed_ari_min": float(ari_min["semillas"]),
            "subsample_fraction": params.subsample_fraction,
            "subsample_ari_mean": float(ari_mean["submuestras"]),
            "subsample_ari_min": float(ari_min["submuestras"]),
        },
        "benchmark": {
            "gmm_covariance_type": bench.gmm.covariance_type,
            "gmm_bic": dict(
                zip(bench.gmm_bic["covariance_type"], bench.gmm_bic["bic"], strict=True)
            ),
            "dbscan_eps": bench.eps,
            "dbscan_min_samples": params.dbscan_min_samples,
        },
    }
    metrics["quality"] = {
        "stability_ok": metrics["stability"]["seed_ari_mean"] >= quality.min_stability_ari,
        "silhouette_ok": metrics["kmeans"]["silhouette"] >= quality.min_silhouette,
        "segment_size_ok": float(sizes.min()) >= quality.min_segment_pct,
    }
    config.metrics_file.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), "utf-8")

    stab, km = metrics["stability"], metrics["kmeans"]
    print(
        f"  · estabilidad: ARI entre semillas {stab['seed_ari_mean']:.3f} "
        f"(mín. {stab['seed_ari_min']:.3f}) · submuestras {stab['subsample_ari_mean']:.3f}"
    )
    print(
        f"  · K-Means k = {k}: silhouette {km['silhouette']:.3f}, clúster mínimo {sizes.min():.1%}"
    )
    for row in bench.table.itertuples():
        print(
            f"  · {row.model}: {row.n_clusters} grupos, ruido {row.noise_pct:.1%}, "
            f"silhouette {row.silhouette:.3f}, ARI vs K-Means {row.ari_kmeans:.3f}"
        )
    failed = [name for name, ok in metrics["quality"].items() if not ok]
    if failed:
        print(f"  ! criterios de calidad no cumplidos: {', '.join(failed)}")
    print(
        f"clustering: k = {k} → {config.segments_file.name}, {config.metrics_file.name}, "
        f"{config.benchmark_file.name}, {config.kmeans_file.name}"
    )
    return metrics


STEPS: dict[str, Callable[[Config], object]] = {
    "clean": run_clean,
    "features": run_features,
    "transform": run_transform,
    "k_selection": run_k_selection,
    "clustering": run_clustering,
}


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
