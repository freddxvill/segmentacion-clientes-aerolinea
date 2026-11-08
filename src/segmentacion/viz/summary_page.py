"""Resumen interactivo del proyecto en una sola página HTML (`poe resumen` → report/resumen.html).

Es un *explainer* con scroll narrativo: capítulos con un escenario fijo, pasos de texto y gráficos
SVG que cambian con el scroll. La página es una plantilla (`templates/resumen.html`, con su CSS y
su JS) en la que este módulo inyecta:

- el **texto**: cada `[[clave]]` se sustituye por una cifra ya formateada con `viz.fmt` (la misma
  que usan el sitio y el README) o por un fragmento HTML generado aquí (tablas, fichas);
- los **datos de los gráficos**, como JSON en `<script type="application/json" id="data">`.

Así la página no tiene cifras escritas a mano: se regenera desde los artefactos del pipeline y se
publica junto al sitio (`report/resumen.html`).
"""

import html
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from segmentacion.config import PROJECT_ROOT, Config, get_config
from segmentacion.segments.naming import archetype, rule_in_words
from segmentacion.segments.recommendations import recommendations_for
from segmentacion.viz.fmt import num, pct

TEMPLATE = Path(__file__).with_name("templates") / "resumen.html"
OUTPUT = PROJECT_ROOT / "report" / "resumen.html"
PLACEHOLDER = re.compile(r"\[\[(\w+)\]\]")
DATA_MARK = "/*DATOS*/null"

LRFMC = ("L", "R", "F", "M", "C")
SHORT_NAMES = {
    "VIP": "VIP",
    "Nuevos con potencial": "Nuevos",
    "Veteranos ocasionales": "Veteranos",
    "En riesgo de abandono": "En riesgo",
    "Cazadores de ofertas": "Cazadores",
}
MODEL_NAMES = {
    "K-Means": "K-Means",
    "Agglomerative (Ward)": "Ward (jerárquico)",
    "GMM": "GMM (gaussianas)",
    "Quintiles LRFMC": "Reglas por quintiles",
}
PROFILE_LABELS = {
    "L": "Antigüedad",
    "R": "Meses sin volar",
    "F": "Vuelos",
    "M": "Kilómetros",
    "C": "Precio pagado",
    "T": "Tendencia del ingreso",
}
MAP_POINTS = 1500  # puntos del mapa t-SNE: suficientes para ver las zonas, ligeros para el SVG
RAW_EDGES = np.arange(0, 121, 4)  # vuelos: 30 tramos de 4; el último incluye a los de ≥ 116
SCALED_BANDWIDTH = 0.2  # suavizado (en desviaciones) del histograma de vuelos ya transformado


def metric_label(metric: str, value: float) -> str:
    """Valor de una métrica LRFMC con su unidad, como lo muestra la página («42 meses», «71 %»)."""
    return {
        "L": lambda v: f"{num(v)} meses",
        "R": lambda v: f"{num(v, 1)} meses",
        "F": lambda v: f"{num(v)} vuelos",
        "M": lambda v: f"{num(v)} km",
        "C": lambda v: pct(v, 0),
    }[metric](value)


@dataclass(frozen=True)
class Inputs:
    """Artefactos del pipeline que alimentan la página (ver `load_inputs`)."""

    profiles: pd.DataFrame  # segment_profiles.csv, en el orden de los segmentos
    lineage: dict[str, Any]
    metrics: dict[str, Any]
    k_table: pd.DataFrame
    stability: pd.DataFrame
    benchmark: pd.DataFrame
    tests: pd.DataFrame
    lrfmc: pd.DataFrame
    scaled: pd.DataFrame
    tsne: pd.DataFrame  # columnas x, y, segment (una sola perplexity)
    trustworthiness: float
    min_stability_ari: float
    min_silhouette: float
    random_state: int


def load_inputs(config: Config) -> Inputs:
    profiles = pd.read_csv(config.segment_profiles_file)
    projection = json.loads(config.projection_metrics_file.read_text(encoding="utf-8"))
    perplexity = projection["tsne_perplexities"][0]
    tsne = pd.read_parquet(config.tsne_file)
    tsne = tsne[tsne["perplexity"] == perplexity].merge(profiles[["cluster", "segment"]])
    return Inputs(
        profiles=profiles,
        lineage=json.loads(config.lineage_file.read_text(encoding="utf-8")),
        metrics=json.loads(config.metrics_file.read_text(encoding="utf-8")),
        k_table=pd.read_csv(config.k_selection_file),
        stability=pd.read_csv(config.stability_file),
        benchmark=pd.read_csv(config.benchmark_file),
        tests=pd.read_csv(config.segment_tests_file),
        lrfmc=pd.read_parquet(config.lrfmc_file),
        scaled=pd.read_parquet(config.scaled_file),
        tsne=tsne.rename(columns={"TSNE1": "x", "TSNE2": "y"})[["x", "y", "segment"]],
        trustworthiness=projection["trustworthiness"][f"t-SNE (perplexity {perplexity})"],
        min_stability_ari=config.quality.min_stability_ari,
        min_silhouette=config.quality.min_silhouette,
        random_state=config.random_state,
    )


# --- Utilidades ---------------------------------------------------------------------------------


def largest_remainder(shares: Sequence[float], total: int = 100) -> list[int]:
    """Reparte `total` unidades en proporción a `shares` (resto mayor): suman exactamente `total`.

    Es lo que necesita la cuadrícula de 100 cuadrados: redondear cada % por separado puede dar 99
    o 101 cuadrados.
    """
    raw = np.asarray(shares, dtype=float)
    raw = raw / raw.sum() * total
    base = np.floor(raw).astype(int)
    order = np.argsort(-(raw - base), kind="stable")
    base[order[: total - base.sum()]] += 1
    return base.tolist()


def histogram_shares(values: Sequence[float], edges: Sequence[float]) -> list[float]:
    """Proporción de valores en cada tramo; los que quedan fuera van al primero o al último."""
    clipped = np.clip(np.asarray(values, dtype=float), edges[0], edges[-1])
    counts, _ = np.histogram(clipped, bins=edges)
    return (counts / counts.sum()).round(5).tolist()


def smooth_shares(values: Sequence[float], edges: Sequence[float], bandwidth: float) -> list[float]:
    """Densidad suavizada (núcleo gaussiano) en el centro de cada tramo, normalizada a 1.

    Los vuelos son enteros: tras transformarlos quedan pocos valores distintos y un histograma
    fino sale «en peine», con tramos vacíos entre ellos. Suavizar muestra la forma real.
    """
    edges = np.asarray(edges, dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2
    unique, counts = np.unique(np.round(np.asarray(values, dtype=float), 4), return_counts=True)
    weights = np.exp(-0.5 * ((centers[:, None] - unique[None, :]) / bandwidth) ** 2)
    density = (weights * counts[None, :]).sum(axis=1)
    return (density / density.sum()).round(5).tolist()


def map_sample(points: pd.DataFrame, segments: Sequence[str], n: int, seed: int) -> list[list]:
    """Muestra estratificada de una proyección 2D: `[x, y, índice de segmento]`, x e y en [0, 1].

    `points` tiene columnas `x`, `y` y `segment`. El orden de salida es aleatorio, para que los
    puntos aparezcan mezclados (no segmento a segmento) cuando el gráfico los revela.
    """
    frac = min(1.0, n / len(points))
    sample = points.groupby("segment", group_keys=False).sample(frac=frac, random_state=seed)
    sample = sample.sample(frac=1.0, random_state=seed)
    xy = sample[["x", "y"]].to_numpy(dtype=float)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    xy = (xy - lo) / np.where(hi > lo, hi - lo, 1.0)
    index = {name: i for i, name in enumerate(segments)}
    return [
        [round(float(x), 4), round(float(y), 4), index[s]]
        for (x, y), s in zip(xy, sample["segment"], strict=True)
    ]


def table_html(headers: Sequence[str], rows: Sequence[Sequence[str]], numeric: set[int]) -> str:
    """Tabla HTML sencilla (escapada) para los bloques «Ver los datos»."""

    def cell(tag: str, i: int, value: str) -> str:
        cls = ' class="n"' if i in numeric else ""
        return f"<{tag}{cls}>{html.escape(str(value))}</{tag}>"

    head = "".join(cell("th", i, h) for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(cell("td", i, v) for i, v in enumerate(row)) + "</tr>" for row in rows
    )
    return (
        f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
    )


def profile_svg(z: dict[str, float]) -> str:
    """Mini gráfico del perfil de un segmento: barras divergentes en z-score (0 = socio medio)."""
    rows, center, scale = list(PROFILE_LABELS), 236, 70  # 1 desviación = 70 px
    parts = [
        f'<line x1="{center}" x2="{center}" y1="2" y2="{len(rows) * 22}" class="mid"/>',
    ]
    for i, key in enumerate(rows):
        value = float(np.clip(z[key], -1.6, 1.6))
        y = i * 22 + 4
        x, width = (
            (center, value * scale) if value >= 0 else (center + value * scale, -value * scale)
        )
        parts.append(f'<text x="0" y="{y + 12}">{html.escape(PROFILE_LABELS[key])}</text>')
        parts.append(
            f'<rect class="b" x="{x:.1f}" y="{y}" width="{width:.1f}" height="14" rx="2"/>'
        )
    parts.append(f'<text x="{center - 62}" y="{len(rows) * 22 + 16}" class="ax">menos</text>')
    parts.append(f'<text x="{center + 30}" y="{len(rows) * 22 + 16}" class="ax">más</text>')
    label = "Perfil frente al socio medio: " + ", ".join(
        f"{PROFILE_LABELS[k]} {num(z[k], 1)}" for k in rows
    )
    return (
        f'<svg class="seg-prof" viewBox="0 0 400 {len(rows) * 22 + 22}" role="img" '
        f'aria-label="{html.escape(label)}">{"".join(parts)}</svg>'
    )


# --- Texto y datos ------------------------------------------------------------------------------


def _segments(inp: Inputs) -> pd.DataFrame:
    return inp.profiles.set_index("segment")


def _z(row: pd.Series) -> dict[str, float]:
    return {key: float(row[f"z_{key}"]) for key in PROFILE_LABELS}


def segment_steps(inp: Inputs) -> str:
    return "\n".join(
        f'<li><em class="k t{i}">{html.escape(name)}</em>. '
        f"{html.escape(archetype(name).description)}</li>"
        for i, name in enumerate(inp.profiles["segment"])
    )


def segment_cards(inp: Inputs) -> str:
    cards, total = [], len(inp.profiles)
    for i, row in inp.profiles.reset_index(drop=True).iterrows():
        name, first = row["segment"], recommendations_for(row["segment"])[0]
        cards.append(
            f'<article class="seg" style="--c:var(--c{i});--tc:var(--t{i})">'
            f'<header><p class="seg-n">{i + 1} / {total}</p><h3>{html.escape(name)}</h3>'
            f'<p class="seg-size">{num(row["members"])} socios</p></header>'
            '<div class="seg-kpis">'
            f"<div><b>{pct(row['pct'], 0)}</b><span>de los socios</span>"
            f'<i style="--w:{row["pct"] * 100:.1f}%"></i></div>'
            f"<div><b>{pct(row['revenue_share'], 0)}</b><span>del ingreso</span>"
            f'<i style="--w:{row["revenue_share"] * 100:.1f}%"></i></div></div>'
            f'<p class="seg-why"><span class="lab2">Por qué este nombre</span>'
            f"{html.escape(rule_in_words(archetype(name)))}.</p>"
            f"{profile_svg(_z(row))}"
            f'<div class="seg-act"><span class="lab2">Acción prioritaria</span>'
            f"<p>{html.escape(first.action)}</p>"
            f'<p class="kpi">Se mide con: {html.escape(first.kpi)}</p></div>'
            "</article>"
        )
    return "\n".join(cards)


def data_tables(inp: Inputs) -> dict[str, str]:
    seg, lrfmc = inp.profiles, inp.lrfmc
    overall = lrfmc[list(LRFMC)].median()
    fmt_metric = {
        "L": lambda v: num(v),
        "R": lambda v: num(v, 1),
        "F": lambda v: num(v),
        "M": lambda v: num(v),
        "C": lambda v: pct(v, 0),
    }
    metric_names = {
        "L": "L · Antigüedad (meses)",
        "R": "R · Meses sin volar",
        "F": "F · Vuelos en dos años",
        "M": "M · Kilómetros",
        "C": "C · Precio pagado / tarifa completa",
    }
    shorts = [SHORT_NAMES.get(s, s) for s in seg["segment"]]
    flights = lrfmc["F"]
    bench = inp.benchmark.dropna(subset=["silhouette"])
    stab = inp.metrics["stability"]
    return {
        "table_valor": table_html(
            ["Segmento", "% de los socios", "% del ingreso"],
            [
                [s, pct(p), pct(r)]
                for s, p, r in zip(seg["segment"], seg["pct"], seg["revenue_share"], strict=True)
            ],
            {1, 2},
        ),
        "table_metricas": table_html(
            ["Métrica (mediana)", "Socio típico", *shorts],
            [
                [
                    metric_names[m],
                    fmt_metric[m](overall[m]),
                    *(fmt_metric[m](v) for v in seg[f"{m}_median"]),
                ]
                for m in LRFMC
            ],
            set(range(1, len(shorts) + 2)),
        ),
        "table_cola": table_html(
            ["Socios que vuelan…", "Vuelos en dos años"],
            [
                ["la mitad, como mucho", num(flights.median())],
                ["el 75 %, como mucho", num(flights.quantile(0.75))],
                ["el 95 %, como mucho", num(flights.quantile(0.95))],
                ["el 99 %, como mucho", num(flights.quantile(0.99))],
                ["el que más", num(flights.max())],
            ],
            {1},
        ),
        "table_cuantos": table_html(
            ["Grupos (k)", "Separación (silhouette)", "Grupo más pequeño"],
            [
                [str(k), num(s, 3), pct(m)]
                for k, s, m in zip(
                    inp.k_table["k"],
                    inp.k_table["silhouette"],
                    inp.k_table["min_cluster_pct"],
                    strict=True,
                )
            ],
            {0, 1, 2},
        ),
        "table_fiable": table_html(
            ["Comprobación", "Resultado"],
            [
                [
                    f"Coincidencia entre {stab['seeds']} puntos de partida (ARI medio)",
                    num(stab["seed_ari_mean"], 3),
                ],
                ["Peor par de puntos de partida (ARI mínimo)", num(stab["seed_ari_min"], 3)],
                [
                    f"Entrenando con el {pct(stab['subsample_fraction'], 0)} de los socios "
                    "(ARI medio)",
                    num(stab["subsample_ari_mean"], 3),
                ],
                *[
                    [f"Separación con {MODEL_NAMES.get(m, m)} (silhouette)", num(s, 3)]
                    for m, s in zip(bench["model"], bench["silhouette"], strict=True)
                ],
            ],
            {1},
        ),
        "table_segmentos": table_html(
            ["Segmento", "Acción", "Se mide con"],
            [[s, r.action, r.kpi] for s in seg["segment"] for r in recommendations_for(s)],
            set(),
        ),
    }


def page_text(inp: Inputs) -> dict[str, str]:
    """Valor de cada `[[clave]]` de la plantilla (cifras formateadas y fragmentos HTML)."""
    seg = _segments(inp)
    vip, risk, deals = (
        seg.loc["VIP"],
        seg.loc["En riesgo de abandono"],
        seg.loc["Cazadores de ofertas"],
    )
    overall = inp.lrfmc[list(LRFMC)].median()
    stab = inp.metrics["stability"]
    bench = inp.benchmark.dropna(subset=["silhouette"]).set_index("model")["silhouette"]
    tests = inp.tests.set_index("variable")
    ks = inp.k_table["k"]
    text = {
        "members": num(inp.lineage["output_rows"]),
        "seg_pct_min": pct(inp.profiles["pct"].min(), 0),
        "seg_pct_max": pct(inp.profiles["pct"].max(), 0),
        "vip_pct": pct(vip["pct"], 0),
        "vip_rev": pct(vip["revenue_share"], 0),
        "inactive_pct": pct(risk["pct"] + deals["pct"], 0),
        "inactive_rev": pct(risk["revenue_share"] + deals["revenue_share"], 0),
        "med_L": f"{num(overall['L'])} meses",
        "med_R": f"{num(overall['R'], 1)} meses",
        "risk_R": f"{num(risk['R_median'], 1)} meses",
        "med_F": num(overall["F"]),
        "vip_F": num(vip["F_median"]),
        "med_M": f"{num(overall['M'])} km",
        "deals_C": pct(deals["C_median"], 0),
        "risk_C": pct(risk["C_median"], 0),
        "max_F": num(inp.lrfmc["F"].max()),
        "k": str(inp.metrics["k"]),
        "k_min": str(int(ks.min())),
        "k_max": str(int(ks.max())),
        "map_n": num(min(MAP_POINTS, len(inp.tsne))),
        "map_trust": num(inp.trustworthiness, 2),
        "seeds": str(stab["seeds"]),
        "ari_seed": num(stab["seed_ari_mean"], 2),
        "ari_sub": num(stab["subsample_ari_mean"], 2),
        "subsample_pct": pct(stab["subsample_fraction"], 0),
        "ari_min": num(inp.min_stability_ari, 2),
        "kmeans_sil": num(bench["K-Means"], 2),
        "best_other_sil": num(bench.drop("K-Means").max(), 2),
        "rev_effect": pct(tests.loc["revenue_total", "effect"], 0),
        "segment_steps": segment_steps(inp),
        "segment_cards": segment_cards(inp),
    }
    return text | data_tables(inp)


def chart_data(inp: Inputs) -> dict[str, Any]:
    """Datos de los gráficos (JSON): solo lo que dibuja el JS, ya agregado."""
    profiles, lrfmc = inp.profiles, inp.lrfmc
    segments = list(profiles["segment"])
    flights, scaled = lrfmc["F"].to_numpy(dtype=float), inp.scaled["F"].to_numpy(dtype=float)
    scaled_edges = np.linspace(
        np.floor(scaled.min() * 10) / 10, np.ceil(scaled.max() * 10) / 10, 31
    )
    # El 5 % que más vuela empieza en el primer tramo que cubre el percentil 95 (o en el último).
    first = int(np.searchsorted(RAW_EDGES, np.quantile(flights, 0.95)))
    tail_from = float(RAW_EDGES[min(first, len(RAW_EDGES) - 2)])
    stab = inp.stability
    bench = inp.benchmark.dropna(subset=["silhouette"])
    return {
        "segments": [
            {
                "name": name,
                "short": SHORT_NAMES.get(name, name),
                "pct": round(float(row["pct"]), 4),
                "rev": round(float(row["revenue_share"]), 4),
                "med": {m: round(float(row[f"{m}_median"]), 3) for m in LRFMC},
                # Etiquetas formateadas aquí, con el mismo redondeo que el texto de la página.
                "label": {
                    "pct": pct(row["pct"], 0),
                    "rev": pct(row["revenue_share"], 0),
                    "med": {m: metric_label(m, row[f"{m}_median"]) for m in LRFMC},
                },
            }
            for name, row in profiles.set_index("segment").iterrows()
        ],
        "waffle": {
            "members": largest_remainder(profiles["pct"]),
            "revenue": largest_remainder(profiles["revenue_share"]),
        },
        "overall": {m: round(float(lrfmc[m].median()), 3) for m in LRFMC},
        "overall_label": {m: metric_label(m, lrfmc[m].median()) for m in LRFMC},
        "hist": {
            "raw": {"edges": RAW_EDGES.tolist(), "shares": histogram_shares(flights, RAW_EDGES)},
            "scaled": {
                "edges": scaled_edges.round(3).tolist(),
                "shares": smooth_shares(scaled, scaled_edges, SCALED_BANDWIDTH),
            },
            "median": float(np.median(flights)),
            "max": float(flights.max()),
            "tail_from": tail_from,
            "tail_share": round(float((flights >= tail_from).mean()), 4),
        },
        "k": [
            {"k": int(k), "sil": round(float(s), 4)}
            for k, s in zip(inp.k_table["k"], inp.k_table["silhouette"], strict=True)
        ],
        "chosen_k": int(inp.metrics["k"]),
        "stability": {
            "seeds": stab.loc[stab["method"] == "semillas", "ari"].round(4).tolist(),
            "subsample": stab.loc[stab["method"] == "submuestras", "ari"].round(4).tolist(),
            "threshold": inp.min_stability_ari,
        },
        "bench": [
            {"name": MODEL_NAMES.get(m, m), "sil": round(float(s), 4), "main": m == "K-Means"}
            for m, s in zip(bench["model"], bench["silhouette"], strict=True)
        ],
        "map": map_sample(inp.tsne, segments, MAP_POINTS, inp.random_state),
        "quality": {"min_silhouette": inp.min_silhouette},
    }


def render(template: str, text: dict[str, str], data: dict[str, Any]) -> str:
    """Sustituye cada `[[clave]]` y el marcador de datos. Falla si falta alguna clave."""
    missing = sorted(set(PLACEHOLDER.findall(template)) - set(text))
    if missing:
        raise KeyError(f"Faltan valores para: {', '.join(missing)}")
    if DATA_MARK not in template:
        raise ValueError(f"La plantilla no tiene el marcador de datos {DATA_MARK!r}.")
    page = PLACEHOLDER.sub(lambda m: text[m.group(1)], template)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return page.replace(DATA_MARK, payload)


def build_page(inp: Inputs, template: str | None = None) -> str:
    template = TEMPLATE.read_text(encoding="utf-8") if template is None else template
    return render(template, page_text(inp), chart_data(inp))


def write_page(config: Config, path: Path = OUTPUT) -> Path:
    path.write_text(build_page(load_inputs(config)), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(f"Resumen guardado en {write_page(get_config())}")
