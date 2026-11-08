"""Resumen interactivo de una página (viz.summary_page): plantilla, texto y datos."""

import json
import re

import numpy as np
import pandas as pd
import pytest

from segmentacion.segments.naming import SEGMENT_ORDER
from segmentacion.viz.fmt import pct
from segmentacion.viz.summary_page import (
    DATA_MARK,
    PLACEHOLDER,
    TEMPLATE,
    Inputs,
    build_page,
    chart_data,
    histogram_shares,
    largest_remainder,
    map_sample,
    metric_label,
    page_text,
    render,
    smooth_shares,
)


@pytest.fixture(scope="module")
def inputs() -> Inputs:
    """Artefactos sintéticos con la forma de los reales (no hace falta el dataset)."""
    rng = np.random.default_rng(0)
    shares, revenue = [0.213, 0.231, 0.217, 0.194, 0.144], [0.559, 0.232, 0.12, 0.06, 0.029]
    profiles = pd.DataFrame(
        [
            {
                "cluster": i,
                "segment": name,
                "members": 10_000 + i,
                "pct": shares[i],
                "revenue_share": revenue[i],
                **{f"z_{key}": float(rng.normal()) for key in "LRFMCT"},
                **dict(
                    zip(
                        [f"{m}_median" for m in "LRFMC"],
                        [40 + i, 2 + i, 10 - i, 9_000 + i, 0.5 + 0.08 * i],
                        strict=True,
                    )
                ),
            }
            for i, name in enumerate(SEGMENT_ORDER)
        ]
    )
    lrfmc = pd.DataFrame(
        {
            "L": rng.uniform(12, 120, 600),
            "R": rng.uniform(0, 24, 600),
            "F": rng.integers(2, 200, 600),
            "M": rng.uniform(100, 60_000, 600),
            "C": rng.uniform(0.3, 1.3, 600),
        }
    )
    return Inputs(
        profiles=profiles,
        lineage={"output_rows": 62_044},
        metrics={
            "k": 5,
            "stability": {
                "seeds": 4,
                "seed_ari_mean": 0.98,
                "seed_ari_min": 0.96,
                "subsample_fraction": 0.8,
                "subsample_ari_mean": 0.97,
            },
        },
        k_table=pd.DataFrame(
            {
                "k": range(2, 11),
                "silhouette": np.linspace(0.29, 0.2, 9),
                "min_cluster_pct": np.linspace(0.4, 0.05, 9),
            }
        ),
        stability=pd.DataFrame(
            {"method": ["semillas"] * 6 + ["submuestras"] * 3, "ari": rng.uniform(0.95, 0.99, 9)}
        ),
        benchmark=pd.DataFrame(
            {
                "model": ["K-Means", "Quintiles LRFMC", "GMM", "Agglomerative (Ward)", "DBSCAN"],
                "silhouette": [0.21, 0.11, 0.12, 0.14, np.nan],
            }
        ),
        tests=pd.DataFrame({"variable": ["revenue_total"], "effect": [0.67]}),
        lrfmc=lrfmc,
        scaled=(lrfmc - lrfmc.mean()) / lrfmc.std(),
        tsne=pd.DataFrame(
            {
                "x": rng.normal(size=500),
                "y": rng.normal(size=500),
                "segment": rng.choice(SEGMENT_ORDER, 500),
            }
        ),
        trustworthiness=0.987,
        min_stability_ari=0.9,
        min_silhouette=0.2,
        random_state=42,
    )


def test_largest_remainder_sums_exactly():
    shares = [0.213, 0.231, 0.217, 0.194, 0.144]
    counts = largest_remainder(shares)
    assert sum(counts) == 100
    assert all(abs(c - s * 100) < 1 for c, s in zip(counts, shares, strict=True))
    assert sum(largest_remainder([1, 1, 1])) == 100  # 33,3 cada uno: el resto va a uno


def test_histograms_are_shares():
    edges = np.arange(0, 121, 4)
    shares = histogram_shares([1, 5, 300], edges)  # 300 se recorta al último tramo
    assert sum(shares) == pytest.approx(1, abs=1e-4)  # redondeo a 5 decimales
    assert shares[-1] == pytest.approx(1 / 3, abs=1e-4)
    smooth = smooth_shares([0.0, 0.0, 1.0], np.linspace(-2, 3, 31), bandwidth=0.2)
    assert sum(smooth) == pytest.approx(1, abs=1e-4)
    assert all(s >= 0 for s in smooth)


def test_map_sample_is_stratified_and_normalized(inputs):
    points = map_sample(inputs.tsne, SEGMENT_ORDER, n=200, seed=0)
    xy = np.array([[p[0], p[1]] for p in points])
    assert xy.min() >= 0 and xy.max() <= 1
    assert {p[2] for p in points} <= set(range(len(SEGMENT_ORDER)))
    original = inputs.tsne["segment"].value_counts(normalize=True)
    sampled = pd.Series([SEGMENT_ORDER[p[2]] for p in points]).value_counts(normalize=True)
    assert (sampled - original).abs().max() < 0.03


def test_metric_label_matches_site_format():
    assert metric_label("M", 10_200) == "10.200 km"
    assert metric_label("R", 3.5) == "3,5 meses"
    assert metric_label("C", 0.505) == pct(0.505, 0)


def test_template_uses_exactly_the_text_keys(inputs):
    used = set(PLACEHOLDER.findall(TEMPLATE.read_text(encoding="utf-8")))
    assert used == set(page_text(inputs))


def test_build_page_fills_text_and_data(inputs):
    page = build_page(inputs)
    assert not PLACEHOLDER.findall(page)
    assert DATA_MARK not in page
    payload = re.search(r'<script type="application/json" id="data">(.*?)</script>', page, re.S)
    data = json.loads(payload.group(1))
    assert [s["name"] for s in data["segments"]] == list(SEGMENT_ORDER)
    assert sum(data["waffle"]["members"]) == sum(data["waffle"]["revenue"]) == 100
    assert len(data["hist"]["raw"]["shares"]) == len(data["hist"]["scaled"]["shares"])
    assert [b["name"] for b in data["bench"] if b["main"]] == ["K-Means"]  # DBSCAN (NaN) fuera
    assert "62.044 socios" in page and pct(0.559, 0) in page


def test_chart_labels_use_python_rounding(inputs):
    data = chart_data(inputs)
    vip = data["segments"][0]
    assert vip["label"]["pct"] == pct(vip["pct"], 0)
    assert vip["label"]["med"]["C"] == metric_label("C", 0.5)


def test_render_escapes_json_and_reports_missing_keys():
    page = render(f"<script>{DATA_MARK}</script>[[a]]", {"a": "x"}, {"s": "</script>"})
    assert "<\\/script>" in page and page.endswith("x")
    with pytest.raises(KeyError, match="b"):
        render(f"{DATA_MARK}[[b]]", {}, {})
    with pytest.raises(ValueError, match="marcador"):
        render("[[a]]", {"a": "x"}, {})
