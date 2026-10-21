import pandas as pd
import pytest

from segmentacion.segments.naming import (
    ARCHETYPES,
    SEGMENT_ORDER,
    archetype,
    archetype_scores,
    assignment_margin,
    name_segments,
    ordered_segments,
    perturbation_agreement,
    perturbed_archetypes,
)
from segmentacion.segments.recommendations import (
    RECOMMENDATIONS,
    recommendations_for,
    recommendations_table,
)


def centroids(rows: dict[int, dict[str, float]]) -> pd.DataFrame:
    """Centroides en z-score; las features no indicadas valen 0 (socio medio)."""
    columns = ["L", "R", "F", "M", "C", "T"]
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=columns).fillna(0.0)


# Un centroide «de libro» por arquetipo, en un orden distinto al de ARCHETYPES.
TEXTBOOK = centroids(
    {
        0: {"C": -1.5, "F": -0.8},  # cazadores de ofertas
        1: {"F": 1.3, "M": 1.2, "R": -1.0, "C": 0.3},  # VIP
        2: {"R": 1.0, "T": -0.5, "C": 0.7},  # en riesgo
        3: {"L": -0.9, "R": -0.4, "F": 0.4},  # nuevos
        4: {"L": 0.9, "F": -0.2, "M": -0.1},  # veteranos
    }
)


def test_vip_rule_matches_high_frf_centroid():
    names = name_segments(TEXTBOOK)
    assert names[1] == "VIP"
    # El VIP es, además, el arquetipo con mayor puntuación de ese clúster.
    assert archetype_scores(TEXTBOOK).loc[1].idxmax() == "VIP"


def test_each_textbook_centroid_gets_its_archetype():
    assert name_segments(TEXTBOOK) == {
        0: "Cazadores de ofertas",
        1: "VIP",
        2: "En riesgo de abandono",
        3: "Nuevos con potencial",
        4: "Veteranos ocasionales",
    }


def test_names_are_unique():
    # Dos clústeres casi iguales (ambos «VIP») reciben igualmente nombres distintos.
    similar = centroids(
        {0: {"F": 1.3, "M": 1.2, "R": -1.0}, 1: {"F": 1.2, "M": 1.1, "R": -0.9}, 2: {"C": -1.0}}
    )
    names = name_segments(similar)
    assert len(set(names.values())) == len(names) == 3
    assert "VIP" in names.values()


def test_more_clusters_than_archetypes_raises():
    too_many = centroids({i: {"F": i} for i in range(len(ARCHETYPES) + 1)})
    with pytest.raises(ValueError, match="arquetipos"):
        name_segments(too_many)


def test_scores_are_dot_product_of_weights():
    scores = archetype_scores(centroids({0: {"F": 2.0, "R": 1.0}}))
    # VIP = F·1 + M·1 + R·(−1) + C·0,5 = 2 − 1 = 1
    assert scores.loc[0, "VIP"] == pytest.approx(1.0)


def test_margin_and_order():
    names = name_segments(TEXTBOOK)
    assert (assignment_margin(TEXTBOOK, names) > 0).all()
    assert ordered_segments(names) == list(SEGMENT_ORDER)
    assert ordered_segments({7: "Cazadores de ofertas", 2: "VIP"}) == [
        "VIP",
        "Cazadores de ofertas",
    ]


def test_every_segment_has_recommendations():
    assert set(RECOMMENDATIONS) == set(SEGMENT_ORDER)
    for name in SEGMENT_ORDER:
        recs = recommendations_for(name)
        assert 2 <= len(recs) <= 3
        assert all(r.action and r.kpi for r in recs)
        assert archetype(name).description and archetype(name).rule


def test_recommendations_table_and_unknown_segment():
    table = recommendations_table()
    assert [row["segment"] for row in table][0] == "VIP"
    assert len(table) == sum(len(r) for r in RECOMMENDATIONS.values())
    with pytest.raises(KeyError, match="Sin recomendaciones"):
        recommendations_for("Desconocido")


def test_textbook_names_survive_perturbed_weights():
    import numpy as np

    rng = np.random.default_rng(0)
    expected = name_segments(TEXTBOOK)
    for _ in range(50):
        archetypes = perturbed_archetypes(0.3, rng)
        assert [a.name for a in archetypes] == list(SEGMENT_ORDER)
        assert name_segments(TEXTBOOK, archetypes) == expected


def test_perturbation_agreement_is_a_share():
    assert perturbation_agreement(TEXTBOOK, runs=20, scale=0.3, random_state=0) == 1.0
    # Dos clústeres casi idénticos: pequeñas perturbaciones pueden intercambiar sus nombres.
    ambiguous = centroids({0: {"F": 1.0, "L": -1.0}, 1: {"F": 1.0, "L": -0.99}})
    assert 0 <= perturbation_agreement(ambiguous, runs=50, scale=0.3, random_state=0) <= 1
