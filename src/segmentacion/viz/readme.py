"""Figura de resultados del README (Fase 11): `python -m segmentacion.viz.readme`.

Une el perfil de los segmentos (heatmap de centroides) y su peso en socios e ingreso en un solo
PNG estático, con los mismos gráficos que el sitio. Fondo blanco: GitHub muestra el PNG igual en
tema claro y oscuro.
"""

from pathlib import Path

import altair as alt
import pandas as pd

from segmentacion.config import PROJECT_ROOT, Config, get_config
from segmentacion.viz import segment_charts as sc
from segmentacion.viz.notebook import static_png
from segmentacion.viz.theme import enable_theme

README_FIGURE = PROJECT_ROOT / "docs" / "img" / "segmentos.png"


def readme_figure(profiles: pd.DataFrame) -> alt.VConcatChart:
    enable_theme()
    return alt.vconcat(
        sc.centroid_heatmap(profiles),
        sc.size_value_chart(profiles),
        spacing=36,
    ).configure(background="white", padding=16)


def export_readme_figure(config: Config, path: Path = README_FIGURE) -> Path:
    profiles = pd.read_csv(config.segment_profiles_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(static_png(readme_figure(profiles).to_dict()))
    return path


if __name__ == "__main__":
    print(f"Figura guardada en {export_readme_figure(get_config())}")
