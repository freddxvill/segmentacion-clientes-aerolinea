"""PCA 3D en Plotly (Altair no soporta 3D).

Usa los mismos colores por clúster que Altair (`theme.plotly_color_map`). El HTML carga
plotly.js desde CDN para no incrustar ~4 MB de JavaScript en cada página del sitio.
"""

import pandas as pd
import plotly.graph_objects as go

from segmentacion.viz.fmt import pct
from segmentacion.viz.projection_charts import _segment_order, cluster_names
from segmentacion.viz.theme import FONT, INK, plotly_color_map

GRID = "rgba(138, 137, 127, 0.25)"


def pca_3d_figure(points: pd.DataFrame, explained: pd.DataFrame) -> go.Figure:
    """Nube PCA 3D rotable, una traza por clúster (clic en la leyenda para ocultarla)."""
    names = cluster_names(points["cluster"]).to_numpy()
    order = _segment_order(pd.Series(names))
    colors = plotly_color_map(order)
    variance = explained.set_index("component")["variance"]
    fig = go.Figure()
    for segment in order:
        part = points[names == segment]
        fig.add_trace(
            go.Scatter3d(
                x=part["PC1"],
                y=part["PC2"],
                z=part["PC3"],
                mode="markers",
                name=segment,
                marker={"size": 2.5, "color": colors[segment], "opacity": 0.7},
                hovertemplate=f"{segment}<br>PC1 %{{x:.2f}}<br>PC2 %{{y:.2f}}"
                "<br>PC3 %{z:.2f}<extra></extra>",
            )
        )
    axis = {
        "backgroundcolor": "rgba(0,0,0,0)",
        "gridcolor": GRID,
        "zerolinecolor": GRID,
        "showspikes": False,
    }
    fig.update_layout(
        title={
            "text": f"Segmentos en PCA 3D · {pct(variance.iloc[:3].sum())} de la varianza",
            "x": 0,
            "font": {"size": 15},
        },
        font={"family": FONT, "color": INK},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": -0.05, "itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        height=520,
        scene={
            "xaxis": {**axis, "title": f"PC1 ({pct(variance['PC1'])})"},
            "yaxis": {**axis, "title": f"PC2 ({pct(variance['PC2'])})"},
            "zaxis": {**axis, "title": f"PC3 ({pct(variance['PC3'])})"},
            "camera": {"eye": {"x": 1.6, "y": 1.4, "z": 0.8}},
        },
    )
    return fig


def figure_html(fig: go.Figure) -> str:
    """Fragmento HTML responsivo con plotly.js desde CDN."""
    return fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        config={"responsive": True, "displaylogo": False},
        default_width="100%",
    )
