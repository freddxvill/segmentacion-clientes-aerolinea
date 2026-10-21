"""Tema Altair y paletas compartidas por todos los gráficos del sitio.

Los colores de las marcas sirven en tema claro y oscuro. El color del texto, la grilla y los
ejes lo sobrescribe `report/styles.scss` según el tema activo de Quarto, por eso aquí se usa
un gris medio legible en ambos fondos.
"""

from collections.abc import Sequence

import altair as alt

THEME_NAME = "segmentacion"

# Paleta categórica validada (CVD-safe en pares adyacentes). El orden es parte de la
# accesibilidad: se asigna siempre en este orden, nunca ciclando.
CATEGORICAL: tuple[str, ...] = (
    "#2a78d6",  # azul
    "#eb6834",  # naranja
    "#1baf7a",  # aguamarina
    "#eda100",  # amarillo
    "#e87ba4",  # magenta
    "#008300",  # verde
    "#6250d6",  # violeta
    "#e34948",  # rojo
)
# Escala secuencial de un solo tono (azul) para magnitudes y mapas de calor.
SEQUENTIAL: tuple[str, ...] = ("#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
# Divergente azul ↔ rojo con punto medio neutro (correlaciones).
DIVERGING: tuple[str, str, str] = ("#2a78d6", "#d9d8d3", "#e34948")
NEUTRAL = "#9a9992"  # contexto: totales, población de referencia, marcas no seleccionadas
INK = "#6f6e69"  # texto por defecto (lo sobrescribe styles.scss)
GRID = "#8a897f33"
MAIN = CATEGORICAL[0]  # serie única

TIER_COLORS: dict[int, str] = {4: CATEGORICAL[0], 5: CATEGORICAL[1], 6: CATEGORICAL[2]}
GENDER_COLORS: dict[str, str] = {"Hombre": CATEGORICAL[0], "Mujer": CATEGORICAL[1]}

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

# Formato numérico y de fechas en español (vega-embed: formatLocale / timeFormatLocale).
FORMAT_LOCALE = {
    "decimal": ",",
    "thousands": ".",
    "grouping": [3],
    "currency": ["", " ¥"],
}
TIME_FORMAT_LOCALE = {
    "dateTime": "%A, %e de %B de %Y, %X",
    "date": "%d/%m/%Y",
    "time": "%H:%M:%S",
    "periods": ["AM", "PM"],
    "days": ["domingo", "lunes", "martes", "miércoles", "jueves", "viernes", "sábado"],
    "shortDays": ["dom", "lun", "mar", "mié", "jue", "vie", "sáb"],
    "months": [
        "enero",
        "febrero",
        "marzo",
        "abril",
        "mayo",
        "junio",
        "julio",
        "agosto",
        "septiembre",
        "octubre",
        "noviembre",
        "diciembre",
    ],  # fmt: skip
    "shortMonths": [
        "ene",
        "feb",
        "mar",
        "abr",
        "may",
        "jun",
        "jul",
        "ago",
        "sep",
        "oct",
        "nov",
        "dic",
    ],  # fmt: skip
}


def segment_colors(segments: Sequence[str]) -> dict[str, str]:
    """Color fijo por segmento (en el orden recibido). Lo usan Altair y Plotly por igual."""
    if len(segments) > len(CATEGORICAL):
        raise ValueError(f"Máximo {len(CATEGORICAL)} segmentos con color propio.")
    return dict(zip(segments, CATEGORICAL, strict=False))


def segment_scale(segments: Sequence[str]) -> alt.Scale:
    """Escala de color Altair para segmentos, coherente con `segment_colors`."""
    colors = segment_colors(segments)
    return alt.Scale(domain=list(colors), range=list(colors.values()))


def plotly_color_map(segments: Sequence[str]) -> dict[str, str]:
    """`color_discrete_map` para Plotly, idéntico al de Altair."""
    return segment_colors(segments)


@alt.theme.register(THEME_NAME, enable=False)
def _theme() -> alt.theme.ThemeConfig:
    axis = {
        "labelColor": INK,
        "titleColor": INK,
        "labelFont": FONT,
        "titleFont": FONT,
        "labelFontSize": 11,
        "titleFontSize": 12,
        "titleFontWeight": "normal",
        "gridColor": GRID,
        "domainColor": GRID,
        "tickColor": GRID,
    }
    return {
        "config": {
            "background": "transparent",
            "font": FONT,
            "view": {"stroke": None, "continuousWidth": 560, "continuousHeight": 300},
            "title": {
                "font": FONT,
                "fontSize": 15,
                "fontWeight": 600,
                "color": INK,
                "subtitleColor": INK,
                "subtitleFontSize": 12,
                "anchor": "start",
                "offset": 12,
            },
            "axis": axis,
            "axisX": {"grid": False},
            "legend": {
                "labelColor": INK,
                "titleColor": INK,
                "labelFont": FONT,
                "titleFont": FONT,
                "titleFontWeight": "normal",
            },
            "header": {"labelColor": INK, "titleColor": INK, "labelFont": FONT},
            "range": {"category": list(CATEGORICAL), "ramp": list(SEQUENTIAL)},
            "bar": {"color": MAIN, "cornerRadiusEnd": 2},
            "line": {"color": MAIN, "strokeWidth": 2},
            "area": {"color": MAIN},
            "point": {"color": MAIN, "size": 64},
            "circle": {"color": MAIN},
            "rect": {"color": MAIN},
            "tick": {"color": MAIN},
            "text": {"color": INK, "font": FONT},
        }
    }


def enable_theme() -> None:
    """Activa el tema y las opciones de render (SVG, locale español, menú reducido)."""
    alt.theme.enable(THEME_NAME)
    alt.renderers.set_embed_options(
        renderer="svg",
        actions={"export": True, "source": False, "compiled": False, "editor": False},
        formatLocale=FORMAT_LOCALE,
        timeFormatLocale=TIME_FORMAT_LOCALE,
    )
