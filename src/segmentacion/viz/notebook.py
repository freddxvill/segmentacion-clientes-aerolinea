"""Renderer de Altair para los notebooks de `notebooks/`.

GitHub muestra los notebooks sin ejecutar JavaScript, así que un gráfico Altair normal aparece en
blanco. Este renderer emite un único `text/html` con un PNG estático dentro del contenedor:
donde hay JavaScript (JupyterLab, VS Code, nbviewer) vega-embed lo reemplaza por el gráfico
interactivo; donde no, queda la imagen.
"""

import base64
import json
import uuid
from typing import Any

import altair as alt
import vl_convert as vlc
from altair.utils.html import spec_to_html
from altair.vegalite.v6 import display

from segmentacion.viz.theme import FORMAT_LOCALE, TIME_FORMAT_LOCALE, enable_theme

RENDERER_NAME = "notebook"
STATIC_WIDTH = 760  # ancho del PNG cuando el gráfico usa width="container"


def _fixed_width(spec: Any) -> Any:
    """Copia del spec con `width: "container"` → ancho fijo (vl-convert no tiene contenedor)."""
    if isinstance(spec, dict):
        return {
            k: STATIC_WIDTH if k == "width" and v == "container" else _fixed_width(v)
            for k, v in spec.items()
        }
    if isinstance(spec, list):
        return [_fixed_width(v) for v in spec]
    return spec


def static_png(spec: dict) -> bytes:
    """PNG del gráfico con formato numérico y de fechas en español."""
    return vlc.vegalite_to_png(
        _fixed_width(spec),
        scale=1.5,
        format_locale=FORMAT_LOCALE,
        time_format_locale=TIME_FORMAT_LOCALE,
    )


def _renderer(spec: dict, **metadata: Any) -> dict[str, str]:
    div_id = f"altair-{uuid.uuid4().hex}"
    embed_options = {
        **alt.renderers.options.get("embed_options", {}),
        **metadata.get("embed_options", {}),
    }
    html = spec_to_html(
        spec,
        mode="vega-lite",
        vega_version=display.VEGA_VERSION,
        vegaembed_version=display.VEGAEMBED_VERSION,
        vegalite_version=display.VEGALITE_VERSION,
        output_div=div_id,
        embed_options=embed_options,
        json_kwds={"ensure_ascii": False},
        fullhtml=False,
        template="universal",
    )
    png = base64.b64encode(static_png(spec)).decode("ascii")
    image = (
        f'<img src="data:image/png;base64,{png}" style="max-width:100%" '
        f'alt="{json.dumps(_title(spec), ensure_ascii=False)[1:-1]}">'
    )
    # El PNG ocupa el contenedor hasta que vega-embed dibuja el gráfico interactivo.
    html = html.replace(f'<div id="{div_id}"></div>', f'<div id="{div_id}">{image}</div>')
    return {"text/html": html}


def _title(spec: dict) -> str:
    title = spec.get("title", "Gráfico")
    if isinstance(title, dict):
        title = title.get("text", "Gráfico")
    return str(title)


alt.renderers.register(RENDERER_NAME, _renderer)


def enable_notebook_renderer() -> None:
    """Tema del proyecto + renderer interactivo con PNG de respaldo."""
    enable_theme()
    alt.renderers.enable(RENDERER_NAME, embed_options=alt.renderers.options["embed_options"])
