# Segmentación avanzada de clientes con LRFMC

Segmentación de los socios de un programa de viajero frecuente de una aerolínea con el modelo
**LRFMC** (Length, Recency, Frequency, Monetary, Coefficient) y K-Means. Los resultados se
publican como sitio Quarto en GitHub Pages.

> 🚧 Proyecto en construcción. El plan completo está en `docs/plan-data.md`.

## Requisitos

- [uv](https://docs.astral.sh/uv/) y Python 3.13
- [Quarto](https://quarto.org/) ≥ 1.5
- Cuenta de Kaggle con la API configurada y las reglas de la competición
  [`sa-customer-segmentation`](https://www.kaggle.com/competitions/sa-customer-segmentation) aceptadas

## Puesta en marcha

```bash
uv sync                  # instala dependencias
uv run poe download      # descarga data/raw/flight.csv y actualiza docs/data-card.md
uv run poe test          # tests unitarios
uv run poe test-data     # tests sobre el dataset real
uv run poe notebooks     # re-ejecuta los notebooks de notebooks/
uv run poe preview       # previsualiza el sitio Quarto
```

## Notebooks

| Notebook | Contenido |
|---|---|
| [`01_datos_y_limpieza.ipynb`](notebooks/01_datos_y_limpieza.ipynb) | Carga, diagnóstico de calidad y limpieza (62.988 → 62.044 socios) |
| [`02_eda.ipynb`](notebooks/02_eda.ipynb) | Análisis exploratorio de las métricas LRFMC y del perfil de los socios |

Los gráficos son interactivos en JupyterLab, VS Code o nbviewer; en GitHub se ven como imagen.

Si descargas el archivo manualmente, cópialo en `data/raw/flight.csv` y ejecuta
`uv run poe verify-data`.

Para ver todas las tareas disponibles: `uv run poe`.
