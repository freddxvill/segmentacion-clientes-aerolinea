# Segmentación avanzada de clientes con LRFMC

Segmentación de los socios de un programa de viajero frecuente de una aerolínea con el modelo
**LRFMC** (Length, Recency, Frequency, Monetary, Coefficient) y K-Means. Los resultados se
publican como sitio Quarto en GitHub Pages.

> 🚧 Proyecto en construcción.

## Requisitos

- [uv](https://docs.astral.sh/uv/) y Python 3.13
- [Quarto](https://quarto.org/) ≥ 1.5
- Cuenta de Kaggle (para descargar los datos, ver [Datos](#datos))

## Datos

El dataset **no está incluido en el repositorio**: debe descargarse desde la competición de Kaggle
[**sa-customer-segmentation**](https://www.kaggle.com/competitions/sa-customer-segmentation/data)
(datos anonimizados de 62.988 socios de un programa de viajero frecuente; requiere iniciar sesión
y aceptar las reglas de la competición).

**Opción A: descarga automática** (requiere la API de Kaggle configurada en
`~/.kaggle/kaggle.json`):

```bash
uv run poe download
```

**Opción B: descarga manual**

1. Entra en <https://www.kaggle.com/competitions/sa-customer-segmentation/data>, acepta las reglas
   y descarga los archivos (*Download All*).
2. Copia `flight_train.csv` y `flight_test.csv` en `data/raw/` (`sample.csv` no se usa).
3. Ejecuta `uv run poe verify-data`: une ambos archivos en `data/raw/flight.csv`, verifica
   tamaño, columnas y socios únicos, y actualiza `docs/data-card.md`.

Ficha completa del dataset (diccionario, procedencia y hallazgos): [`docs/data-card.md`](docs/data-card.md).

## Puesta en marcha

```bash
uv sync                  # instala dependencias
uv run poe download      # descarga los datos de Kaggle (ver Datos)
uv run poe clean         # limpieza → data/interim/flight_clean.parquet
uv run poe features      # features LRFMC → data/processed/lrfmc.parquet
uv run poe test          # tests unitarios
uv run poe test-data     # tests sobre el dataset real
uv run poe notebooks     # re-ejecuta los notebooks de notebooks/
uv run poe preview       # previsualiza el sitio Quarto
```

Para ver todas las tareas disponibles: `uv run poe`.

## Notebooks

| Notebook | Contenido |
|---|---|
| [`01_datos_y_limpieza.ipynb`](notebooks/01_datos_y_limpieza.ipynb) | Carga, diagnóstico de calidad y limpieza (62.988 → 62.044 socios) |
| [`02_eda.ipynb`](notebooks/02_eda.ipynb) | Análisis exploratorio de las métricas LRFMC y del perfil de los socios |
| [`03_features_y_transformacion.ipynb`](notebooks/03_features_y_transformacion.ipynb) | Cálculo de las features LRFMC y de las variables de perfilado |

Los gráficos son interactivos en JupyterLab, VS Code o nbviewer; en GitHub se ven como imagen.
