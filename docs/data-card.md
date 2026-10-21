# Data card: `flight.csv` (programa de viajero frecuente)

> **Estado:** descargado y verificado (2026-10-20). El bloque **Metadatos** lo genera de forma
> automática `uv run poe download` (o `uv run poe verify-data`); no lo edites a mano.

## Metadatos

<!-- META:START -->
| Campo | Valor |
|---|---|
| Fuente | https://www.kaggle.com/competitions/sa-customer-segmentation |
| Archivos fuente (filas) | `flight_train.csv` (55,000) + `flight_test.csv` (7,988) |
| Archivo | `flight.csv` |
| Fecha de verificación | 2026-10-20 |
| Tamaño | 8.08 MB |
| SHA-256 | `ed3c71576cab60a1643153098fd7ff87203360dafb543a133f3d998b89522044` |
| Codificación | `utf-8-sig` |
| Filas | 62,988 |
| Columnas | 23 |
| Columnas no esperadas | ninguna |
<!-- META:END -->

## Procedencia y licencia

- **Publicación:** competición de Kaggle
  [`sa-customer-segmentation`](https://www.kaggle.com/competitions/sa-customer-segmentation).
- **Origen:** datos anonimizados de socios de un programa de viajero frecuente (FFP) de una
  aerolínea. **La procedencia original (aerolínea, país, período) no está documentada por quien
  los publicó.** El dataset circula desde hace años en cursos y publicaciones académicas sobre el
  modelo LRFMC (p. ej., Wang & Huang, 2021).
- **Licencia:** _pendiente de confirmar_ en las reglas de la competición. Mientras no se
  confirme que permite redistribuir los datos, el CSV **no se versiona** en el repositorio.

## Diccionario de datos

| Grupo | Columna | Descripción | Uso en el proyecto |
|---|---|---|---|
| Socio | `MEMBER_NO` | ID del socio | Clave |
| | `FFP_DATE` | Fecha de alta en el programa | **L** |
| | `FIRST_FLIGHT_DATE` | Fecha del primer vuelo | EDA |
| | `GENDER` | Género | Perfilado |
| | `FFP_TIER` | Nivel del programa | Validación externa |
| | `WORK_CITY` | Ciudad de trabajo | EDA / perfilado |
| | `WORK_PROVINCE` | Provincia de trabajo | EDA / perfilado |
| | `WORK_COUNTRY` | País de trabajo | EDA / perfilado |
| | `AGE` | Edad | Perfilado |
| Vuelos | `LOAD_TIME` | Fecha de corte de la ventana de observación | **L** |
| | `FLIGHT_COUNT` | Nº de vuelos en la ventana | **F** |
| | `SUM_YR_1` | Ingreso por tarifas, año 1 | Limpieza / perfilado |
| | `SUM_YR_2` | Ingreso por tarifas, año 2 | Limpieza / perfilado |
| | `SEG_KM_SUM` | Km totales volados | **M** |
| | `LAST_FLIGHT_DATE` | Fecha del último vuelo | EDA |
| | `LAST_TO_END` | Días desde el último vuelo hasta `LOAD_TIME` | **R** |
| | `AVG_INTERVAL` | Intervalo promedio entre vuelos | Perfilado |
| | `MAX_INTERVAL` | Intervalo máximo entre vuelos | Perfilado |
| | `avg_discount` | Coeficiente de descuento promedio | **C** |
| Puntos | `BP_SUM` | Puntos base acumulados | Perfilado |
| | `EXCHANGE_COUNT` | Nº de canjes | Perfilado |
| | `Points_Sum` | Puntos totales | Perfilado |
| | `Point_NotFlight` | Puntos no originados en vuelos | Perfilado |

## Archivos de la competición y cómo se usan

| Archivo | Filas | Uso |
|---|---|---|
| `flight_train.csv` | 55.000 | Fuente → `flight.csv` |
| `flight_test.csv` | 7.988 | Fuente → `flight.csv` |
| `sample.csv` | 7.988 | **No se usa** (plantilla de envío de la competición: `MEMBER_NO`, `FLIGHT_COUNT`) |

- La competición separa a los socios en train y test **sin solape de `MEMBER_NO`**. Juntos suman
  62.988 socios, el tamaño conocido del dataset LRFMC.
- **El test no es una muestra aleatoria:** solo contiene socios con `FLIGHT_COUNT` entre 2 y 10.
  Segmentar solo con el train excluiría a clientes de baja frecuencia, por eso `flight.csv` es la
  **concatenación textual** de ambos archivos (sin re-serializar valores).

## Hallazgos de la verificación inicial (entrada para la Fase 2)

- `LOAD_TIME` es **única**: `3/31/2014` (fecha de corte de la ventana de observación). ✅
- Las fechas vienen en formato **`M/D/YYYY`** (p. ej. `11/2/2006`).
- Nulos: `WORK_PROVINCE` 3.248 · `WORK_CITY` 2.269 · `SUM_YR_1` 551 · `AGE` 420 ·
  `SUM_YR_2` 138 · `WORK_COUNTRY` 26 · `GENDER` 3.
- Placeholders de texto que deben tratarse como nulos:
  - `WORK_CITY`: `.` (667), `-` (79), `。` (15), `0` (4).
  - `WORK_PROVINCE`: `.` (930), `-` (262), `` ` `` (20), `0` (14).
  - `WORK_COUNTRY`: algunos caracteres chinos sueltos (`中`, `北`…) en lugar de un código de país.
- `GENDER`: Male 48.134 · Female 14.851.
- `FFP_TIER`: 4 → 58.066 · 5 → 3.409 · 6 → 1.513 (muy desbalanceado).
- `AGE`: rango [6, 110] → se aplica la regla de [18, 100] del spec.
- `avg_discount`: rango [0, 1.5].

## Supuestos por confirmar en el EDA (Fase 3)

- [ ] Semántica de `avg_discount` (¿1.0 = tarifa completa? ¿0 = vuelo de canje?).
- [x] `LOAD_TIME` es una única fecha de corte para todos los socios.
- [ ] Rango válido de `AGE` y valores de `GENDER`.
- [x] Formato de las fechas (`M/D/YYYY`).
- [ ] Licencia de la competición (¿permite redistribuir?).

## Limpieza (Fase 2)

`uv run poe clean` genera `data/interim/flight_clean.parquet` y `cleaning_lineage.json`.

- **62.988 → 62.044 socios** (944 eliminados, 98,5% retenido):
  689 con ingreso nulo y 255 con ingreso 0 que sí volaron (canjes).
- Anulados sin eliminar la fila: 421 `LAST_FLIGHT_DATE` = `2014/2/29` (inexistente; `LAST_TO_END`
  sigue siendo válido), placeholders de ubicación y 49 edades fuera de [18, 100].
- Detalle de reglas: `docs/plan-data.md` §3.3.
