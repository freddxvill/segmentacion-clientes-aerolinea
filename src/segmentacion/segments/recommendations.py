"""Acciones de marketing por segmento, cada una con el KPI que mide su efecto (Fase 9).

Son propuestas a validar con experimentos (grupo de control incluido), no resultados. Cada
acción se apoya en el perfil del segmento que justifica el nombre (ver `naming.ARCHETYPES`).
"""

from dataclasses import dataclass

from segmentacion.segments.naming import SEGMENT_ORDER


@dataclass(frozen=True)
class Recommendation:
    action: str
    kpi: str


RECOMMENDATIONS: dict[str, tuple[Recommendation, ...]] = {
    "VIP": (
        Recommendation(
            "Reconocimiento sin descuento: upgrades, sala VIP y embarque prioritario en sus rutas "
            "habituales.",
            "Tasa de retención a 12 meses y vuelos por socio frente a un grupo de control.",
        ),
        Recommendation(
            "Acceso anticipado a ventas y un canal de atención dedicado.",
            "NPS del segmento y reclamaciones por cada 1.000 vuelos.",
        ),
        Recommendation(
            "Alerta temprana cuando un VIP pasa más de 3 meses sin volar, con contacto personal.",
            "% de VIP que pasan al segmento «En riesgo de abandono» en el trimestre.",
        ),
    ),
    "Nuevos con potencial": (
        Recommendation(
            "Fast track de nivel: millas bonus al alcanzar un número de vuelos en los primeros "
            "6 meses.",
            "% de socios que suben de nivel (FFP_TIER) en 12 meses.",
        ),
        Recommendation(
            "Bono por vuelo adicional y ofertas de rutas nuevas desde su ciudad.",
            "Vuelos por socio y % que pasa a «VIP» frente a un grupo de control.",
        ),
    ),
    "Veteranos ocasionales": (
        Recommendation(
            "Campañas de ocio y temporada alineadas con sus fechas habituales de viaje.",
            "Vuelos por socio al año y tasa de conversión de la campaña.",
        ),
        Recommendation(
            "Recordatorio de los puntos acumulados y canjes fáciles para mantener el vínculo.",
            "Tasa de canje y recencia media del segmento.",
        ),
    ),
    "En riesgo de abandono": (
        Recommendation(
            "Win-back con millas extra de caducidad corta (por ejemplo, 90 días).",
            "% de socios reactivados (al menos un vuelo) en 90 días.",
        ),
        Recommendation(
            "Encuesta breve de satisfacción y contacto de servicio al cliente.",
            "Tasa de respuesta y distribución de los motivos de abandono.",
        ),
        Recommendation(
            "Ofrecer upgrades o flexibilidad en lugar de descuentos: pagaban tarifas casi "
            "completas.",
            "Ingreso recuperado por socio contactado frente a un grupo de control.",
        ),
    ),
    "Cazadores de ofertas": (
        Recommendation(
            "Alertas de tarifas promocionales en sus rutas habituales, solo en temporada baja.",
            "Tasa de reactivación en 90 días e ingreso por km de los vuelos reactivados.",
        ),
        Recommendation(
            "Paquetes (vuelo + servicios) en lugar de descuentos generales sobre la tarifa.",
            "Margen por pasajero frente a una promoción de descuento directo.",
        ),
    ),
}


def recommendations_for(name: str) -> tuple[Recommendation, ...]:
    if name not in RECOMMENDATIONS:
        raise KeyError(f"Sin recomendaciones para el segmento {name!r}.")
    return RECOMMENDATIONS[name]


def recommendations_table() -> list[dict[str, str]]:
    """Todas las acciones en el orden de los segmentos: `[segment, action, kpi]`."""
    return [
        {"segment": name, "action": r.action, "kpi": r.kpi}
        for name in SEGMENT_ORDER
        for r in RECOMMENDATIONS[name]
    ]
