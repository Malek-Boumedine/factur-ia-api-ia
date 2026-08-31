"""Instrumentation OpenTelemetry : traces distribuées et métriques Prometheus.

Port du ``core/telemetry.py`` de l'API data, adapté à ce service : mêmes
interrupteurs, même mécanique, mêmes garanties. Deux différences : pas
d'instrumentation SQLAlchemy (aucune base applicative ici — le SQLite de MLflow
n'est pas un moteur à instrumenter), et des **métriques de qualité
d'extraction** en plus des métriques HTTP (cf. plus bas).

Deux interrupteurs indépendants, tous deux désactivés par défaut — si aucun
n'est actif, ``setup_telemetry`` ne fait rien (ni import du SDK ni
instrumentation), aucun changement de comportement en local ou en CI :

- ``OTEL_ENABLED`` : **traces** des requêtes HTTP entrantes (FastAPI) et des
  appels httpx sortants (callback API data, Groq — dont le SDK repose sur
  httpx), exportées en OTLP/HTTP vers ``OTEL_EXPORTER_OTLP_ENDPOINT``. Un
  collector injoignable ne fait jamais tomber l'API : l'export part d'un
  thread de fond et échoue en silence.
- ``OTEL_METRICS_ENABLED`` : **métriques** des mêmes instrumentations
  (histogrammes de durée HTTP entrant/sortant) plus les métriques de qualité
  d'extraction, exposées au format Prometheus sur ``/metrics``. En production,
  ``/metrics`` ne doit pas être public : ne pas activer tel quel sur Cloud Run.

Métriques de qualité d'extraction — la raison d'être de ce module ici : MLflow
(``core/monitoring.py``) trace *chaque* extraction avec son identifiant pour le
diagnostic, Prometheus *agrège* pour l'alerte (dégradation du score moyen, taux
d'échec élevé). Quatre instruments, alimentés par ``record_extraction_quality``
depuis ``monitoring.py`` — no-op tant que les métriques sont désactivées :

- ``extraction_total`` (compteur) : une extraction par ``statut``
  (succès/échec) — l'alerte de taux d'échec se calcule dessus ;
- ``extraction_score_confiance`` (histogramme 0-1) : score global, **succès
  seulement** — un échec a un score conventionnellement à 0 (marqueur du
  contrat), le compter ici ferait chuter la moyenne pour une raison qui n'est
  pas une dérive de qualité ; le compteur par statut porte déjà ce signal ;
- ``extraction_taux_champs_reconnus`` (histogramme 0-1) : part des champs
  extraits avec une confiance suffisante, succès seulement (même raison) ;
- ``extraction_duree_seconds`` (histogramme) : durée du pipeline complet.

Labels à cardinalité strictement bornée : ``statut`` (2 valeurs),
``type_document`` (5 : les 4 du contrat + ``non_calcule``), ``mode_extraction``
(3). **Jamais** ``id_document`` (cardinalité non bornée, et fuite d'identifiant
— c'est le rôle de MLflow) ni le modèle LLM (question MLflow, tracée par run).

Limite assumée du modèle pull en serverless : sur Cloud Run avec mise à
l'échelle à zéro, ces métriques vivent en mémoire d'instance — chaque
recyclage remet les compteurs à zéro et les valeurs jamais scrapées sont
perdues ; avec plusieurs instances, un scrape sans affinité n'en voit qu'une.
``rate()`` de Prometheus tolère les remises à zéro, mais la réponse propre en
serverless est un export *push* (OTLP vers un collector) — à traiter avec la
stack d'observabilité mutualisée, hors de ce dépôt.

Les conventions sémantiques HTTP *stables* sont adoptées via la variable
standard ``OTEL_SEMCONV_STABILITY_OPT_IN=http`` (surchargeable par
l'environnement) : sans elles, l'histogramme HTTP n'a pas de label
``http.route`` et aucun découpage par route n'est possible.

Garanties sur les données sensibles — aucun secret ni donnée de facture ne
part dans les spans ni dans les métriques :

- **headers jamais capturés** (comportement par défaut des instrumentations ;
  ne jamais définir les variables ``OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_*``) :
  ``X-OCR-Secret-Token`` (callback) et ``Authorization`` (clé Groq) restent
  hors traces ;
- **corps de requête/réponse jamais capturés** (non supporté par ces
  instrumentations) : ni le document reçu, ni le payload du callback, ni les
  échanges avec Groq ;
- **query strings retirées des URLs** par les hooks ci-dessous : les spans ne
  gardent que schéma + hôte + chemin ;
- **labels de métriques à cardinalité bornée, sans ID réel** : route templatée,
  méthode, code de statut côté serveur ; hôte et port de destination côté
  client (jamais le chemin ni la query) ; étiquettes catégorielles bornées
  côté qualité.

Le scraping (``src/scraping/``) n'est jamais instrumenté : l'instrumentation
httpx est globale au *processus*, mais elle n'est posée que par
``setup_telemetry(app)`` dans ``main.py`` — le batch tourne dans son propre
processus (``python -m src.scraping``) qui n'importe ni ``main.py`` ni ce
module. Ses requêtes vers les sites externes restent hors télémétrie.
"""

import logging
import os
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI

from src.core.config import settings

if TYPE_CHECKING:
    from opentelemetry.metrics import Counter, Histogram
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import Span

# Instruments de qualité d'extraction, créés par `_build_meter_provider` quand
# les métriques sont activées ; sinon None et `record_extraction_quality` ne
# fait rien.
_extraction_counter: "Counter | None" = None
_score_histogram: "Histogram | None" = None
_champs_histogram: "Histogram | None" = None
_duree_histogram: "Histogram | None" = None

# Buckets des histogrammes de qualité : les défauts OTel sont calibrés pour des
# latences (0, 5, 10, 25… 10000), inutilisables pour des scores 0-1. Un pas de
# 0.1 suffit pour situer une dérive ; les bornes 0 et 1 sont implicites
# (buckets extrêmes de l'histogramme Prometheus).
_SCORE_BUCKETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Buckets de durée du pipeline, calibrés sur ses deux régimes : PDF natif +
# LLM (quelques secondes) et OCR + LLM (dizaines de secondes). Le défaut OTel
# plafonne trop bas pour l'OCR.
_DUREE_BUCKETS = [1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]

# Sondes Cloud Run appelées en continu (/health, /ready) et endpoint Prometheus
# (/metrics, scrapé toutes les 15 s) : aucune valeur d'observabilité, exclus du
# tracing ET des métriques (l'exclusion ASGI coupe les deux avant tout
# enregistrement). Sondées en continu, ces routes représenteraient l'essentiel
# du volume et écraseraient les statistiques du trafic réel. Regex cherchées
# (re.search) dans « scheme://host/chemin » — sans query string.
EXCLUDED_URLS = r"/health$,/ready$,/metrics$"

# Variables standard OpenTelemetry relayées vers os.environ : le SDK et les
# instrumentations ne lisent pas le .env (chargé par pydantic-settings sans
# export dans l'environnement).
_OTEL_ENV_VARS = (
    "OTEL_SERVICE_NAME",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_TRACES_EXPORTER",
    "OTEL_TRACES_SAMPLER",
    "OTEL_TRACES_SAMPLER_ARG",
    "OTEL_SEMCONV_STABILITY_OPT_IN",
)


def scrub_url(url: object) -> str:
    """Réduit une URL à « schéma://hôte/chemin » : sans credentials, query ni
    fragment."""
    parts = urlsplit(str(url))
    netloc = parts.netloc.rpartition("@")[2]
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def scrub_span_url_attributes(span: "Span") -> None:
    """Écrase les attributs d'URL d'un span par leur version sans query string.

    Couvre les deux générations de conventions sémantiques HTTP : ``http.url``
    (anciennes) et ``url.full`` / ``url.query`` (stables, activées par
    ``OTEL_SEMCONV_STABILITY_OPT_IN=http``).
    """
    attributes = getattr(span, "attributes", None) or {}
    for key in ("http.url", "url.full"):
        value = attributes.get(key)
        if value is not None:
            span.set_attribute(key, scrub_url(value))
    if "url.query" in attributes:
        span.set_attribute("url.query", "")


def record_extraction_quality(
    *,
    statut: str,
    type_document: str,
    mode_extraction: str,
    score_confiance: float | None,
    taux_champs_reconnus: float | None,
    duree_secondes: float,
) -> None:
    """Alimente les métriques Prometheus de qualité pour une extraction.

    Appelée par ``monitoring.py`` (point d'appel unique du pipeline), une fois
    par extraction, succès comme échec. No-op si les métriques sont
    désactivées (les instruments valent alors ``None``).

    Args:
        statut: ``succes`` ou ``echec`` (marqueur ``score_confiance = 0``).
        type_document: type suggéré par l'IA, ou ``non_calcule`` (échec) —
            valeurs bornées, jamais de texte libre.
        mode_extraction: chemin d'extraction (``pdf_natif``, ``ocr``,
            ``inconnu``).
        score_confiance: score global 0-1, ``None`` sur un échec — le score
            conventionnel à 0 n'entre pas dans l'histogramme (cf. docstring de
            module), seul le compteur par statut porte l'échec.
        taux_champs_reconnus: part des champs à confiance suffisante, ``None``
            sur un échec (même raison).
        duree_secondes: durée du pipeline complet, callback compris.
    """
    qualite = {"type_document": type_document, "mode_extraction": mode_extraction}

    if _extraction_counter is not None:
        _extraction_counter.add(1, {"statut": statut, **qualite})
    if _score_histogram is not None and score_confiance is not None:
        _score_histogram.record(score_confiance, qualite)
    if _champs_histogram is not None and taux_champs_reconnus is not None:
        _champs_histogram.record(taux_champs_reconnus, qualite)
    if _duree_histogram is not None:
        _duree_histogram.record(
            duree_secondes, {"statut": statut, "mode_extraction": mode_extraction}
        )


def _server_request_hook(span: "Span", scope: dict[str, Any]) -> None:
    """Hook FastAPI/ASGI : scrubbing d'URL des spans serveur."""
    scrub_span_url_attributes(span)


def _client_request_hook(span: "Span", request_info: Any) -> None:
    """Hook httpx (clients sync) : scrubbing d'URL des spans sortants."""
    scrub_span_url_attributes(span)


async def _async_client_request_hook(span: "Span", request_info: Any) -> None:
    """Hook httpx (clients async) : scrubbing d'URL des spans sortants."""
    scrub_span_url_attributes(span)


def _build_tracer_provider() -> "TracerProvider":
    """Construit le pipeline de traces : provider + export OTLP ou console."""
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SpanExporter,
    )

    # Un collector absent ou injoignable ne doit ni faire tomber l'API ni
    # remplir les logs : les échecs d'export sont réduits au silence.
    # L'activation se vérifie avec OTEL_TRACES_EXPORTER=console.
    for noisy_logger in (
        "opentelemetry.exporter.otlp.proto.http.trace_exporter",
        "opentelemetry.sdk.trace.export",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.CRITICAL)

    exporter: SpanExporter
    if settings.OTEL_TRACES_EXPORTER == "console":
        exporter = ConsoleSpanExporter()
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        # Endpoint lu depuis OTEL_EXPORTER_OTLP_ENDPOINT (défaut : localhost:4318).
        exporter = OTLPSpanExporter()

    # Le sampler est construit depuis OTEL_TRACES_SAMPLER / _ARG (défaut :
    # parentbased_always_on).
    provider = TracerProvider(
        resource=Resource.create({"service.name": settings.OTEL_SERVICE_NAME})
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return provider


def _build_meter_provider(app: FastAPI) -> "MeterProvider":
    """Construit le pipeline de métriques et monte ``/metrics`` sur l'app.

    Le ``PrometheusMetricReader`` expose les métriques OTel au format
    Prometheus (mode pull, aucun collector requis) dans un registre dédié —
    jamais le registre global de ``prometheus_client``, pour rester sans
    effet de bord. La route est hors contrat OpenAPI. Crée aussi les quatre
    instruments de qualité d'extraction alimentés par ``monitoring.py``.
    """
    from fastapi import Response
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        generate_latest,
    )

    global _extraction_counter, _score_histogram, _champs_histogram, _duree_histogram

    registry = CollectorRegistry()
    reader = PrometheusMetricReader(registry=registry)
    provider = MeterProvider(
        resource=Resource.create({"service.name": settings.OTEL_SERVICE_NAME}),
        metric_readers=[reader],
    )

    meter = provider.get_meter("src.core.telemetry")
    # Exposé côté Prometheus sous le nom `extraction_total`. Le taux d'échec
    # s'alerte avec : rate(extraction_total{statut="echec"}) / rate(extraction_total).
    _extraction_counter = meter.create_counter(
        "extraction",
        unit="1",
        description="Extractions traitées, par statut, type et mode d'extraction",
    )
    # La moyenne s'alerte avec : rate(..._sum) / rate(..._count) ; les buckets
    # donnent en plus les quantiles, qu'une jauge ne permettrait pas.
    _score_histogram = meter.create_histogram(
        "extraction.score_confiance",
        unit="1",
        description="Score de confiance global des extractions réussies (0-1)",
        explicit_bucket_boundaries_advisory=_SCORE_BUCKETS,
    )
    _champs_histogram = meter.create_histogram(
        "extraction.taux_champs_reconnus",
        unit="1",
        description=(
            "Part des champs extraits avec une confiance suffisante, "
            "extractions réussies (0-1)"
        ),
        explicit_bucket_boundaries_advisory=_SCORE_BUCKETS,
    )
    _duree_histogram = meter.create_histogram(
        "extraction.duree",
        unit="s",
        description="Durée du pipeline d'extraction complet, callback compris",
        explicit_bucket_boundaries_advisory=_DUREE_BUCKETS,
    )

    def metrics_endpoint() -> Response:
        """Expose le registre au format texte Prometheus."""
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    app.add_api_route(
        "/metrics", metrics_endpoint, methods=["GET"], include_in_schema=False
    )
    return provider


def setup_telemetry(app: FastAPI) -> None:
    """Active traces et/ou métriques sur l'app, ou ne fait rien si tout est
    désactivé.

    Les instrumentations sont posées une seule fois et partagées par les deux
    pipelines ; la brique non activée reçoit un provider no-op explicite
    (jamais le provider global, pour rester déterministe).
    """
    if not (settings.OTEL_ENABLED or settings.OTEL_METRICS_ENABLED):
        return

    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.metrics import NoOpMeterProvider
    from opentelemetry.trace import NoOpTracerProvider

    # Relais des variables standard avant toute initialisation des
    # instrumentations (le opt-in semconv est lu une seule fois, au premier
    # instrument) ; l'environnement réel reste prioritaire (setdefault).
    for name in _OTEL_ENV_VARS:
        value = getattr(settings, name)
        if value is not None:
            os.environ.setdefault(name, str(value))

    tracer_provider = (
        _build_tracer_provider() if settings.OTEL_ENABLED else NoOpTracerProvider()
    )
    meter_provider = (
        _build_meter_provider(app)
        if settings.OTEL_METRICS_ENABLED
        else NoOpMeterProvider()
    )

    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        excluded_urls=EXCLUDED_URLS,
        server_request_hook=_server_request_hook,
    )
    # Instrumentation globale au processus : couvre les clients httpx créés à
    # la volée (callback vers l'API data) comme ceux du SDK Groq. Le scraping
    # n'est pas concerné — il tourne dans un autre processus (cf. docstring).
    HTTPXClientInstrumentor().instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        request_hook=_client_request_hook,
        async_request_hook=_async_client_request_hook,
    )
