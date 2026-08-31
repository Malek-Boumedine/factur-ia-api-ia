"""Tests de l'instrumentation OpenTelemetry (`src/core/telemetry.py`).

Aucun collector ni réseau requis (la garde réseau de ``conftest.py`` reste
satisfaite) : la désactivation doit être transparente, ``/metrics`` est
interrogé via un transport ASGI en mémoire, et le scrubbing des données
sensibles est vérifié sur de vrais spans SDK gardés en mémoire (jamais
exportés).

Miroir du ``test_telemetry.py`` de l'API data, plus les métriques de qualité
d'extraction propres à ce service (alimentées par ``track_extraction_quality``).
"""

from decimal import Decimal
from typing import Any

import httpx
import pytest
import src.core.telemetry as telemetry
from fastapi import FastAPI
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import Span
from opentelemetry.util.http import parse_excluded_urls
from src.callback.schemas import LigneOcr, OcrWebhookPayload
from src.core.config import settings
from src.core.monitoring import (
    TRACKED_FIELDS,
    ModeExtraction,
    track_extraction_quality,
)
from src.core.telemetry import (
    EXCLUDED_URLS,
    _async_client_request_hook,
    _client_request_hook,
    _server_request_hook,
    record_extraction_quality,
    scrub_span_url_attributes,
    scrub_url,
    setup_telemetry,
)

# Valeurs sensibles plantées dans le payload de test : aucune ne doit se
# retrouver dans la sortie de /metrics (mêmes valeurs que test_monitoring).
_SIRET_EMETTEUR = "73282932000074"
_IBAN = "FR7630006000011234567890189"
_NUMERO_FACTURE = "FA-2026-042"
_DESIGNATION = "Prestation de conseil"
_TOTAL_HT = Decimal("1234.56")

_VALEURS_SENSIBLES = (
    _SIRET_EMETTEUR,
    _IBAN,
    _NUMERO_FACTURE,
    _DESIGNATION,
    str(_TOTAL_HT),
    "2026-07-06",
)


def _payload_succes() -> OcrWebhookPayload:
    """Payload d'extraction réussie, garni de valeurs sensibles réalistes."""
    return OcrWebhookPayload(
        id_document=42,
        score_confiance=Decimal("0.9000"),
        siret_emetteur=_SIRET_EMETTEUR,
        numero_facture=_NUMERO_FACTURE,
        date_emission="2026-07-06",  # type: ignore[arg-type]
        total_ht=_TOTAL_HT,
        total_tva=Decimal("246.91"),
        total_ttc=Decimal("1481.47"),
        iban=_IBAN,
        lignes=[
            LigneOcr(
                designation=_DESIGNATION,
                quantite=Decimal("2"),
                prix_unitaire_ht=Decimal("617.28"),
                taux_tva=Decimal("20"),
            )
        ],
        type_document="facture",
        par_champ=dict.fromkeys(TRACKED_FIELDS, Decimal("1.0000")),
    )


def _payload_echec() -> OcrWebhookPayload:
    """Payload d'échec canonique : marqueur ``score_confiance = 0``."""
    return OcrWebhookPayload(
        id_document=55,
        score_confiance=Decimal("0"),
        total_ht=Decimal("0"),
        total_tva=Decimal("0"),
        total_ttc=Decimal("0"),
        lignes=[],
    )


def make_recorded_span(
    attributes: dict[str, Any],
) -> tuple[Span, InMemorySpanExporter]:
    """Crée un span SDK réel (exporté en mémoire) portant les attributs donnés."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)
    span = tracer.start_span("GET", attributes=attributes)
    return span, exporter


def _valeur_serie(corps: str, nom: str, labels: dict[str, str]) -> float | None:
    """Valeur d'une série Prometheus dans la sortie texte de /metrics.

    Compare les labels *métier* exactement, en ignorant les labels
    ``otel_scope_*`` que l'exporter ajoute à chaque série. ``None`` si aucune
    série ne correspond.
    """
    for ligne in corps.splitlines():
        if not ligne.startswith(f"{nom}{{"):
            continue
        bloc = ligne[len(nom) + 1 : ligne.rindex("}")]
        paires = {
            cle: valeur.strip('"')
            for cle, valeur in (paire.split("=", 1) for paire in bloc.split(","))
            if not cle.startswith("otel_scope")
        }
        if paires == labels:
            return float(ligne.rsplit(" ", 1)[1])
    return None


def _desinstrumenter(app: FastAPI) -> None:
    """Retire les instrumentations globales et remet les instruments à zéro.

    Indispensable en fin de test : l'instrumentation httpx est globale au
    processus et les instruments de qualité sont des globals de module — sans
    nettoyage, ils fuiraient sur les autres tests.
    """
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    HTTPXClientInstrumentor().uninstrument()
    FastAPIInstrumentor.uninstrument_app(app)
    telemetry._extraction_counter = None
    telemetry._score_histogram = None
    telemetry._champs_histogram = None
    telemetry._duree_histogram = None


class TestSetupTelemetryDesactive:
    """Tout désactivé (défaut), l'instrumentation doit être invisible."""

    def test_aucun_middleware_ajoute(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "OTEL_ENABLED", False)
        monkeypatch.setattr(settings, "OTEL_METRICS_ENABLED", False)
        app = FastAPI()
        middlewares_avant = list(app.user_middleware)

        setup_telemetry(app)

        assert app.user_middleware == middlewares_avant
        assert not hasattr(app, "_original_build_middleware_stack")

    def test_app_non_marquee_instrumentee(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "OTEL_ENABLED", False)
        monkeypatch.setattr(settings, "OTEL_METRICS_ENABLED", False)
        app = FastAPI()

        setup_telemetry(app)

        assert not getattr(app, "_is_instrumented_by_opentelemetry", False)

    def test_aucune_route_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "OTEL_ENABLED", False)
        monkeypatch.setattr(settings, "OTEL_METRICS_ENABLED", False)
        app = FastAPI()

        setup_telemetry(app)

        assert all(getattr(r, "path", None) != "/metrics" for r in app.routes)

    def test_enregistrement_qualite_est_un_noop(self) -> None:
        """Sans activation, les instruments valent None et l'appel ne fait rien.

        C'est la garantie « transparent en local et en CI » : le pipeline
        appelle ``track_extraction_quality`` à chaque extraction, l'appel doit
        être gratuit et sans effet.
        """
        assert telemetry._extraction_counter is None

        record_extraction_quality(  # ne doit pas lever
            statut="succes",
            type_document="facture",
            mode_extraction="pdf_natif",
            score_confiance=0.9,
            taux_champs_reconnus=1.0,
            duree_secondes=1.5,
        )

    def test_settings_de_test_desactivent_la_telemetrie(self) -> None:
        """La suite tourne télémétrie éteinte : aucun test existant perturbé."""
        assert settings.OTEL_ENABLED is False
        assert settings.OTEL_METRICS_ENABLED is False


class TestScrubUrl:
    """La query string, les credentials et le fragment sont retirés."""

    @pytest.mark.parametrize(
        ("url", "attendu"),
        [
            (
                "https://api.groq.com/openai/v1/chat/completions?key=secret",
                "https://api.groq.com/openai/v1/chat/completions",
            ),
            (
                # pragma: allowlist nextline secret
                "http://user:secret@localhost:8000/documents/webhook/ocr?a=1#frag",
                "http://localhost:8000/documents/webhook/ocr",
            ),
            # URL déjà propre : inchangée.
            (
                "http://localhost:8000/documents/webhook/ocr",
                "http://localhost:8000/documents/webhook/ocr",
            ),
        ],
    )
    def test_scrubbing(self, url: str, attendu: str) -> None:
        assert scrub_url(url) == attendu


class TestScrubSpanUrlAttributes:
    """Le scrubbing agit sur de vrais spans, pour les deux générations de
    conventions sémantiques HTTP."""

    def test_anciennes_conventions_http_url(self) -> None:
        span, exporter = make_recorded_span(
            {"http.url": "https://api.gouv.fr/search?q=13002526500013"}
        )
        scrub_span_url_attributes(span)
        span.end()

        (fini,) = exporter.get_finished_spans()
        assert fini.attributes is not None
        assert fini.attributes["http.url"] == "https://api.gouv.fr/search"
        assert "13002526500013" not in str(fini.attributes)

    def test_nouvelles_conventions_url_full_et_query(self) -> None:
        span, exporter = make_recorded_span(
            {
                "url.full": "https://api.gouv.fr/search?q=13002526500013",
                "url.query": "q=13002526500013",
            }
        )
        scrub_span_url_attributes(span)
        span.end()

        (fini,) = exporter.get_finished_spans()
        assert fini.attributes is not None
        assert fini.attributes["url.full"] == "https://api.gouv.fr/search"
        assert fini.attributes["url.query"] == ""
        assert "13002526500013" not in str(fini.attributes)

    def test_span_sans_attribut_url_inchange(self) -> None:
        span, exporter = make_recorded_span({"http.method": "GET"})
        scrub_span_url_attributes(span)
        span.end()

        (fini,) = exporter.get_finished_spans()
        assert fini.attributes is not None
        assert dict(fini.attributes) == {"http.method": "GET"}


class TestHooks:
    """Les hooks branchés sur les instrumentations délèguent au scrubbing."""

    def test_hook_serveur(self) -> None:
        span, exporter = make_recorded_span(
            {"http.url": "http://api/extractions?page=2"}
        )
        _server_request_hook(span, {"type": "http"})
        span.end()

        (fini,) = exporter.get_finished_spans()
        assert fini.attributes is not None
        assert fini.attributes["http.url"] == "http://api/extractions"

    def test_hook_client_sync(self) -> None:
        span, exporter = make_recorded_span({"http.url": "http://api/search?q=x"})
        _client_request_hook(span, None)
        span.end()

        (fini,) = exporter.get_finished_spans()
        assert fini.attributes is not None
        assert fini.attributes["http.url"] == "http://api/search"

    async def test_hook_client_async(self) -> None:
        span, exporter = make_recorded_span({"http.url": "http://api/search?q=x"})
        await _async_client_request_hook(span, None)
        span.end()

        (fini,) = exporter.get_finished_spans()
        assert fini.attributes is not None
        assert fini.attributes["http.url"] == "http://api/search"


class TestUrlsExclues:
    """/health, /ready et /metrics sont hors instrumentation ; les routes
    métier non.

    Les URLs testées reproduisent le format vu par le middleware ASGI :
    « scheme://host/chemin », sans query string.
    """

    @pytest.mark.parametrize(
        "url",
        [
            "http://testserver/health",
            "http://testserver/ready",
            "http://testserver/metrics",
        ],
    )
    def test_exclues(self, url: str) -> None:
        assert parse_excluded_urls(EXCLUDED_URLS).url_disabled(url)

    @pytest.mark.parametrize(
        "url",
        [
            "http://testserver/extractions",
            "http://testserver/healthcheck",
            "http://testserver/ready-set-go",
            "http://testserver/metrics-export",
        ],
    )
    def test_non_exclues(self, url: str) -> None:
        assert not parse_excluded_urls(EXCLUDED_URLS).url_disabled(url)


class TestSetupTelemetryActive:
    """Activée, l'instrumentation s'accroche bien à l'app FastAPI.

    L'instrumentation 0.65b0 ne passe pas par ``add_middleware`` : elle patche
    ``app.build_middleware_stack`` et pose le drapeau
    ``_is_instrumented_by_opentelemetry``. L'exporter console évite toute
    dépendance réseau ; les instrumentations globales (httpx) sont retirées en
    fin de test pour ne pas fuir sur les autres tests.
    """

    def test_app_instrumentee_puis_retiree(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "OTEL_ENABLED", True)
        monkeypatch.setattr(settings, "OTEL_METRICS_ENABLED", False)
        monkeypatch.setattr(settings, "OTEL_TRACES_EXPORTER", "console")
        app = FastAPI()

        try:
            setup_telemetry(app)
            assert getattr(app, "_is_instrumented_by_opentelemetry", False)
            assert hasattr(app, "_original_build_middleware_stack")
            # Traces seules : pas d'endpoint ni d'instruments de métriques.
            assert all(getattr(r, "path", None) != "/metrics" for r in app.routes)
            assert telemetry._extraction_counter is None
        finally:
            _desinstrumenter(app)


class TestMetricsEndpoint:
    """Métriques seules activées : /metrics répond au format Prometheus, avec
    des labels templatés sans ID réel, et les traces restent no-op.

    Le mode des conventions sémantiques est mémorisé au premier instrument du
    process : on force sa relecture pour obtenir les conventions stables
    (label http_route), comme au démarrage réel de l'app.
    """

    async def test_metrics_exposees_labels_templates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from opentelemetry.instrumentation._semconv import (
            _OpenTelemetrySemanticConventionStability,
        )

        monkeypatch.setattr(settings, "OTEL_ENABLED", False)
        monkeypatch.setattr(settings, "OTEL_METRICS_ENABLED", True)
        monkeypatch.setenv("OTEL_SEMCONV_STABILITY_OPT_IN", "http")
        monkeypatch.setattr(
            _OpenTelemetrySemanticConventionStability, "_initialized", False
        )

        app = FastAPI()

        @app.get("/documents/{id_document}")
        async def get_document(id_document: int) -> dict[str, int]:
            return {"id": id_document}

        try:
            setup_telemetry(app)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                reponse = await client.get("/documents/987654321")
                assert reponse.status_code == 200
                metrics = await client.get("/metrics")

            assert metrics.status_code == 200
            assert metrics.headers["content-type"].startswith("text/plain")
            corps = metrics.text
            # Histogramme HTTP présent, labellé par la route templatée.
            assert "http_server_request_duration_seconds" in corps
            assert 'http_route="/documents/{id_document}"' in corps
            # Jamais l'identifiant réel dans les labels.
            assert "987654321" not in corps
        finally:
            _desinstrumenter(app)

    async def test_scrape_de_metrics_non_compte(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Le scrape de /metrics ne doit pas alimenter ses propres compteurs."""
        from opentelemetry.instrumentation._semconv import (
            _OpenTelemetrySemanticConventionStability,
        )

        monkeypatch.setattr(settings, "OTEL_ENABLED", False)
        monkeypatch.setattr(settings, "OTEL_METRICS_ENABLED", True)
        monkeypatch.setenv("OTEL_SEMCONV_STABILITY_OPT_IN", "http")
        monkeypatch.setattr(
            _OpenTelemetrySemanticConventionStability, "_initialized", False
        )
        app = FastAPI()

        try:
            setup_telemetry(app)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                await client.get("/metrics")
                metrics = await client.get("/metrics")

            assert 'http_route="/metrics"' not in metrics.text
        finally:
            _desinstrumenter(app)


class TestMetriquesQualite:
    """Les métriques de qualité d'extraction sont alimentées via le point
    d'appel unique du pipeline (``track_extraction_quality``) et exposées sur
    /metrics avec leurs labels bornés — jamais l'``id_document``."""

    async def _scraper_apres_extractions(self, monkeypatch: pytest.MonkeyPatch) -> str:
        """Active les métriques, trace un succès et un échec, scrape /metrics."""
        monkeypatch.setattr(settings, "OTEL_ENABLED", False)
        monkeypatch.setattr(settings, "OTEL_METRICS_ENABLED", True)
        app = FastAPI()

        try:
            setup_telemetry(app)
            track_extraction_quality(
                _payload_succes(),
                mode_extraction=ModeExtraction.PDF_NATIF,
                duree_secondes=2.5,
            )
            track_extraction_quality(
                _payload_echec(),
                mode_extraction=ModeExtraction.OCR,
                duree_secondes=0.4,
            )

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                metrics = await client.get("/metrics")
            assert metrics.status_code == 200
            return metrics.text
        finally:
            _desinstrumenter(app)

    async def test_compteur_par_statut(self, monkeypatch: pytest.MonkeyPatch) -> None:
        corps = await self._scraper_apres_extractions(monkeypatch)

        # Le compteur porte l'alerte de taux d'échec : une série par statut,
        # avec le type et le mode en dimensions d'explication.
        assert (
            _valeur_serie(
                corps,
                "extraction_total",
                {
                    "statut": "succes",
                    "type_document": "facture",
                    "mode_extraction": "pdf_natif",
                },
            )
            == 1.0
        )
        assert (
            _valeur_serie(
                corps,
                "extraction_total",
                {
                    "statut": "echec",
                    "type_document": "non_calcule",
                    "mode_extraction": "ocr",
                },
            )
            == 1.0
        )

    async def test_histogrammes_de_qualite_succes_seulement(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Score et taux n'enregistrent que les succès : le score conventionnel
        à 0 d'un échec fausserait la moyenne (l'alerte d'échec vit sur le
        compteur, signaux indépendants)."""
        corps = await self._scraper_apres_extractions(monkeypatch)

        # Une seule observation (le succès), avec sa valeur exacte dans _sum.
        labels_succes = {"type_document": "facture", "mode_extraction": "pdf_natif"}
        assert (
            _valeur_serie(corps, "extraction_score_confiance_count", labels_succes)
            == 1.0
        )
        assert (
            _valeur_serie(corps, "extraction_score_confiance_sum", labels_succes) == 0.9
        )
        assert (
            _valeur_serie(corps, "extraction_taux_champs_reconnus_count", labels_succes)
            == 1.0
        )
        # Buckets adaptés aux scores 0-1 (les défauts OTel visent des latences).
        assert 'le="0.9"' in corps
        # L'échec n'a alimenté aucun histogramme de qualité : aucune série
        # de score pour son couple de labels, aucun bucket sur tout /metrics
        # ne dépasse une observation.
        assert (
            _valeur_serie(
                corps,
                "extraction_score_confiance_count",
                {"type_document": "non_calcule", "mode_extraction": "ocr"},
            )
            is None
        )

    async def test_duree_par_statut(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """La durée est observée pour les deux issues : un échec a aussi un
        coût, et c'est souvent lui le plus long (retries, timeouts)."""
        corps = await self._scraper_apres_extractions(monkeypatch)

        assert (
            _valeur_serie(
                corps,
                "extraction_duree_seconds_count",
                {"statut": "succes", "mode_extraction": "pdf_natif"},
            )
            == 1.0
        )
        assert (
            _valeur_serie(
                corps,
                "extraction_duree_seconds_count",
                {"statut": "echec", "mode_extraction": "ocr"},
            )
            == 1.0
        )

    async def test_aucune_donnee_sensible_ni_id_document(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """La sortie scrapée ne contient que des agrégats et des labels bornés.

        Vérifié sur ce que /metrics renvoie réellement : ni valeur de facture,
        ni ``id_document`` (cardinalité non bornée + fuite d'identifiant), ni
        secret.
        """
        corps = await self._scraper_apres_extractions(monkeypatch)

        for valeur in _VALEURS_SENSIBLES:
            assert valeur not in corps, f"donnée sensible exposée : {valeur}"
        assert "id_document" not in corps
        assert settings.SECRET_OCR_TOKEN not in corps
        assert settings.GROQ_API_KEY not in corps
