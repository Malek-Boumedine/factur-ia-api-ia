"""Tests du client callback OCR (``send_callback``).

``httpx`` est systématiquement mocké (via ``_build_client``, même approche que
``_get_client`` dans les tests du client LLM) : aucun appel réseau réel. On
teste la *logique* du client — sérialisation du payload (Decimal en chaînes,
précision préservée), headers d'authentification (secret partagé, et jeton
IAM Cloud Run quand ``DATA_API_IAM_AUTH_ENABLED`` est actif — obtention
mockée, cf. ``test_gcp_identity.py``), retries sur le transitoire uniquement,
échec immédiat sur le définitif — pas l'API data elle-même.
"""

import json
import time
from datetime import date
from decimal import Decimal
from typing import Any, NoReturn

import httpx
import pytest
from src.callback import client as callback_client
from src.callback.client import CallbackError, send_callback
from src.callback.schemas import LigneOcr, OcrWebhookPayload
from src.core import gcp_identity
from src.core.config import settings
from src.core.gcp_identity import IdentityTokenError

from tests.test_gcp_identity import _fake_jwt

# Corps de la réponse 200 de l'API data (contrat du webhook OCR).
_OK_BODY = {
    "message": "Extraction enregistrée.",
    "id_extraction": 101,
    "statut": "a_valider",
    "id_facture": 55,
}


def _payload() -> OcrWebhookPayload:
    """Payload de succès représentatif (montants en Decimal, date réelle)."""
    return OcrWebhookPayload(
        id_document=42,
        score_confiance=Decimal("0.87"),
        siret_emetteur="12345678900011",
        numero_facture="FA-2026-042",
        date_emission=date(2026, 7, 6),
        total_ht=Decimal("1000.00"),
        total_tva=Decimal("200.00"),
        total_ttc=Decimal("1200.00"),
        lignes=[
            LigneOcr(
                designation="Prestation de conseil",
                quantite=Decimal("2"),
                prix_unitaire_ht=Decimal("500.00"),
                taux_tva=Decimal("20"),
            )
        ],
        type_document="facture",
        par_champ={
            "siret_emetteur": Decimal("1.0000"),
            "lignes": Decimal("0.8333"),
        },
    )


class _FakeResponse:
    """Fausse réponse httpx : statut + corps JSON optionnel (objet, liste...)."""

    def __init__(self, status_code: int, json_body: Any = None):
        self.status_code = status_code
        self._json_body = json_body

    def json(self) -> Any:
        if self._json_body is None:
            raise ValueError("pas de corps JSON")
        return self._json_body


class _FakeHttpxClient:
    """Faux ``httpx.Client`` : capture les POST et rejoue un scénario.

    ``outcomes`` fournit un résultat par tentative : une ``_FakeResponse``
    renvoyée telle quelle, ou une exception levée (timeout, réseau).
    """

    def __init__(self, outcomes: list[_FakeResponse | Exception]) -> None:
        self._outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    def __enter__(self) -> "_FakeHttpxClient":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def post(self, url: str, *, content: str, headers: dict[str, str]) -> _FakeResponse:
        self.calls.append({"url": url, "content": content, "headers": headers})
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _install(
    monkeypatch: pytest.MonkeyPatch, outcomes: list[_FakeResponse | Exception]
) -> _FakeHttpxClient:
    """Installe le faux client httpx à la place de ``_build_client``."""
    fake = _FakeHttpxClient(outcomes)
    monkeypatch.setattr(callback_client, "_build_client", lambda: fake)
    return fake


def test_success_posts_once_with_token_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un 200 → un seul POST, sur l'URL du callback, avec le token en header."""
    fake = _install(monkeypatch, [_FakeResponse(200, _OK_BODY)])

    send_callback(_payload())

    (call,) = fake.calls
    assert call["url"] == settings.ocr_callback_url
    assert call["headers"]["X-OCR-Secret-Token"] == settings.SECRET_OCR_TOKEN
    assert call["headers"]["Content-Type"] == "application/json"


def test_serialization_preserves_amounts_and_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Les Decimal partent en chaînes JSON (précision intacte), les dates en ISO."""
    fake = _install(monkeypatch, [_FakeResponse(200, _OK_BODY)])

    send_callback(_payload())

    body = json.loads(fake.calls[0]["content"])
    assert body["total_ht"] == "1000.00"
    assert body["total_tva"] == "200.00"
    assert body["total_ttc"] == "1200.00"
    assert body["score_confiance"] == "0.87"
    assert body["date_emission"] == "2026-07-06"
    assert body["id_document"] == 42
    assert body["lignes"][0]["prix_unitaire_ht"] == "500.00"
    assert body["lignes"][0]["taux_tva"] == "20"


def test_serialization_of_optional_contract_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Les champs optionnels du contrat partent en JSON : ``type_document`` en
    chaîne simple, les scores de ``par_champ`` en chaînes (comme les Decimal)."""
    fake = _install(monkeypatch, [_FakeResponse(200, _OK_BODY)])

    send_callback(_payload())

    body = json.loads(fake.calls[0]["content"])
    assert body["type_document"] == "facture"
    assert body["par_champ"] == {"siret_emetteur": "1.0000", "lignes": "0.8333"}


def test_serialization_without_optional_fields_is_backward_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un payload sans les champs optionnels (ex. payload d'échec) reste valide et
    les sérialise à ``null`` : l'ajout au contrat est purement additif."""
    fake = _install(monkeypatch, [_FakeResponse(200, _OK_BODY)])
    minimal = OcrWebhookPayload(
        id_document=42,
        score_confiance=Decimal("0"),
        total_ht=Decimal("0"),
        total_tva=Decimal("0"),
        total_ttc=Decimal("0"),
    )

    send_callback(minimal)

    body = json.loads(fake.calls[0]["content"])
    assert body["type_document"] is None
    assert body["par_champ"] is None
    assert body["score_confiance"] == "0"


def test_success_logs_returned_ids(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Le 200 journalise les ids retournés par l'API data (traçabilité)."""
    _install(monkeypatch, [_FakeResponse(200, _OK_BODY)])

    with caplog.at_level("INFO"):
        send_callback(_payload())

    assert "id_extraction=101" in caplog.text
    assert "id_facture=55" in caplog.text
    assert "statut=a_valider" in caplog.text


def test_success_without_json_body_does_not_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un 200 sans corps JSON exploitable reste un succès (log dégradé)."""
    _install(monkeypatch, [_FakeResponse(200)])

    send_callback(_payload())  # ne doit pas lever


def test_success_with_non_dict_json_body_does_not_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un 200 dont le corps JSON n'est pas un objet (ex. liste) reste un succès :
    la journalisation dégradée ne doit pas faire planter la tâche de fond."""
    _install(monkeypatch, [_FakeResponse(200, [1, 2, 3])])

    send_callback(_payload())  # ne doit pas lever


@pytest.mark.parametrize("status_code", [403, 404])
def test_definitive_4xx_fails_immediately_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    status_code: int,
) -> None:
    """403 (token refusé) et 404 (id inconnu) → échec immédiat, aucun retry."""
    fake = _install(monkeypatch, [_FakeResponse(status_code)])

    with caplog.at_level("WARNING"), pytest.raises(CallbackError) as exc_info:
        send_callback(_payload())

    assert len(fake.calls) == 1  # rejouer donnerait le même résultat
    assert str(status_code) in str(exc_info.value)
    # Le token ne doit jamais fuiter, ni dans les logs ni dans l'exception.
    assert settings.SECRET_OCR_TOKEN not in caplog.text
    assert settings.SECRET_OCR_TOKEN not in str(exc_info.value)


def test_timeout_then_success_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un timeout transitoire est retenté ; le 200 suivant conclut en succès."""
    fake = _install(
        monkeypatch,
        [httpx.TimeoutException("délai dépassé"), _FakeResponse(200, _OK_BODY)],
    )

    send_callback(_payload())  # ne doit pas lever

    assert len(fake.calls) == 2


def test_persistent_timeout_exhausts_retries(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Timeout à chaque tentative → HTTP_MAX_RETRIES tentatives puis abandon."""
    outcomes: list[_FakeResponse | Exception] = [
        httpx.TimeoutException("délai dépassé")
        for _ in range(settings.HTTP_MAX_RETRIES)
    ]
    fake = _install(monkeypatch, outcomes)

    with caplog.at_level("WARNING"), pytest.raises(CallbackError):
        send_callback(_payload())

    assert len(fake.calls) == settings.HTTP_MAX_RETRIES
    assert settings.SECRET_OCR_TOKEN not in caplog.text


def test_network_error_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """Une erreur réseau (ConnectError) est transitoire : retentée comme un timeout."""
    fake = _install(
        monkeypatch,
        [httpx.ConnectError("connexion refusée"), _FakeResponse(200, _OK_BODY)],
    )

    send_callback(_payload())

    assert len(fake.calls) == 2


def test_5xx_retried_then_abandoned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un 5xx persistant (API data indisponible) est retenté puis abandonné."""
    outcomes: list[_FakeResponse | Exception] = [
        _FakeResponse(503) for _ in range(settings.HTTP_MAX_RETRIES)
    ]
    fake = _install(monkeypatch, outcomes)

    with pytest.raises(CallbackError):
        send_callback(_payload())

    assert len(fake.calls) == settings.HTTP_MAX_RETRIES


def test_5xx_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un 5xx suivi d'un 200 : le retry conclut en succès."""
    fake = _install(monkeypatch, [_FakeResponse(500), _FakeResponse(200, _OK_BODY)])

    send_callback(_payload())

    assert len(fake.calls) == 2


# --- Authentification IAM Cloud Run (X-Serverless-Authorization) ------------


def _enable_iam_with_fake_token(monkeypatch: pytest.MonkeyPatch) -> str:
    """Active le flag IAM, vide le cache et mocke l'obtention du jeton."""
    token = _fake_jwt(time.time() + 3600)
    monkeypatch.setattr(settings, "DATA_API_IAM_AUTH_ENABLED", True)
    monkeypatch.setattr(gcp_identity, "_cached_token", None)
    monkeypatch.setattr(gcp_identity, "_cached_expiry", 0.0)
    monkeypatch.setattr(gcp_identity, "_fetch_identity_token", lambda: token)
    return token


def test_iam_enabled_sends_both_auth_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag IAM actif : le jeton d'identité part dans X-Serverless-Authorization,
    ET le secret partagé X-OCR-Secret-Token part toujours (auth applicative)."""
    token = _enable_iam_with_fake_token(monkeypatch)
    fake = _install(monkeypatch, [_FakeResponse(200, _OK_BODY)])

    send_callback(_payload())

    (call,) = fake.calls
    assert call["headers"]["X-Serverless-Authorization"] == f"Bearer {token}"
    assert call["headers"]["X-OCR-Secret-Token"] == settings.SECRET_OCR_TOKEN


def test_iam_disabled_sends_no_iam_header_and_never_fetches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag IAM inactif (défaut dev/test) : aucun en-tête IAM, aucune tentative
    d'obtention de jeton — et le secret partagé part toujours."""

    def _boom() -> NoReturn:
        raise AssertionError("obtention de jeton tentée alors que le flag est inactif")

    monkeypatch.setattr(gcp_identity, "_fetch_identity_token", _boom)
    fake = _install(monkeypatch, [_FakeResponse(200, _OK_BODY)])

    send_callback(_payload())

    (call,) = fake.calls
    assert "X-Serverless-Authorization" not in call["headers"]
    assert call["headers"]["X-OCR-Secret-Token"] == settings.SECRET_OCR_TOKEN


def test_identity_token_error_is_transient_and_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Jeton inobtenable (serveur de métadonnées en vrac) : transitoire — rejoué
    à chaque tentative, aucun POST parti sans jeton, CallbackError à la fin."""

    def _no_token() -> NoReturn:
        raise IdentityTokenError()

    monkeypatch.setattr(callback_client, "serverless_authorization_header", _no_token)
    fake = _install(monkeypatch, [])

    with pytest.raises(CallbackError) as exc_info:
        send_callback(_payload())

    assert len(fake.calls) == 0  # jamais de POST sans l'en-tête IAM requis
    assert "jeton d'identité" in str(exc_info.value)


def test_identity_token_error_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un échec d'obtention isolé profite des retries : le POST suivant conclut."""
    outcomes: list[Exception | dict[str, str]] = [IdentityTokenError(), {}]

    def _flaky_header() -> dict[str, str]:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(
        callback_client, "serverless_authorization_header", _flaky_header
    )
    fake = _install(monkeypatch, [_FakeResponse(200, _OK_BODY)])

    send_callback(_payload())  # ne doit pas lever

    assert len(fake.calls) == 1


def test_403_diagnostic_names_both_causes_without_asserting_one(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Le 403 n'accuse plus le secret partagé : la plateforme (IAM Cloud Run)
    renvoie le même statut, le message énumère les deux hypothèses. Flag IAM
    inactif → l'indice pointe DATA_API_IAM_AUTH_ENABLED (le cas du bug vécu)."""
    _install(monkeypatch, [_FakeResponse(403)])

    with caplog.at_level("ERROR"), pytest.raises(CallbackError) as exc_info:
        send_callback(_payload())

    message = str(exc_info.value)
    assert "token OCR partagé refusé" not in message
    assert "token OCR partagé refusé" not in caplog.text
    assert "IAM Cloud Run" in message
    assert "X-OCR-Secret-Token" in message
    assert "activer DATA_API_IAM_AUTH_ENABLED" in message
    assert "activer DATA_API_IAM_AUTH_ENABLED" in caplog.text


def test_403_hint_absent_when_iam_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flag IAM actif : le 403 garde le diagnostic neutre, sans l'indice
    d'activation (la variable est déjà en place, le pointer serait trompeur)."""
    _enable_iam_with_fake_token(monkeypatch)
    _install(monkeypatch, [_FakeResponse(403)])

    with pytest.raises(CallbackError) as exc_info:
        send_callback(_payload())

    message = str(exc_info.value)
    assert "IAM Cloud Run" in message
    assert "activer DATA_API_IAM_AUTH_ENABLED" not in message


def test_build_client_applique_le_timeout_configure() -> None:
    """Le client HTTP réel est construit avec le timeout de la configuration.

    Seul test à instancier le vrai ``httpx.Client`` (sans jamais s'en servir) :
    les autres le remplacent pour se concentrer sur la logique d'envoi. Il
    couvre le câblage de la configuration vers le transport, qui n'apparaît
    nulle part ailleurs — un timeout non appliqué laisserait la tâche de fond
    pendre indéfiniment sur une API data muette.
    """
    with callback_client._build_client() as client:
        assert client.timeout.connect == settings.HTTP_TIMEOUT_SECONDS
        assert client.timeout.read == settings.HTTP_TIMEOUT_SECONDS
