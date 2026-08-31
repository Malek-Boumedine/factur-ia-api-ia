"""Tests du jeton d'identité Google (``core/gcp_identity.py``).

L'obtention réelle (``fetch_id_token``, serveur de métadonnées Cloud Run) est
systématiquement mockée : aucun environnement GCP requis, aucun réseau (la
garde de ``conftest.py`` l'interdit de toute façon). On teste la *logique* du
module — interrupteur ``DATA_API_IAM_AUTH_ENABLED``, format de l'en-tête,
audience, cache et renouvellement anticipé, traduction des erreurs.
"""

import base64
import json
import time
from typing import NoReturn

import pytest
from src.core import gcp_identity
from src.core.config import settings
from src.core.gcp_identity import IdentityTokenError, serverless_authorization_header


def _fake_jwt(exp: float) -> str:
    """Jeton JWT factice (non signé) portant la claim ``exp`` donnée."""
    payload = (
        base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode())
        .rstrip(b"=")
        .decode()
    )
    return f"entete.{payload}.signature"


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repart d'un cache vide à chaque test (état module-niveau partagé)."""
    monkeypatch.setattr(gcp_identity, "_cached_token", None)
    monkeypatch.setattr(gcp_identity, "_cached_expiry", 0.0)


def _enable_iam(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "DATA_API_IAM_AUTH_ENABLED", True)


def _forbid_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fait échouer le test si le module tente d'obtenir un jeton."""

    def _boom() -> NoReturn:
        raise AssertionError("obtention de jeton tentée alors qu'interdite")

    monkeypatch.setattr(gcp_identity, "_fetch_identity_token", _boom)


def test_disabled_returns_empty_without_fetching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interrupteur à faux (défaut dev/test) : dict vide, zéro obtention."""
    _forbid_fetch(monkeypatch)

    assert serverless_authorization_header() == {}


def test_enabled_returns_bearer_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """Interrupteur à vrai : en-tête X-Serverless-Authorization en Bearer."""
    _enable_iam(monkeypatch)
    token = _fake_jwt(time.time() + 3600)
    monkeypatch.setattr(gcp_identity, "_fetch_identity_token", lambda: token)

    assert serverless_authorization_header() == {
        "X-Serverless-Authorization": f"Bearer {token}"
    }


def test_audience_is_data_api_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """L'audience passée à google-auth est l'URL canonique de l'API data.

    C'est elle que Cloud Run vérifie côté API data : un jeton signé pour un
    autre format d'URL serait rejeté comme un jeton absent, sans erreur
    visible — d'où l'assertion sur la valeur exacte.
    """
    _enable_iam(monkeypatch)
    audiences: list[str] = []

    def _fake_fetch(request: object, audience: str) -> str:
        audiences.append(audience)
        return _fake_jwt(time.time() + 3600)

    monkeypatch.setattr(gcp_identity, "fetch_id_token", _fake_fetch)

    serverless_authorization_header()

    assert audiences == [settings.DATA_API_BASE_URL]


def test_token_is_cached_between_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deux appels successifs : une seule obtention, le cache sert le second."""
    _enable_iam(monkeypatch)
    calls: list[None] = []

    def _fake_fetch() -> str:
        calls.append(None)
        return _fake_jwt(time.time() + 3600)

    monkeypatch.setattr(gcp_identity, "_fetch_identity_token", _fake_fetch)

    first = serverless_authorization_header()
    second = serverless_authorization_header()

    assert first == second
    assert len(calls) == 1


def test_near_expiry_token_is_refreshed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un jeton à moins de la marge de son expiration est renouvelé d'avance."""
    _enable_iam(monkeypatch)
    tokens = iter(
        [
            # Expire dans 60 s : sous la marge de 300 s, donc déjà « périmé ».
            _fake_jwt(time.time() + 60),
            _fake_jwt(time.time() + 3600),
        ]
    )
    calls: list[None] = []

    def _fake_fetch() -> str:
        calls.append(None)
        return next(tokens)

    monkeypatch.setattr(gcp_identity, "_fetch_identity_token", _fake_fetch)

    serverless_authorization_header()
    serverless_authorization_header()

    assert len(calls) == 2


def test_fetch_failure_raises_identity_token_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Toute erreur d'obtention est traduite en IdentityTokenError (chaînée)."""
    _enable_iam(monkeypatch)

    def _fake_fetch(request: object, audience: str) -> NoReturn:
        raise RuntimeError("serveur de métadonnées injoignable")

    monkeypatch.setattr(gcp_identity, "fetch_id_token", _fake_fetch)

    with pytest.raises(IdentityTokenError) as exc_info:
        serverless_authorization_header()

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_unreadable_exp_falls_back_to_one_hour() -> None:
    """Claim ``exp`` illisible : repli sur une heure de vie supposée."""
    now = time.time()

    expiry = gcp_identity._read_expiry("pas-un-jwt", now)

    assert expiry == now + gcp_identity._FALLBACK_LIFETIME


def test_read_expiry_parses_exp_claim() -> None:
    """La claim ``exp`` d'un jeton bien formé est lue telle quelle."""
    exp = time.time() + 1234

    assert gcp_identity._read_expiry(_fake_jwt(exp), time.time()) == exp
