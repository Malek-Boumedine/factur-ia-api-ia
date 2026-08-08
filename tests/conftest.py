"""Configuration pytest à la racine des tests.

Deux responsabilités : charger la configuration de test, et rendre *impossible*
tout appel réseau réel pendant la suite.
"""

import socket
from pathlib import Path
from typing import Any, NoReturn

import pytest
from dotenv import load_dotenv

# --- Configuration de test --------------------------------------------------
#
# Chargé au niveau module (avant tout import de src.core.config par les tests),
# car `Settings()` est instancié à l'import du module de config : sans ces
# variables, l'instanciation échoue et la collecte pytest plante en CI (aucune
# variable d'environnement fournie par le runner).

_ENV_TEST = Path(__file__).parent.parent / ".env.test"
load_dotenv(_ENV_TEST, override=True)


# --- Interdiction du réseau -------------------------------------------------

_MESSAGE_RESEAU = (
    "Appel réseau réel tenté depuis un test (cible : {cible}). La suite doit "
    "rester hermétique : le LLM Groq et le callback de l'API data sont toujours "
    "mockés. Mocke la frontière concernée (`structurer.call_llm`, "
    "`service.send_callback`) plutôt que de laisser passer l'appel."
)


@pytest.fixture(autouse=True)
def _reseau_interdit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fait échouer tout test qui tenterait une vraie connexion sortante.

    Les mocks sont posés test par test ; cette garde transforme cette convention
    en garantie vérifiée par la machine. Sans elle, un test qui oublie de mocker
    ``call_llm`` appellerait Groq pour de vrai — silencieusement s'il se trouve
    qu'une clé valide traîne dans l'environnement du poste, et en facturant des
    appels.

    On bloque l'**établissement de connexion**, pas la création de socket : les
    bibliothèques qui instancient des sockets sans jamais s'en servir (ou qui
    utilisent ``socketpair`` en interne) ne sont pas gênées.

    Le ``TestClient`` de FastAPI n'est pas concerné : il parle à l'application
    via un transport ASGI en mémoire, sans jamais ouvrir de socket — c'est
    précisément ce qui rend les tests d'endpoint compatibles avec cette garde.
    Le store MLflow des tests de monitoring est un fichier SQLite temporaire :
    lui non plus n'ouvre aucune connexion réseau.
    """

    def _refuse(cible: Any) -> NoReturn:
        raise RuntimeError(_MESSAGE_RESEAU.format(cible=cible))

    def _connect(self: socket.socket, address: Any) -> NoReturn:
        _refuse(address)

    def _create_connection(address: Any, *args: Any, **kwargs: Any) -> NoReturn:
        _refuse(address)

    monkeypatch.setattr(socket.socket, "connect", _connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _connect)
    monkeypatch.setattr(socket, "create_connection", _create_connection)
