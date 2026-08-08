"""Tests de la garde réseau posée par ``conftest.py``.

Une garde qui ne garde rien est pire que pas de garde : elle donne une fausse
assurance. Ces tests vérifient qu'elle se déclenche réellement, sur la primitive
système comme sur la bibliothèque HTTP effectivement utilisée par le service
(``httpx``, côté callback) et sur le SDK Groq.

Ils n'ouvrent évidemment aucune connexion : c'est justement l'échec de la
tentative qui est vérifié.
"""

import socket

import httpx
import pytest

# Adresse de documentation réservée (RFC 5737), jamais routable : même si la
# garde venait à sauter, aucun test ne pourrait joindre quoi que ce soit.
_ADRESSE_INJOIGNABLE = ("192.0.2.1", 80)


def test_connexion_socket_directe_refusee() -> None:
    """Une connexion sortante brute est refusée avec un message explicite."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(RuntimeError, match="Appel réseau réel"):
            sock.connect(_ADRESSE_INJOIGNABLE)


def test_create_connection_refusee() -> None:
    """``socket.create_connection`` (chemin de la plupart des clients) est refusé."""
    with pytest.raises(RuntimeError, match="Appel réseau réel"):
        socket.create_connection(_ADRESSE_INJOIGNABLE)


def test_requete_httpx_refusee() -> None:
    """Un POST httpx réel est bloqué : c'est la stack du callback API data.

    ``httpx`` est aussi le transport du SDK Groq : bloquer ici couvre les deux
    frontières réseau du service. L'erreur remonte telle quelle (httpcore ne
    convertit en ``ConnectError`` que les défaillances système qu'il connaît),
    ce qui laisse le message de diagnostic intact.
    """
    with pytest.raises(RuntimeError, match="Appel réseau réel"):
        httpx.post("http://192.0.2.1/documents/webhook/ocr", json={})


def test_message_oriente_vers_la_frontiere_a_mocker() -> None:
    """Le message d'erreur dit quoi faire, pas seulement ce qui a échoué."""
    with pytest.raises(RuntimeError) as erreur:
        socket.create_connection(_ADRESSE_INJOIGNABLE)

    message = str(erreur.value)
    assert "call_llm" in message
    assert "send_callback" in message
