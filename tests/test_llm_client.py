"""Tests du client LLM Groq (``call_llm``).

Le SDK ``groq`` est systématiquement mocké : aucun appel réseau réel ni clé
valide en CI. On teste la *logique* du client (transmission de la config au SDK,
extraction du contenu de la réponse, gestion d'erreur, cache du client), pas le
comportement du modèle Groq lui-même.
"""

from typing import Any

import pytest
from src.extractions import llm_client
from src.extractions.llm_client import LlmClientError, call_llm


class _FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    """Faux endpoint ``chat.completions`` : capture les kwargs et renvoie ou lève.

    ``content`` fixe le texte renvoyé par la réponse ; ``error`` (si fourni) est
    levé à l'appel de ``create`` pour simuler une défaillance de l'API Groq.
    """

    def __init__(
        self, content: str | None = "reponse", error: Exception | None = None
    ) -> None:
        self._content = content
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return _FakeResponse(self._content)


class _FakeClient:
    """Faux client Groq exposant ``chat.completions.create``."""

    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = type("_Chat", (), {"completions": completions})()


@pytest.fixture(autouse=True)
def _reset_client_cache() -> None:
    """Réinitialise le cache module-level du client entre les tests."""
    llm_client._client = None


def test_call_llm_returns_model_content(monkeypatch: pytest.MonkeyPatch) -> None:
    completions = _FakeCompletions(content="JSON structuré")
    monkeypatch.setattr(llm_client, "_get_client", lambda: _FakeClient(completions))

    result = call_llm("prompt système", "texte brut extrait")

    assert result == "JSON structuré"


def test_call_llm_passes_config_and_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    # Modèle depuis la config, température basse et stream=False (déterministe,
    # réponse complète), et les deux messages system/user dans le bon ordre.
    completions = _FakeCompletions()
    monkeypatch.setattr(llm_client, "_get_client", lambda: _FakeClient(completions))

    call_llm("consignes", "contenu")

    (kwargs,) = completions.calls
    assert kwargs["model"] == llm_client.settings.GROQ_MODEL
    assert kwargs["temperature"] == 0.0
    assert kwargs["stream"] is False
    assert kwargs["messages"] == [
        {"role": "system", "content": "consignes"},
        {"role": "user", "content": "contenu"},
    ]


def test_call_llm_empty_content_returns_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completions = _FakeCompletions(content=None)
    monkeypatch.setattr(llm_client, "_get_client", lambda: _FakeClient(completions))

    assert call_llm("système", "utilisateur") == ""


def test_call_llm_api_error_raises_llm_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completions = _FakeCompletions(
        error=RuntimeError("rate limit / réseau / clé invalide")
    )
    monkeypatch.setattr(llm_client, "_get_client", lambda: _FakeClient(completions))

    with pytest.raises(LlmClientError):
        call_llm("système", "utilisateur")


def test_client_built_once(monkeypatch: pytest.MonkeyPatch) -> None:
    # Le client Groq (coûteux) ne doit être construit qu'une seule fois.
    calls = {"count": 0}

    class _FakeGroqModule:
        @staticmethod
        def Groq(api_key: str, timeout: float) -> _FakeClient:
            calls["count"] += 1
            return _FakeClient(_FakeCompletions())

    monkeypatch.setitem(__import__("sys").modules, "groq", _FakeGroqModule)

    first = llm_client._get_client()
    second = llm_client._get_client()

    assert first is second
    assert calls["count"] == 1
