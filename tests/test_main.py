"""Tests de la configuration du logging applicatif (``src.main``).

``logging.basicConfig`` est mocké : on vérifie les paramètres passés (niveau
piloté par ``settings.DEBUG``, horodatage + nom du logger dans le format), pas
l'état global du logging — pytest installe ses propres handlers racine, qui
rendraient un ``basicConfig`` réel sans effet et le test non déterministe.
"""

import logging
from typing import Any

import pytest
from src import main
from src.core.config import settings


def _capture_basic_config(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Mocke ``logging.basicConfig`` et renvoie le dict des paramètres capturés."""
    captured: dict[str, Any] = {}

    def _fake_basic_config(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(logging, "basicConfig", _fake_basic_config)
    return captured


def test_logging_configured_with_info_level_and_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hors mode debug, la config logging demande le niveau INFO (sinon les logs
    du pipeline sont perdus) et un format avec horodatage + nom du logger."""
    captured = _capture_basic_config(monkeypatch)
    monkeypatch.setattr(settings, "DEBUG", False)

    main._configure_logging()

    assert captured["level"] == logging.INFO
    assert "%(asctime)s" in captured["format"]
    assert "%(name)s" in captured["format"]
    assert "%(levelname)s" in captured["format"]


def test_logging_debug_level_when_debug_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``settings.DEBUG`` actif → niveau DEBUG : active les logs de diagnostic
    du pipeline (ex. JSON structuré avant validation)."""
    captured = _capture_basic_config(monkeypatch)
    monkeypatch.setattr(settings, "DEBUG", True)

    main._configure_logging()

    assert captured["level"] == logging.DEBUG
