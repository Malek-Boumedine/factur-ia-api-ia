"""Tests de la configuration du logging applicatif (``src.main``).

``logging.basicConfig`` est mocké : on vérifie les paramètres passés (niveau
INFO, horodatage + nom du logger dans le format), pas l'état global du logging —
pytest installe ses propres handlers racine, qui rendraient un ``basicConfig``
réel sans effet et le test non déterministe.
"""

import logging
from typing import Any

import pytest
from src import main


def test_logging_configured_with_info_level_and_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La config logging demande le niveau INFO (sinon les logs du pipeline sont
    perdus) et un format avec horodatage + nom du logger (diagnostic en prod)."""
    captured: dict[str, Any] = {}

    def _fake_basic_config(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(logging, "basicConfig", _fake_basic_config)

    main._configure_logging()

    assert captured["level"] == logging.INFO
    assert "%(asctime)s" in captured["format"]
    assert "%(name)s" in captured["format"]
    assert "%(levelname)s" in captured["format"]
