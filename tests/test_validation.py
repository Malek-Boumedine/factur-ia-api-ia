"""Tests de la validation du JSON de sortie et de la gestion des inexploitables.

On teste ``validate_extraction`` (construction/validation du ``OcrWebhookPayload``,
résolution de la divergence null sur les totaux, décision d'inexploitabilité) et les
primitives associées (``build_failure_payload``, ``PayloadValidationError``). Aucun
appel réseau : logique pure autour de Pydantic.
"""

from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from src.extractions.validation import (
    PayloadValidationError,
    build_failure_payload,
    validate_extraction,
)

_ID = 42
_SCORE = Decimal("0.87")


def _facture(**overrides: Any) -> dict[str, Any]:
    """Sous-ensemble contrat nominal (miroir de ``structure_invoice``)."""
    facture: dict[str, Any] = {
        "siret_emetteur": "73282932000074",
        "siret_destinataire": "55210055400021",
        "numero_facture": "FA-2026-042",
        "date_emission": "2026-01-15",
        "total_ht": Decimal("1000.00"),
        "total_tva": Decimal("200.00"),
        "total_ttc": Decimal("1200.00"),
        "iban": "FR7630006000011234567890189",
        "lignes": [
            {
                "designation": "Prestation de conseil",
                "quantite": 2,
                "prix_unitaire_ht": Decimal("500.00"),
                "taux_tva": Decimal("20.00"),
            }
        ],
    }
    facture.update(overrides)
    return facture


# --- Payload valide (nominal) ----------------------------------------------


def test_payload_valide_nominal() -> None:
    """Extraction complète et cohérente → payload valide, champs et score préservés."""
    payload = validate_extraction(_ID, _facture(), _SCORE)

    assert payload.id_document == _ID
    assert payload.score_confiance == _SCORE
    assert payload.total_ttc == Decimal("1200.00")
    assert payload.date_emission == date(2026, 1, 15)
    assert payload.siret_emetteur == "73282932000074"
    assert len(payload.lignes) == 1
    assert payload.lignes[0].designation == "Prestation de conseil"


def test_score_confiance_preserve() -> None:
    """Le score fourni (issu de compute_confidence) est transmis tel quel."""
    payload = validate_extraction(_ID, _facture(), Decimal("0.05"))
    assert payload.score_confiance == Decimal("0.05")


# --- Divergence null sur les totaux ----------------------------------------


def test_total_null_ramene_a_zero() -> None:
    """Un total illisible (null) est ramené à 0, l'extraction reste exploitable."""
    payload = validate_extraction(_ID, _facture(total_tva=None), _SCORE)

    assert payload.total_tva == Decimal("0")
    # Les autres totaux et le reste sont préservés, score > 0 (pas un échec).
    assert payload.total_ht == Decimal("1000.00")
    assert payload.score_confiance == _SCORE


def test_totaux_null_mais_lignes_presentes_exploitable() -> None:
    """Tous les totaux null mais des lignes présentes → exploitable (totaux à 0)."""
    payload = validate_extraction(
        _ID,
        _facture(total_ht=None, total_tva=None, total_ttc=None),
        _SCORE,
    )
    assert payload.total_ht == Decimal("0")
    assert payload.total_ttc == Decimal("0")
    assert payload.score_confiance == _SCORE  # pas le marqueur d'échec
    assert len(payload.lignes) == 1


def test_date_illisible_ramenee_a_none() -> None:
    """Une date non parsable est ramenée à None sans faire échouer le payload."""
    payload = validate_extraction(_ID, _facture(date_emission="15/01/2026"), _SCORE)
    assert payload.date_emission is None
    assert payload.score_confiance == _SCORE


# --- Lignes vides ----------------------------------------------------------


def test_lignes_vides_mais_totaux_presents_exploitable() -> None:
    """Aucune ligne mais des totaux présents (en-tête seul) → exploitable."""
    payload = validate_extraction(_ID, _facture(lignes=[]), _SCORE)
    assert payload.lignes == []
    assert payload.score_confiance == _SCORE
    assert payload.total_ttc == Decimal("1200.00")


# --- Extraction inexploitable (score_confiance = 0) ------------------------


def test_inexploitable_aucun_total_aucune_ligne() -> None:
    """Aucun total ET aucune ligne → inexploitable, marqueur score_confiance=0."""
    payload = validate_extraction(
        _ID,
        _facture(total_ht=None, total_tva=None, total_ttc=None, lignes=[]),
        _SCORE,
    )
    assert payload.score_confiance == Decimal("0")
    assert payload.id_document == _ID
    # Totaux à 0 (remplissage), jamais utilisés comme marqueur.
    assert payload.total_ht == Decimal("0")


def test_inexploitable_ignore_le_score_fourni() -> None:
    """Sur le chemin inexploitable, le score fourni est ignoré et forcé à 0."""
    payload = validate_extraction(
        _ID,
        _facture(total_ht=None, total_tva=None, total_ttc=None, lignes=[]),
        Decimal("0.99"),
    )
    assert payload.score_confiance == Decimal("0")


def test_facture_entierement_vide_inexploitable() -> None:
    """Extraction entièrement vide → inexploitable (score=0)."""
    facture_vide: dict[str, Any] = {
        "siret_emetteur": None,
        "siret_destinataire": None,
        "numero_facture": None,
        "date_emission": None,
        "total_ht": None,
        "total_tva": None,
        "total_ttc": None,
        "iban": None,
        "lignes": [],
    }
    payload = validate_extraction(_ID, facture_vide, _SCORE)
    assert payload.score_confiance == Decimal("0")


# --- Erreur de validation Pydantic -----------------------------------------


def test_ligne_sans_designation_leve_validation_error() -> None:
    """Une ligne sans désignation viole le contrat → PayloadValidationError."""
    lignes = [
        {
            "quantite": 1,
            "prix_unitaire_ht": Decimal("100.00"),
            "taux_tva": Decimal("20.00"),
        }
    ]
    with pytest.raises(PayloadValidationError):
        validate_extraction(_ID, _facture(lignes=lignes), _SCORE)


def test_total_non_convertible_leve_validation_error() -> None:
    """Un total présent mais non convertible en Decimal → PayloadValidationError."""
    with pytest.raises(PayloadValidationError):
        validate_extraction(_ID, _facture(total_ht="illisible"), _SCORE)


# --- Payload d'échec (primitive) -------------------------------------------


def test_build_failure_payload_forme() -> None:
    """Le payload d'échec porte le marqueur score=0, totaux à 0, listes vides."""
    payload = build_failure_payload(_ID)

    assert payload.id_document == _ID
    assert payload.score_confiance == Decimal("0")
    assert payload.total_ht == Decimal("0")
    assert payload.total_tva == Decimal("0")
    assert payload.total_ttc == Decimal("0")
    assert payload.lignes == []
    assert payload.siret_emetteur is None
    assert payload.date_emission is None
