"""Tests de la structuration LLM (``structure_invoice``).

Le client Groq (``call_llm``) est mocké : aucun appel réseau réel. On teste la
*logique* de structuration (passage du prompt et du schéma au client, parsing du
JSON en dict, conversion des décimaux, gestion d'erreur), pas le comportement du
modèle Groq lui-même.
"""

import json
from decimal import Decimal
from typing import Any

import pytest
from src.extractions import structurer
from src.extractions.prompts import INVOICE_JSON_SCHEMA, SYSTEM_PROMPT, TypeDocument
from src.extractions.structurer import LlmStructurationError, structure_invoice


def _model_payload(**overrides: Any) -> str:
    """Sérialise une réponse modèle type, surchargeable champ par champ.

    Miroir du schéma LLM plat : champs contrat + ``type_document``.
    """
    payload: dict[str, Any] = {
        "siret_emetteur": "12345678900011",
        "siret_destinataire": "98765432100022",
        "numero_facture": "FA-2026-042",
        "date_emission": "2026-07-06",
        "total_ht": 1000.00,
        "total_tva": 200.00,
        "total_ttc": 1200.00,
        "iban": "FR7630006000011234567890189",
        "lignes": [
            {
                "designation": "Prestation de conseil",
                "quantite": 2,
                "prix_unitaire_ht": 500.00,
                "taux_tva": 20.00,
            }
        ],
        "type_document": "facture",
    }
    payload.update(overrides)
    return json.dumps(payload)


# Réponse JSON type d'un modèle sur une facture bien formée (miroir du schéma).
_VALID_JSON = _model_payload()


def _fake_call_llm(returned: str, recorder: dict[str, Any] | None = None) -> Any:
    """Fabrique un faux ``call_llm`` renvoyant ``returned`` et capturant ses args."""

    def _call(
        system_prompt: str, user_content: str, *, response_format: Any = None
    ) -> str:
        if recorder is not None:
            recorder["system_prompt"] = system_prompt
            recorder["user_content"] = user_content
            recorder["response_format"] = response_format
        return returned

    return _call


def test_structure_invoice_returns_parsed_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(structurer, "call_llm", _fake_call_llm(_VALID_JSON))

    result = structure_invoice("texte brut de facture")

    facture = result["facture"]
    assert facture["numero_facture"] == "FA-2026-042"
    assert facture["siret_emetteur"] == "12345678900011"
    assert facture["lignes"][0]["designation"] == "Prestation de conseil"


def test_structure_invoice_parses_amounts_as_decimal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # parse_float=Decimal : les montants doivent être des Decimal (précision
    # monétaire), pas des float.
    monkeypatch.setattr(structurer, "call_llm", _fake_call_llm(_VALID_JSON))

    result = structure_invoice("texte brut")

    facture = result["facture"]
    assert isinstance(facture["total_ttc"], Decimal)
    assert facture["total_ttc"] == Decimal("1200.00")
    assert isinstance(facture["lignes"][0]["prix_unitaire_ht"], Decimal)


def test_structure_invoice_passes_prompt_and_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Le prompt système, le texte brut et le response_format (schéma strict)
    # doivent être transmis tels quels au client.
    recorder: dict[str, Any] = {}
    monkeypatch.setattr(structurer, "call_llm", _fake_call_llm(_VALID_JSON, recorder))

    structure_invoice("le texte brut extrait")

    assert recorder["system_prompt"] == SYSTEM_PROMPT
    assert recorder["user_content"] == "le texte brut extrait"
    assert recorder["response_format"] is INVOICE_JSON_SCHEMA


def test_structure_invoice_keeps_null_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    # Champs absents → null côté modèle → None côté dict (pas d'invention).
    payload = _model_payload(
        siret_emetteur=None,
        siret_destinataire=None,
        numero_facture="F-1",
        date_emission=None,
        total_ht=None,
        total_tva=None,
        total_ttc=None,
        iban=None,
        lignes=[],
    )
    monkeypatch.setattr(structurer, "call_llm", _fake_call_llm(payload))

    result = structure_invoice("facture minimale")

    facture = result["facture"]
    assert facture["total_ht"] is None
    assert facture["date_emission"] is None
    assert facture["lignes"] == []


def test_structure_invoice_invalid_json_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        structurer, "call_llm", _fake_call_llm("ceci n'est pas du JSON {")
    )

    with pytest.raises(LlmStructurationError):
        structure_invoice("texte brut")


def test_structure_invoice_empty_response_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(structurer, "call_llm", _fake_call_llm(""))

    with pytest.raises(LlmStructurationError):
        structure_invoice("texte brut")


def test_structure_invoice_non_object_json_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Un JSON valide mais qui n'est pas un objet (ex. une liste) est inexploitable.
    monkeypatch.setattr(structurer, "call_llm", _fake_call_llm("[1, 2, 3]"))

    with pytest.raises(LlmStructurationError):
        structure_invoice("texte brut")


@pytest.mark.parametrize(
    ("raw_type", "expected"),
    [
        ("facture", TypeDocument.FACTURE),
        ("devis", TypeDocument.DEVIS),
        ("avoir", TypeDocument.AVOIR),
        ("inconnu", TypeDocument.INCONNU),
    ],
)
def test_structure_invoice_detects_type(
    monkeypatch: pytest.MonkeyPatch, raw_type: str, expected: TypeDocument
) -> None:
    # Chaque valeur de type renvoyée par le modèle est exposée comme TypeDocument.
    payload = _model_payload(type_document=raw_type)
    monkeypatch.setattr(structurer, "call_llm", _fake_call_llm(payload))

    result = structure_invoice("texte brut")

    assert result["type_document"] is expected


def test_structure_invoice_unexpected_type_defaults_to_inconnu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Valeur hors enum (ne devrait pas arriver en mode strict) → INCONNU par défaut.
    payload = _model_payload(type_document="bon_de_commande")
    monkeypatch.setattr(structurer, "call_llm", _fake_call_llm(payload))

    result = structure_invoice("texte brut")

    assert result["type_document"] is TypeDocument.INCONNU


def test_structure_invoice_missing_type_defaults_to_inconnu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Champ absent de la réponse → INCONNU par défaut (pop avec fallback None).
    payload = json.dumps(
        {
            "siret_emetteur": None,
            "siret_destinataire": None,
            "numero_facture": "F-1",
            "date_emission": None,
            "total_ht": None,
            "total_tva": None,
            "total_ttc": None,
            "iban": None,
            "lignes": [],
        }
    )
    monkeypatch.setattr(structurer, "call_llm", _fake_call_llm(payload))

    result = structure_invoice("texte brut")

    assert result["type_document"] is TypeDocument.INCONNU


def test_structure_invoice_separates_type_from_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # type_document ne doit PAS polluer le sous-ensemble contrat (facture).
    monkeypatch.setattr(structurer, "call_llm", _fake_call_llm(_VALID_JSON))

    result = structure_invoice("texte brut")

    assert "type_document" not in result["facture"]
    assert set(result) == {"type_document", "facture"}


def test_structuration_schema_constrains_type_document() -> None:
    # Le schéma LLM ajoute type_document à plat, dans required, avec l'enum complet.
    schema = INVOICE_JSON_SCHEMA["json_schema"]["schema"]
    type_field = schema["properties"]["type_document"]

    assert type_field["enum"] == [t.value for t in TypeDocument]
    assert "type_document" in schema["required"]
