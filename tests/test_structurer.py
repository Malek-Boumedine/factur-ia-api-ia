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

    Miroir du schéma LLM plat : champs contrat + les deux champs hors contrat
    (``type_document``, ``delai_paiement_jours``).
    """
    payload: dict[str, Any] = {
        "siret_emetteur": "12345678900011",
        "siret_destinataire": "98765432100022",
        "numero_facture": "FA-2026-042",
        "date_emission": "2026-07-06",
        "date_echeance": "2026-08-05",
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
        "delai_paiement_jours": None,
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
    # Les champs hors contrat ne doivent PAS polluer le sous-ensemble contrat.
    monkeypatch.setattr(structurer, "call_llm", _fake_call_llm(_VALID_JSON))

    result = structure_invoice("texte brut")

    assert "type_document" not in result["facture"]
    assert "delai_paiement_jours" not in result["facture"]
    assert set(result) == {"type_document", "facture"}


# --- Date d'échéance -------------------------------------------------------


def test_date_echeance_absolue_extraite(monkeypatch: pytest.MonkeyPatch) -> None:
    # Cas nominal : l'échéance écrite sur le document remonte telle quelle.
    monkeypatch.setattr(structurer, "call_llm", _fake_call_llm(_VALID_JSON))

    result = structure_invoice("texte brut")

    assert result["facture"]["date_echeance"] == "2026-08-05"


def test_date_echeance_distincte_de_l_emission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Anti-régression du bug d'origine : les deux dates sont deux champs distincts,
    # l'échéance n'écrase pas l'émission et n'en est pas une recopie.
    monkeypatch.setattr(structurer, "call_llm", _fake_call_llm(_VALID_JSON))

    facture = structure_invoice("texte brut")["facture"]

    assert facture["date_emission"] == "2026-07-06"
    assert facture["date_echeance"] != facture["date_emission"]


def test_date_echeance_derivee_du_delai(monkeypatch: pytest.MonkeyPatch) -> None:
    # « Paiement à 30 jours » sans date absolue : Python calcule émission + délai.
    payload = _model_payload(date_echeance=None, delai_paiement_jours=30)
    monkeypatch.setattr(structurer, "call_llm", _fake_call_llm(payload))

    result = structure_invoice("facture payable à 30 jours")

    assert result["facture"]["date_echeance"] == "2026-08-05"  # 2026-07-06 + 30 j


def test_date_echeance_absolue_prioritaire_sur_le_delai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Les deux présents : la date lue sur le document fait autorité, pas le calcul.
    payload = _model_payload(date_echeance="2026-09-30", delai_paiement_jours=30)
    monkeypatch.setattr(structurer, "call_llm", _fake_call_llm(payload))

    result = structure_invoice("texte brut")

    assert result["facture"]["date_echeance"] == "2026-09-30"


def test_delai_sans_date_emission_ne_derive_rien(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Sans date d'émission lisible, il n'y a rien à quoi ajouter le délai : on ne
    # devine pas d'échéance.
    payload = _model_payload(
        date_emission=None, date_echeance=None, delai_paiement_jours=30
    )
    monkeypatch.setattr(structurer, "call_llm", _fake_call_llm(payload))

    result = structure_invoice("texte brut")

    assert result["facture"]["date_echeance"] is None


def test_date_emission_illisible_ne_derive_rien(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Date d'émission non ISO (modèle qui n'a pas suivi la consigne) : pas de calcul
    # hasardeux sur une date qu'on ne sait pas parser.
    payload = _model_payload(
        date_emission="06/07/2026", date_echeance=None, delai_paiement_jours=30
    )
    monkeypatch.setattr(structurer, "call_llm", _fake_call_llm(payload))

    result = structure_invoice("texte brut")

    assert result["facture"]["date_echeance"] is None


@pytest.mark.parametrize("delai", [-10, 400, "trente", True, None])
def test_delai_implausible_ignore(monkeypatch: pytest.MonkeyPatch, delai: Any) -> None:
    # Délai négatif, hors plage (> 1 an), non numérique ou booléen : mieux vaut une
    # échéance nulle qu'une date fabriquée.
    payload = _model_payload(date_echeance=None, delai_paiement_jours=delai)
    monkeypatch.setattr(structurer, "call_llm", _fake_call_llm(payload))

    result = structure_invoice("texte brut")

    assert result["facture"]["date_echeance"] is None


def test_delai_decimal_fractionnaire_ignore(monkeypatch: pytest.MonkeyPatch) -> None:
    # Un délai à virgule n'est pas un nombre de jours : on n'en dérive rien.
    payload = _model_payload(date_echeance=None, delai_paiement_jours=30.5)
    monkeypatch.setattr(structurer, "call_llm", _fake_call_llm(payload))

    result = structure_invoice("texte brut")

    assert result["facture"]["date_echeance"] is None


def test_delai_decimal_entier_accepte(monkeypatch: pytest.MonkeyPatch) -> None:
    # Le schéma impose un entier, mais un modèle peut renvoyer 30.0 (parsé Decimal).
    payload = _model_payload(date_echeance=None, delai_paiement_jours=30.0)
    monkeypatch.setattr(structurer, "call_llm", _fake_call_llm(payload))

    result = structure_invoice("texte brut")

    assert result["facture"]["date_echeance"] == "2026-08-05"


def test_absence_totale_d_echeance_reste_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Facture payable comptant : ni date ni délai → null, sans rien inventer.
    payload = _model_payload(date_echeance=None, delai_paiement_jours=None)
    monkeypatch.setattr(structurer, "call_llm", _fake_call_llm(payload))

    result = structure_invoice("texte brut")

    assert result["facture"]["date_echeance"] is None


def test_system_prompt_teaches_french_amount_reading() -> None:
    # Anti-régression (bug constaté : « 1 850,00 » extrait 850.0) : le prompt doit
    # garder les consignes de LECTURE du format français des montants — espace =
    # séparateur de milliers à agréger, virgule = séparateur décimal — avec les
    # exemples concrets et l'avertissement sur la troncature avant un espace.
    # Espaces normalisés : le prompt est replié à 88 colonnes, un exemple peut
    # être coupé par un retour à la ligne.
    prompt = " ".join(SYSTEM_PROMPT.split())
    assert "séparateur de milliers" in prompt
    assert "séparateur décimal" in prompt
    assert "« 1 850,00 » vaut `1850.00`" in prompt
    assert "« 12 345,67 » vaut `12345.67`" in prompt
    assert "« 850,00 » vaut `850.00`" in prompt
    assert "Ne tronque JAMAIS" in prompt


def test_system_prompt_asks_for_date_echeance() -> None:
    # Anti-régression (bug constaté : échéance jamais extraite car jamais demandée).
    # Le prompt doit réclamer la date d'échéance ET la distinguer explicitement de la
    # date d'émission — la confusion des deux est le mode d'échec principal.
    # Espaces normalisés : le prompt est replié à 88 colonnes.
    prompt = " ".join(SYSTEM_PROMPT.split())
    assert "`date_echeance`" in prompt
    assert "DATE LIMITE DE PAIEMENT" in prompt
    assert "NE CONFONDS PAS les deux dates" in prompt
    assert "à régler avant le" in prompt
    # Formats français en entrée et cas du délai.
    assert "« 15 mars 2026 » vaut `2026-03-15`" in prompt
    assert "le JOUR précède le MOIS" in prompt
    assert "`delai_paiement_jours`" in prompt
    assert "net 30" in prompt


def test_schema_includes_date_echeance_in_contract_subset() -> None:
    # ``date_echeance`` appartient au miroir du contrat (elle part au callback),
    # contrairement au délai qui reste hors contrat.
    schema = INVOICE_JSON_SCHEMA["json_schema"]["schema"]

    assert schema["properties"]["date_echeance"] == {"type": ["string", "null"]}
    assert "date_echeance" in schema["required"]


def test_structuration_schema_constrains_delai_paiement() -> None:
    # Le délai est ajouté à plat, entier nullable, et requis (contrainte strict mode).
    schema = INVOICE_JSON_SCHEMA["json_schema"]["schema"]

    assert schema["properties"]["delai_paiement_jours"] == {"type": ["integer", "null"]}
    assert "delai_paiement_jours" in schema["required"]


def test_structuration_schema_constrains_type_document() -> None:
    # Le schéma LLM ajoute type_document à plat, dans required, avec l'enum complet.
    schema = INVOICE_JSON_SCHEMA["json_schema"]["schema"]
    type_field = schema["properties"]["type_document"]

    assert type_field["enum"] == [t.value for t in TypeDocument]
    assert "type_document" in schema["required"]
