"""Structuration LLM : texte brut de facture → données structurées (JSON).

Avant-dernière étape du pipeline : le texte brut extrait (pdfplumber ou OCR) est
soumis au modèle Groq, qui renvoie les champs de facture sous forme d'objet JSON
conforme au schéma imposé (structured outputs strict, cf. ``prompts.py``). On
parse ce JSON en ``dict`` Python, on en sépare la suggestion de type et on renvoie
le tout.

Le schéma LLM inclut, à plat, deux champs hors contrat (cf. ``prompts.py``), retirés
du ``dict`` après parsing pour que ``facture`` redevienne le pur miroir du contrat :

- ``type_document`` (suggestion devis/facture/avoir/inconnu), renvoyé à part et
  transmis au callback via le champ optionnel du contrat. La décision finale sur le
  type revient à l'humain (human-in-the-loop côté API data) ;
- ``delai_paiement_jours``, consommé ici : quand le modèle n'a lu aucune date
  d'échéance absolue mais a relevé un délai de paiement (« net 30 »), on dérive
  ``date_echeance = date_emission + délai``. L'arithmétique est faite en Python —
  déterministe et testable — plutôt que confiée au modèle. Une date d'échéance
  absolue lue sur le document est toujours prioritaire sur le délai.

Ce module ne fait *que* la structuration. Il ne valide PAS le résultat contre
``OcrWebhookPayload`` (types, champs requis, cohérence des montants) et ne calcule
PAS le ``score_confiance`` : ce sont des tâches ultérieures distinctes. Il renvoie
donc des ``dict`` bruts, pas un ``OcrWebhookPayload``.

Deux causes d'échec distinctes = deux exceptions distinctes :

- l'appel au modèle échoue (réseau, clé, timeout, rate limit) → ``LlmClientError``
  levée par ``call_llm``, laissée remonter telle quelle ;
- l'appel réussit mais la réponse n'est pas un JSON exploitable →
  ``LlmStructurationError``.

L'orchestrateur du pipeline attrape les deux pour produire l'échec
(``score_confiance = 0``) côté API data.
"""

import json
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from src.extractions.llm_client import call_llm
from src.extractions.prompts import INVOICE_JSON_SCHEMA, SYSTEM_PROMPT, TypeDocument

# Borne haute de plausibilité d'un délai de paiement, en jours. Au-delà d'un an, un
# délai relevé par le modèle est bien plus probablement une erreur de lecture (un
# montant, une référence) qu'un vrai terme contractuel : on ignore le délai plutôt
# que de fabriquer une échéance absurde. Le maximum légal français est de 60 jours
# (90 en jours fin de mois) ; la marge est volontairement large.
_MAX_DELAI_JOURS = 365


class LlmStructurationError(Exception):
    """Réponse du modèle inexploitable : JSON absent, tronqué ou malformé.

    Relevée quand l'appel au modèle a réussi mais que son contenu ne peut pas être
    parsé en JSON (réponse vide, tronquée par une limite de tokens, ou non conforme
    malgré le schéma). Distincte de ``LlmClientError`` (échec de l'appel lui-même).
    L'orchestrateur du pipeline attrape cette exception pour produire un résultat
    d'échec (``score_confiance = 0``) côté API data.
    """


def _as_delai_jours(value: Any) -> int | None:
    """Convertit le délai de paiement brut en nombre de jours exploitable.

    Renvoie ``None`` si le délai est absent, non numérique, négatif ou hors de la
    plage plausible (``_MAX_DELAI_JOURS``) : dans tous ces cas on préfère laisser
    ``date_echeance`` à ``null`` plutôt que d'en dériver une fausse. Le schéma LLM
    impose un entier, mais un modèle peut renvoyer ``30.0`` (parsé en ``Decimal``) —
    on l'accepte s'il est bien entier.
    """
    if isinstance(value, bool):
        return None  # ``bool`` est un ``int`` en Python : jamais un délai
    if isinstance(value, Decimal):
        if value != value.to_integral_value():
            return None
        value = int(value)
    if not isinstance(value, int):
        return None
    if not 0 <= value <= _MAX_DELAI_JOURS:
        return None
    return value


def _derive_date_echeance(facture: dict[str, Any], delai_jours: Any) -> None:
    """Complète ``date_echeance`` depuis le délai de paiement, sur place.

    N'intervient que si le modèle n'a lu **aucune** date d'échéance absolue : une
    date écrite sur le document fait toujours autorité sur un délai. La dérivation
    exige une ``date_emission`` ISO parsable (sans elle, il n'y a rien à quoi
    ajouter le délai) et un délai plausible ; à défaut, ``date_echeance`` reste
    telle quelle (``null``) — on ne devine pas.
    """
    if facture.get("date_echeance") is not None:
        return

    delai = _as_delai_jours(delai_jours)
    if delai is None:
        return

    emission = facture.get("date_emission")
    if not isinstance(emission, str):
        return
    try:
        parsed = date.fromisoformat(emission)
    except ValueError:
        return  # date d'émission illisible : rien de fiable à dériver

    facture["date_echeance"] = (parsed + timedelta(days=delai)).isoformat()


def structure_invoice(raw_text: str) -> dict[str, Any]:
    """Structure le texte brut d'une facture en champs extraits + suggestion de type.

    Soumet ``raw_text`` au modèle Groq avec le prompt système et le schéma strict
    (``INVOICE_JSON_SCHEMA``), puis parse la réponse JSON. Les nombres décimaux
    sont convertis en ``Decimal`` (``parse_float=Decimal``) pour préserver la
    précision monétaire exacte, en vue de la validation ``Decimal`` ultérieure.

    Sépare ensuite les champs hors contrat du sous-ensemble « données extraites » :
    ``type_document`` est extrait du ``dict`` parsé et converti en ``TypeDocument``
    (``INCONNU`` si absent ou valeur inattendue) ; ``delai_paiement_jours`` est
    extrait puis consommé pour dériver ``date_echeance`` (émission + délai) si le
    modèle n'a lu aucune date d'échéance absolue. Le reste devient ``facture``,
    reflet du sous-ensemble « données extraites » de ``OcrWebhookPayload`` (sans
    ``id_document`` ni ``score_confiance``, ni aucun des deux champs hors contrat).
    Hormis cette dérivation, le contenu de ``facture`` n'est ni validé ni complété
    ici : champs manquants, types incohérents ou totaux ``null`` relèvent de la
    tâche validation/score.

    Args:
        raw_text: texte brut de la facture (issu de pdfplumber ou de l'OCR).

    Returns:
        Un ``dict`` à deux clés : ``type_document`` (``TypeDocument``, suggestion IA
        non contraignante, transmise au callback via le champ optionnel du contrat)
        et ``facture`` (``dict`` des champs contrat, montants en ``Decimal``).

    Raises:
        LlmStructurationError: la réponse du modèle n'est pas un JSON exploitable.
        LlmClientError: l'appel au modèle a échoué (propagée par ``call_llm``).
    """
    content = call_llm(
        SYSTEM_PROMPT,
        raw_text,
        response_format=INVOICE_JSON_SCHEMA,
    )

    try:
        data = json.loads(content, parse_float=Decimal)
    except (json.JSONDecodeError, ValueError) as exc:
        raise LlmStructurationError(
            "Réponse du modèle inexploitable : JSON absent ou malformé."
        ) from exc

    if not isinstance(data, dict):
        raise LlmStructurationError(
            "Réponse du modèle inexploitable : objet JSON attendu."
        )

    # Sépare les champs hors contrat du sous-ensemble OcrWebhookPayload : retirés de
    # ``data``, qui redevient le pur miroir contrat.
    raw_type = data.pop("type_document", None)
    delai_jours = data.pop("delai_paiement_jours", None)
    try:
        type_document = TypeDocument(raw_type)
    except ValueError:
        type_document = TypeDocument.INCONNU  # absent ou valeur inattendue → défaut

    # Le délai n'entre pas dans le contrat : il n'y vit que converti en date.
    _derive_date_echeance(data, delai_jours)

    return {"type_document": type_document, "facture": data}
