"""Structuration LLM : texte brut de facture → données structurées (JSON).

Avant-dernière étape du pipeline : le texte brut extrait (pdfplumber ou OCR) est
soumis au modèle Groq, qui renvoie les champs de facture sous forme d'objet JSON
conforme au schéma imposé (structured outputs strict, cf. ``prompts.py``). On
parse ce JSON en ``dict`` Python, on en sépare la suggestion de type et on renvoie
le tout.

Le schéma LLM inclut, à plat, un champ ``type_document`` (suggestion devis/facture/
avoir/inconnu) qui ne fait PAS partie du contrat ``OcrWebhookPayload``. Après
parsing, on l'extrait du ``dict`` et on renvoie deux blocs distincts :
``type_document`` (la suggestion IA, non contraignante, non transmise au callback)
et ``facture`` (le sous-ensemble contrat, débarrassé de ``type_document``). La
décision finale sur le type revient à l'humain (human-in-the-loop côté API data).

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
from decimal import Decimal
from typing import Any

from src.extractions.llm_client import call_llm
from src.extractions.prompts import INVOICE_JSON_SCHEMA, SYSTEM_PROMPT, TypeDocument


class LlmStructurationError(Exception):
    """Réponse du modèle inexploitable : JSON absent, tronqué ou malformé.

    Relevée quand l'appel au modèle a réussi mais que son contenu ne peut pas être
    parsé en JSON (réponse vide, tronquée par une limite de tokens, ou non conforme
    malgré le schéma). Distincte de ``LlmClientError`` (échec de l'appel lui-même).
    L'orchestrateur du pipeline attrape cette exception pour produire un résultat
    d'échec (``score_confiance = 0``) côté API data.
    """


def structure_invoice(raw_text: str) -> dict[str, Any]:
    """Structure le texte brut d'une facture en champs extraits + suggestion de type.

    Soumet ``raw_text`` au modèle Groq avec le prompt système et le schéma strict
    (``INVOICE_JSON_SCHEMA``), puis parse la réponse JSON. Les nombres décimaux
    sont convertis en ``Decimal`` (``parse_float=Decimal``) pour préserver la
    précision monétaire exacte, en vue de la validation ``Decimal`` ultérieure.

    Sépare ensuite la suggestion de type (hors contrat) du sous-ensemble contrat :
    ``type_document`` est extrait du ``dict`` parsé et converti en ``TypeDocument``
    (``INCONNU`` si absent ou valeur inattendue) ; le reste devient ``facture``,
    reflet du sous-ensemble « données extraites » de ``OcrWebhookPayload`` (sans
    ``id_document`` ni ``score_confiance``, et désormais sans ``type_document``).
    Le contenu de ``facture`` n'est ni validé ni complété ici : champs manquants,
    types incohérents ou totaux ``null`` relèvent de la tâche validation/score.

    Args:
        raw_text: texte brut de la facture (issu de pdfplumber ou de l'OCR).

    Returns:
        Un ``dict`` à deux clés : ``type_document`` (``TypeDocument``, suggestion IA
        non contraignante, non transmise au callback) et ``facture`` (``dict`` des
        champs contrat, montants en ``Decimal``).

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

    # Sépare la suggestion IA (hors contrat) du sous-ensemble OcrWebhookPayload :
    # ``type_document`` est retiré de ``data``, qui redevient le pur miroir contrat.
    raw_type = data.pop("type_document", None)
    try:
        type_document = TypeDocument(raw_type)
    except ValueError:
        type_document = TypeDocument.INCONNU  # absent ou valeur inattendue → défaut

    return {"type_document": type_document, "facture": data}
