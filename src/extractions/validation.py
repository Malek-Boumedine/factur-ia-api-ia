"""Validation du JSON de structuration contre le contrat + gestion des inexploitables.

Dernière brique métier de l'epic Structuration : le ``dict facture`` brut produit
par ``structurer.py`` (miroir du sous-ensemble contrat, montants en ``Decimal``,
totaux éventuellement ``null``) est validé contre ``OcrWebhookPayload`` avant de
partir au callback de l'API data. C'est ici qu'on résout la **divergence volontaire**
documentée dans ``prompts.py`` : le schéma LLM autorise ``total_ht`` / ``total_tva``
/ ``total_ttc`` à ``null`` (pour ne pas forcer le modèle à inventer un montant), mais
``OcrWebhookPayload`` les exige non-null. Résolution retenue : un total ``null`` est
ramené à ``Decimal("0")``, laissé à la correction human-in-the-loop côté API data —
on préserve le reste de l'extraction plutôt que de tout jeter.

C'est aussi ici qu'on **pose le marqueur d'échec** ``score_confiance = 0`` que
``confidence.py`` s'interdit (plancher ``0.05`` réservant précisément le ``0`` à ce
cas). Deux formes d'inexploitabilité :

- **contenu vide de sens** : les trois totaux ``null`` **et** aucune ligne → il n'y a
  rien à corriger → ``build_failure_payload`` (renvoyé, pas d'exception : cas normal,
  page blanche / photo ratée) ;
- **échec structurel** : la donnée ne construit pas un ``OcrWebhookPayload`` même
  après coercition → ``PayloadValidationError`` (levée, cohérente avec le patron
  ``LlmStructurationError`` / ``LlmClientError``).

``score_confiance = 0`` reste l'**unique** marqueur d'échec (jamais les totaux à 0,
ambigus — cf. CLAUDE.md). Le ``score_confiance`` fourni en entrée provient de
``compute_confidence`` (toujours > 0) ; ce module ne force ``0`` que sur le chemin
inexploitable — il en est le seul émetteur.

Ce module ne fait *que* valider et décider l'inexploitabilité. Il ne calcule pas la
confiance (``confidence.py``) et ne câble pas le pipeline (orchestration, tâche
suivante) : l'orchestrateur réutilisera ``build_failure_payload`` pour *tous* ses
chemins d'échec (``LlmClientError``, ``LlmStructurationError`` et
``PayloadValidationError``).
"""

from datetime import date
from decimal import Decimal
from typing import Any

from pydantic import ValidationError

from src.callback.schemas import OcrWebhookPayload

# Totaux non-nullables au contrat mais nullables côté LLM : la divergence à résoudre.
_TOTAL_FIELDS = ("total_ht", "total_tva", "total_ttc")


class PayloadValidationError(Exception):
    """Le JSON structuré ne construit pas un ``OcrWebhookPayload`` valide.

    Levée quand la donnée, même après coercition (totaux ``null`` → 0, date illisible
    → ``None``), ne satisfait pas le contrat (ligne sans désignation, montant non
    convertible…). Distincte de ``LlmStructurationError`` (JSON du modèle inexploitable
    en amont). L'orchestrateur du pipeline l'attrape pour produire un résultat d'échec
    (``score_confiance = 0``) via ``build_failure_payload``.
    """


def build_failure_payload(id_document: int) -> OcrWebhookPayload:
    """Construit le payload d'échec canonique (extraction inexploitable).

    Marqueur unique : ``score_confiance = 0``. Les totaux non-nullables du contrat
    sont mis à ``Decimal("0")`` (valeurs de remplissage, jamais interprétées comme
    marqueur), les champs optionnels restent ``None`` / vides. Utilisé aussi bien ici
    (contenu vide de sens) que par l'orchestrateur pour tous ses chemins d'échec.

    Args:
        id_document: identifiant du document concerné (vient de la requête).

    Returns:
        Un ``OcrWebhookPayload`` signalant l'échec (``score_confiance = 0``).
    """
    return OcrWebhookPayload(
        id_document=id_document,
        score_confiance=Decimal("0"),
        total_ht=Decimal("0"),
        total_tva=Decimal("0"),
        total_ttc=Decimal("0"),
        lignes=[],
    )


def _is_inexploitable(facture: dict[str, Any]) -> bool:
    """Indique si l'extraction est vide de sens métier (aucun total ET aucune ligne).

    Conjonction stricte : une facture avec des totaux mais sans lignes (en-tête seul),
    ou avec des lignes mais sans totaux (l'humain les dérive), reste exploitable.
    """
    no_totals = all(facture.get(field) is None for field in _TOTAL_FIELDS)
    lignes = facture.get("lignes")
    no_lignes = not (isinstance(lignes, list) and lignes)
    return no_totals and no_lignes


def _is_iso_date(value: str) -> bool:
    """Vérifie qu'une chaîne est une date ISO (``AAAA-MM-JJ``) parsable."""
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _prepare(facture: dict[str, Any]) -> dict[str, Any]:
    """Coerce le ``dict`` LLM vers le contrat, sans jeter l'extraction.

    - totaux ``null`` (ou absents) → ``Decimal("0")`` (divergence LLM/contrat) ;
    - ``date_emission`` présente mais non parsable → ``None`` (champ nullable au
      contrat : on ne fait pas échouer tout le payload pour une date illisible).

    Les autres champs (SIRET, numéro, IBAN, lignes) sont laissés tels quels : Pydantic
    les valide et coerce en aval.
    """
    prepared = dict(facture)

    for field in _TOTAL_FIELDS:
        if prepared.get(field) is None:
            prepared[field] = Decimal("0")

    date_value = prepared.get("date_emission")
    if isinstance(date_value, str) and not _is_iso_date(date_value):
        prepared["date_emission"] = None

    return prepared


def validate_extraction(
    id_document: int,
    facture: dict[str, Any],
    score_confiance: Decimal,
) -> OcrWebhookPayload:
    """Valide l'extraction structurée et construit le payload contrat à transmettre.

    Si l'extraction est inexploitable (aucun total lisible **et** aucune ligne),
    renvoie le payload d'échec (``score_confiance = 0``) — cas normal, sans exception.
    Sinon, coerce la donnée (totaux ``null`` → 0, date illisible → ``None``) puis
    construit et valide un ``OcrWebhookPayload`` avec le ``score_confiance`` fourni.

    Args:
        id_document: identifiant du document (vient de la requête).
        facture: sous-ensemble « données extraites » issu de ``structure_invoice``
            (montants en ``Decimal``, totaux éventuellement ``null``).
        score_confiance: confiance calculée par ``compute_confidence`` (toujours > 0).
            Ignoré et forcé à ``0`` sur le chemin inexploitable.

    Returns:
        Un ``OcrWebhookPayload`` valide (extraction exploitable) ou le payload d'échec
        (``score_confiance = 0``) si l'extraction est vide de sens.

    Raises:
        PayloadValidationError: la donnée ne construit pas un ``OcrWebhookPayload``
            valide même après coercition (échec structurel).
    """
    if _is_inexploitable(facture):
        return build_failure_payload(id_document)

    prepared = _prepare(facture)
    data = {
        **prepared,
        "id_document": id_document,
        "score_confiance": score_confiance,
    }
    try:
        return OcrWebhookPayload.model_validate(data)
    except ValidationError as exc:
        raise PayloadValidationError(
            "Extraction inexploitable : le JSON structuré ne respecte pas le contrat."
        ) from exc
